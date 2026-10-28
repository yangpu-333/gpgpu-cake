"""CoreX Torch2.7 AdamW API port: replace only the two moment factories.

Native AdamW inherits Adam._init_group in this runtime. Native update functions,
decoupled weight decay, group lists, counters, and all arithmetic stay intact.
"""
import ast
import hashlib
import inspect
from pathlib import Path
import textwrap
import weakref

ADAMW_FILE_SHA = '3112d8799ef9ab608b855a5708087d5576971463923c41d4a6a227006871f551'
NATIVE_INITIALIZER_SHA = '7064f09244564d986bd1070ec306b63c1be6f3f6faa719bf0f7baac24ca5342b'


def install(torch, events):
    import sft_local_moments_v4_20261023 as frozen
    from corex_process_guard import check_live
    cls = torch.optim.AdamW
    assert hashlib.sha256(Path(inspect.getsourcefile(cls)).read_bytes()).hexdigest() == ADAMW_FILE_SHA
    original = cls._init_group
    native = inspect.unwrap(original)
    initializer_text = textwrap.dedent(inspect.getsource(native))
    assert hashlib.sha256(initializer_text.encode()).hexdigest() == NATIVE_INITIALIZER_SHA
    if native.__module__ != 'torch.optim.adam':
        raise ValueError('Require the inspected native Torch2.7 inherited initializer')
    if tuple(inspect.signature(native).parameters) != (
            'self', 'group', 'params_with_grad', 'grads', 'exp_avgs',
            'exp_avg_sqs', 'max_exp_avg_sqs', 'state_steps'):
        raise ValueError('Unsupported native initializer API')
    tree = ast.parse(initializer_text)
    reference_tree = ast.dump(tree, include_attributes=False)
    method = tree.body[0]
    assert isinstance(method, ast.FunctionDef) and method.name == '_init_group'
    assert not method.decorator_list
    target_expression = ast.dump(ast.parse('torch.zeros_like(p, memory_format=torch.preserve_format)', mode='eval').body,
                                 include_attributes=False)
    changed = []
    replacements = []
    for assignment in ast.walk(method):
        if not isinstance(assignment, ast.Assign) or len(assignment.targets) != 1:
            continue
        target = assignment.targets[0]
        if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                and target.value.id == 'state' and isinstance(target.slice, ast.Constant)
                and target.slice.value in ('exp_avg', 'exp_avg_sq')):
            assert ast.dump(assignment.value, include_attributes=False) == target_expression
            field = target.slice.value
            replacements.append((assignment, assignment.value))
            assignment.value = ast.copy_location(ast.Call(
                func=ast.Name(id='_jiuge_CoreX_zero', ctx=ast.Load()),
                args=[ast.Name(id='p', ctx=ast.Load()), ast.Constant(field)], keywords=[]), assignment.value)
            changed.append(field)
    assert sorted(changed) == ['exp_avg', 'exp_avg_sq']
    # Reversing those two substitutions MUST recover the entire native AST.
    values = [assignment.value for assignment, _ in replacements]
    for assignment, old in replacements:
        assignment.value = old
    assert ast.dump(tree, include_attributes=False) == reference_tree
    for (assignment, _), value in zip(replacements, values):
        assignment.value = value
    ast.fix_missing_locations(tree)
    node = frozen.node_for_current_cpus()
    tracked = {}

    def remember(value):
        identity = id(value)
        def discard(reference):
            if tracked.get(identity) is reference:
                tracked.pop(identity, None)
        tracked[identity] = weakref.ref(value, discard)

    def known(value):
        reference = tracked.get(id(value))
        return reference is not None and reference() is value

    def factory(parameter, field):
        if parameter.device.type != 'cpu':
            return torch.zeros_like(parameter, memory_format=torch.preserve_format)
        check_live(parameter.numel()*parameter.element_size())
        placed, receipt = frozen.create_local_zero(parameter, node, torch)
        remember(placed)
        events.append({'field': field, 'elements': placed.numel(), **receipt})
        return placed

    namespace = dict(native.__globals__)
    namespace['_jiuge_CoreX_zero'] = factory
    exec(compile(tree, '<CoreX-native-AdamW-two-moment-factories>', 'exec'), namespace)
    derived = namespace['_init_group']
    if native is not original:
        if not getattr(original, '_torchdynamo_disable', False):
            raise ValueError('Unknown native initializer wrapper')
        derived = native.__globals__['torch']._dynamo.disable(derived)

    def wrapped(opt, group, *args, **kwargs):
        if group['amsgrad']:
            raise ValueError('The existing SFT candidate excludes AMSGrad')
        for parameter in group['params']:
            if parameter.grad is None or parameter.device.type != 'cpu':
                continue
            state = opt.state[parameter]
            if not state:
                continue
            for field in ('exp_avg', 'exp_avg_sq'):
                value = state[field]
                if known(value):
                    continue
                check_live(value.numel()*value.element_size())
                placed, receipt = frozen.relocate_moment(value, node, torch)
                state[field] = placed
                remember(placed)
                events.append({'field': field, 'elements': placed.numel(), **receipt,
                               'initialization_path': 'native_resume_copy',
                               'native_source_moment_created': True,
                               'copied_source_moment_bytes': placed.numel()*placed.element_size()})
        return derived(opt, group, *args, **kwargs)

    wrapped._jiuge_CoreX_only_two_factory_port = True
    cls._init_group = wrapped

    def uninstall():
        assert cls._init_group is wrapped
        cls._init_group = original
    return uninstall
