"""Observe native checkpoint restoration and its next actual HDO update.

Native Megatron remains responsible for saving/loading optimizer, data and RNG.
This observer activates only after setup/load, so checkpoint initialization's
dummy optimizer step cannot be mistaken for a training update.
"""
import json
from pinning_full_audit import FullAudit


def optimizer_leaves(optimizer):
    pending, seen, leaves = [optimizer], set(), []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if getattr(current, 'chained_optimizers', None):
            pending.extend(current.chained_optimizers)
        elif hasattr(current, 'optimizer'):
            pending.append(current.optimizer)
        else:
            leaves.append(current)
    return leaves


class ResumeAudit:
    def __init__(self, campaign, rank):
        self.campaign, self.rank = campaign, rank
        self.before = self.after = None
        self.opt = None

    def after_setup(self, optimizer, iteration):
        import megatron.core.optimizer.cpu_offloading.hybrid_optimizer as module
        leaves = optimizer_leaves(optimizer)
        matches = [value for value in leaves if isinstance(value, module.HybridDeviceOptimizer)]
        if iteration != 2 or len(matches) != 1:
            raise ValueError('Expected one actual HDO restored at native iteration 2')
        self.opt = matches[0]
        self.before = FullAudit(self.campaign, 'resume-restored', self.rank)
        self.before.step = 2
        self.before.save_hdo(self.opt, 'after')
        self.before.close()
        self.before.flush()
        self.after = FullAudit(self.campaign, 'resume-next-step', self.rank)
        self.after.step = 2
        original = module.HybridDeviceOptimizer.step
        def observed(opt, *values, **keywords):
            if opt is not self.opt or self.after.step != 2:
                raise ValueError('Unexpected optimizer or additional resumed update')
            self.after.step += 1
            self.after.save_hdo(opt, 'before')
            result = original(opt, *values, **keywords)
            self.after.save_hdo(opt, 'after')
            return result
        module.HybridDeviceOptimizer.step = observed

    def record_loss(self, result):
        import torch
        if self.after is None or self.after.step != 3:
            raise ValueError('Actual resumed HDO update missing')
        for name, value in result[0].items():
            if isinstance(value, (tuple, list)):
                for index, component in enumerate(value):
                    self.after.tensor(f'step-3/loss/{name}/{index}', torch.as_tensor(component))
            else:
                self.after.tensor(f'step-3/loss/{name}', torch.as_tensor(value))
        self.after.flush()

    def finish(self, success, completed_steps):
        if self.after:
            self.after.close()
            self.after.flush()
        rows = []
        for audit, step in ((self.before, 2), (self.after, 3)):
            if audit is None:
                continue
            expected = {name for name in audit.reference if name.startswith(f'step-{step}/')}
            if step == 2:
                expected = {name for name in expected if not name.endswith('/gradient') and '/loss/' not in name}
            actual = {row['field'] for row in audit.rows}
            rows.append({'arm': audit.arm, 'fields': len(actual), 'expected_fields': len(expected),
                         'complete': actual == expected, 'numerical_passed': all(r['passed'] for r in audit.rows),
                         'max_abs': max((r['max_abs'] for r in audit.rows), default=0.0)})
        receipt = {'schema': 'jiuge7b-native-checkpoint-resume-audit-v1', 'rank': self.rank,
                   'restored_iteration': 2, 'completed_steps': completed_steps, 'runs': rows,
                   'passed': success and completed_steps == [3] and len(rows) == 2 and
                             all(r['complete'] and r['numerical_passed'] for r in rows),
                   'native_checkpoint_loader_used': True, 'formal_performance_passed': False}
        with (self.campaign / f'resume-rank-{self.rank}-completion.json').open('x') as stream:
            json.dump(receipt, stream, indent=2)
        if success and not receipt['passed']:
            raise AssertionError('Full real-model checkpoint resume comparison did not pass')
