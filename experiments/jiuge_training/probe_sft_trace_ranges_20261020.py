"""CPU-only exact numerical/annotation regression; no CUDA allocation."""
import importlib
import json
from pathlib import Path
import torch
from sft_trace_ranges_20261020 import install


def main():
    torch.set_num_threads(1)
    torch.manual_seed(123)
    params = [torch.randn(19,13),torch.randn(7)]
    gradients = [[torch.randn_like(p) for p in params] for _ in range(3)]

    def trial():
        values = [p.clone().requires_grad_() for p in params]
        opt = torch.optim.AdamW(values,lr=1e-5,betas=(.9,.95),eps=1e-8,weight_decay=.1)
        for grads in gradients:
            for p,g in zip(values,grads):
                p.grad = g.clone()
            opt.step()
        return values, opt.state_dict()

    first, a = trial()
    counts = install(torch)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        second, b = trial()
    assert all(torch.equal(p,q) for p,q in zip(first,second))
    assert a['param_groups'] == b['param_groups']
    for key,state in a['state'].items():
        assert state.keys() == b['state'][key].keys()
        assert all(torch.equal(v,b['state'][key][name]) for name,v in state.items())
    events = [e for e in profile.events() if e.name == 'jiuge/cpu_adamw_math']
    assert counts['jiuge/cpu_adamw_math'] == len(events) == 3
    assert not torch.cuda.is_initialized()
    result = {'schema':'jiuge7b-sft-trace-range-probe-v1','passed':True,
              'parameter_and_complete_adamw_state_bitwise_equal':True,'updates':3,
              'functional_range_count':len(events),'cuda_initialized':False,'torch':torch.__version__}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
