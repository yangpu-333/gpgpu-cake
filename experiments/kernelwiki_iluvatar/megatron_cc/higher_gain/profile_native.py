"""Native Megatron step profile for hotspot selection, never promotion timing."""
import argparse
import contextlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--cuda-profile', action='store_true')
args = parser.parse_args()
if args.output.exists():
    parser.error('immutable output already exists')
import torch
import torch.distributed as dist
from stage19_megatron_route_step import load_megatron
parts = load_megatron(Path('/private/atrex-megatron/src/megatron-lm'))
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
import os
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29831'
dist.init_process_group('nccl', rank=0, world_size=1)
parallel_state.initialize_model_parallel()
model_parallel_cuda_manual_seed(24001)
torch.manual_seed(24001)
config = parts[2](num_layers=4, hidden_size=512, num_attention_heads=8,
    normalization='RMSNorm', layernorm_epsilon=1e-5,
    attention_dropout=0.0, hidden_dropout=0.0, add_bias_linear=False,
    sequence_parallel=False)
model = GPTModel(config=config, transformer_layer_spec=parts[1](normalization='RMSNorm'),
    vocab_size=1024, max_sequence_length=128, position_embedding_type='learned_absolute').cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
generator = torch.Generator().manual_seed(24102)
tokens = torch.randint(1024, (2,128), generator=generator).cuda()
labels = torch.randint(1024, (2,128), generator=generator).cuda()
positions = torch.arange(128, device='cuda').unsqueeze(0).expand_as(tokens)
mask = torch.triu(torch.ones((1,1,128,128), dtype=torch.bool, device='cuda'), 1)
original_loss = model.compute_language_model_loss
def loss_route(*a, **kw):
    with torch.autograd.profiler.record_function('native_vocab_parallel_cross_entropy'):
        return original_loss(*a, **kw)
model.compute_language_model_loss = loss_route
def step():
    optimizer.zero_grad(set_to_none=True)
    with torch.autograd.profiler.record_function('native_forward_and_loss'):
        loss = model(tokens, positions, mask, labels=labels).float().mean()
    with torch.autograd.profiler.record_function('native_backward'):
        loss.backward()
    with torch.autograd.profiler.record_function('native_sgd'):
        optimizer.step()
for _ in range(5):
    step()
torch.cuda.synchronize()
with torch.autograd.profiler.profile(use_cuda=args.cuda_profile, record_shapes=True) as prof:
    for _ in range(5):
        step()
    torch.cuda.synchronize()
rows=[]
for event in prof.key_averages(group_by_input_shape=True):
    rows.append({'name':event.key, 'count':event.count, 'shapes':event.input_shapes,
        'self_cpu_us':event.self_cpu_time_total, 'cpu_total_us':event.cpu_time_total,
        'self_device_us':getattr(event,'self_device_time_total',0.0),
        'device_total_us':getattr(event,'device_time_total',0.0)})
rows.sort(key=lambda row:row['self_cpu_us'], reverse=True)
result={'scope':'diagnostic profiler with overhead; not promotion throughput or exclusive GPU attribution',
    'profiled_native_steps':5, 'cuda_profile':args.cuda_profile,
    'configuration':{'layers':4,'hidden':512,'heads':8,'sequence':128,'micro_batch':2,'vocab':1024},
    'events':rows}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'top_self_cpu':rows[:15], 'loss_scope':[r for r in rows if 'native_vocab' in r['name']]}),flush=True)
parallel_state.destroy_model_parallel()
dist.destroy_process_group()
