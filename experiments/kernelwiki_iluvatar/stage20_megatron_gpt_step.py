"""Probe one complete pinned Megatron GPT training step on BI-V150.

Only process-local compatibility mappings are applied. The test uses synthetic
token IDs and the local (non-Transformer-Engine) GPT layer spec.
"""

import argparse
import hashlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

from stage19_megatron_route_step import load_megatron


def run(args, report):
    _, spec_factory, config_type = load_megatron(args.megatron_root)
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29571")
    dist.init_process_group("nccl", rank=0, world_size=1)
    from megatron.core import parallel_state
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(args.seed)
    config = config_type(
        num_layers=1,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        normalization="RMSNorm",
        sequence_parallel=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )
    spec = spec_factory(normalization="RMSNorm")
    model = GPTModel(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=args.vocab_size,
        max_sequence_length=args.sequence_length,
        position_embedding_type="learned_absolute",
    ).cuda()
    report["model"] = {
        "class": type(model).__name__,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "layer_spec": str(spec.module),
    }
    calls = []

    def norm_hook(module, inputs):
        tensor = inputs[0]
        calls.append({
            "module": type(module).__name__,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "stride": list(tensor.stride()),
            "contiguous": tensor.is_contiguous(),
        })

    handles = [m.register_forward_pre_hook(norm_hook) for m in model.modules()
               if type(m).__name__ == "RMSNorm"]
    if not handles:
        raise RuntimeError("model contains no local RMSNorm")
    torch.manual_seed(args.seed)
    input_ids = torch.randint(args.vocab_size, (args.micro_batch_size, args.sequence_length), device="cuda")
    position_ids = torch.arange(args.sequence_length, device="cuda").unsqueeze(0).expand_as(input_ids)
    labels = torch.randint(args.vocab_size, input_ids.shape, device="cuda")
    attention_mask = torch.triu(torch.ones((1, 1, args.sequence_length, args.sequence_length),
                                           device="cuda", dtype=torch.bool), diagonal=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = model(input_ids, position_ids, attention_mask, labels=labels)
    loss = output.float().mean()
    loss.backward()
    optimizer.step()
    end.record()
    torch.cuda.synchronize()
    report["step"] = {
        "input_shape": list(input_ids.shape),
        "output_shape": list(output.shape),
        "loss": float(loss.item()),
        "elapsed_ms_first_step": float(start.elapsed_time(end)),
        "finite": bool(torch.isfinite(loss).item()),
        "norm_calls": calls,
    }
    report["status"] = "passed" if report["step"]["finite"] else "failed"
    for handle in handles:
        handle.remove()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--megatron-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20001)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "scope": "one-layer pinned Megatron GPT local-spec step with synthetic token IDs; first step includes initialization/warmup",
    }
    try:
        run(args, report)
    except Exception as exc:
        report.update(status="error", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(report["status"], args.output)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
