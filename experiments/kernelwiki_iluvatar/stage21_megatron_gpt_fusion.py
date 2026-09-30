"""Compare one local Megatron GPT step with a BI-V150 residual/RMSNorm fusion.

The process-local adapter fuses the attention BDA residual add with the following
pre-MLP RMSNorm for a one-layer, dropout-free GPT. Other model operations stay
in Megatron. Synthetic token IDs are used; this is not production throughput.
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
import traceback
import types
from datetime import datetime, timezone
from pathlib import Path

import torch

from stage19_megatron_route_step import FusedResidualRMSNorm, load_megatron


def run_step(model, inputs, optimizer, *, timed=False):
    optimizer.zero_grad(set_to_none=True)
    if timed:
        torch.cuda.synchronize()
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
    output = model(*inputs[:3], labels=inputs[3])
    loss = output.float().mean()
    loss.backward()
    optimizer.step()
    if timed:
        end.record()
        torch.cuda.synchronize()
        return output.detach(), loss.detach(), float(start.elapsed_time(end))
    return output.detach(), loss.detach(), None


def install_candidate(model):
    layer = model.decoder.layers[0]
    norm = layer.pre_mlp_layernorm
    if type(norm).__name__ != "RMSNorm":
        raise RuntimeError(f"unexpected pre-MLP norm {type(norm).__name__}")
    original_factory = layer.self_attn_bda
    count = {"calls": 0}

    def candidate_factory(training, fused):
        original = original_factory(training, fused)

        def candidate_bda(output_with_bias, residual, dropout):
            if dropout != 0 or not training or fused:
                return original(output_with_bias, residual, dropout)
            x, bias = output_with_bias
            if bias is not None:
                x = x + bias
            y, summed = FusedResidualRMSNorm.apply(x.contiguous(), residual.contiguous(), norm.weight)
            layer._candidate_norm_output = y
            count["calls"] += 1
            return summed

        return candidate_bda

    def candidate_norm_forward(self, hidden_states):
        if not hasattr(layer, "_candidate_norm_output"):
            raise RuntimeError("candidate BDA did not prepare norm output")
        result = layer._candidate_norm_output
        del layer._candidate_norm_output
        return result

    layer.self_attn_bda = candidate_factory
    norm.forward = types.MethodType(candidate_norm_forward, norm)
    return count


def run(args, report):
    _, spec_factory, config_type = load_megatron(args.megatron_root)
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29572")
    dist.init_process_group("nccl", rank=0, world_size=1)
    parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(args.seed)
    torch.manual_seed(args.seed)
    config = config_type(
        num_layers=1, hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        normalization="RMSNorm", sequence_parallel=False,
        attention_dropout=0.0, hidden_dropout=0.0,
    )
    reference = GPTModel(
        config=config, transformer_layer_spec=spec_factory(normalization="RMSNorm"),
        vocab_size=args.vocab_size, max_sequence_length=args.sequence_length,
        position_embedding_type="learned_absolute",
    ).cuda()
    candidate = GPTModel(
        config=config, transformer_layer_spec=spec_factory(normalization="RMSNorm"),
        vocab_size=args.vocab_size, max_sequence_length=args.sequence_length,
        position_embedding_type="learned_absolute",
    ).cuda()
    candidate.load_state_dict(reference.state_dict())
    counter = install_candidate(candidate)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    token_shape = (args.micro_batch_size, args.sequence_length)
    tokens = torch.randint(args.vocab_size, token_shape, generator=generator).cuda()
    positions = torch.arange(args.sequence_length, device="cuda").unsqueeze(0).expand_as(tokens)
    mask = torch.triu(torch.ones((1, 1, args.sequence_length, args.sequence_length),
                                     dtype=torch.bool, device="cuda"), diagonal=1)
    labels = torch.randint(args.vocab_size, token_shape, generator=generator).cuda()
    inputs = (tokens, positions, mask, labels)
    opt_ref = torch.optim.SGD(reference.parameters(), lr=args.learning_rate)
    opt_cand = torch.optim.SGD(candidate.parameters(), lr=args.learning_rate)
    ref_output, ref_loss, _ = run_step(reference, inputs, opt_ref)
    cand_output, cand_loss, _ = run_step(candidate, inputs, opt_cand)
    parameter_errors = [(a - b).abs().max().item() for a, b in
                        zip(reference.parameters(), candidate.parameters())]
    checks = {
        "output_max_abs": float((ref_output - cand_output).abs().max().item()),
        "output_allclose": bool(torch.allclose(ref_output, cand_output,
                                                atol=args.atol, rtol=args.rtol)),
        "loss_abs": float((ref_loss - cand_loss).abs().item()),
        "loss_allclose": bool(torch.allclose(ref_loss, cand_loss,
                                              atol=args.atol, rtol=args.rtol)),
        "parameter_max_abs_after_sgd": float(max(parameter_errors)),
        "finite": bool(torch.isfinite(ref_loss).item() and torch.isfinite(cand_loss).item()),
        "candidate_fused_calls": counter["calls"],
    }
    checks["passed"] = (
        checks["finite"] and checks["candidate_fused_calls"] == 1
        and checks["output_allclose"] and checks["loss_allclose"]
        and checks["parameter_max_abs_after_sgd"] <= args.atol
    )
    report["correctness"] = checks
    timings = {"reference_ms": [], "candidate_ms": []}
    for _ in range(args.warmup):
        run_step(reference, inputs, opt_ref)
        run_step(candidate, inputs, opt_cand)
    for index in range(args.repetitions):
        if index % 2:
            _, _, cand_ms = run_step(candidate, inputs, opt_cand, timed=True)
            _, _, ref_ms = run_step(reference, inputs, opt_ref, timed=True)
        else:
            _, _, ref_ms = run_step(reference, inputs, opt_ref, timed=True)
            _, _, cand_ms = run_step(candidate, inputs, opt_cand, timed=True)
        timings["reference_ms"].append(ref_ms)
        timings["candidate_ms"].append(cand_ms)
    ref_median = statistics.median(timings["reference_ms"])
    cand_median = statistics.median(timings["candidate_ms"])
    timings.update({
        "reference_median_ms": ref_median,
        "candidate_median_ms": cand_median,
        "candidate_over_reference_speedup": ref_median / cand_median,
        "tokens_per_step": tokens.numel(),
    })
    report["timing"] = timings
    report["status"] = "passed" if checks["passed"] else "failed"
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--megatron-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=21001)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--atol", type=float, default=0.0003)
    parser.add_argument("--rtol", type=float, default=0.001)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "seed": args.seed,
        "scope": "one-layer pinned Megatron GPT local-spec step; synthetic tokens; one attention BDA->pre-MLP RMSNorm fusion; no Transformer Engine; no production throughput claim",
        "configuration": {"sequence_length": args.sequence_length,
                          "micro_batch_size": args.micro_batch_size,
                          "hidden_size": args.hidden_size,
                          "warmup": args.warmup,
                          "repetitions": args.repetitions},
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
