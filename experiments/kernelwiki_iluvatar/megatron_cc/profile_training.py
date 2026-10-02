"""Host profile for diagnosis only; never used as promotion throughput evidence."""
import argparse
import cProfile
import json
import pstats
from pathlib import Path
import benchmark

parser = argparse.ArgumentParser()
parser.add_argument("candidate", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
if args.output.exists():
    parser.error("output exists")
root = Path("/private/atrex-megatron/src/megatron-lm")
from stage19_megatron_route_step import load_megatron
parts = load_megatron(root)
candidate = benchmark.inspect_candidate(args.candidate)
settings = argparse.Namespace(seed=23101, layers=4, hidden=512, heads=8,
    sequence=128, micro_batch=2, vocab=1024, autocast=False, warmup=5, samples=4, steps=10)
profiler = cProfile.Profile()
with profiler:
    result = benchmark.model_test(settings, candidate, parts)
profiler.dump_stats(str(args.output))
with args.output.with_suffix(".txt").open("w") as handle:
    pstats.Stats(profiler, stream=handle).sort_stats("tottime").print_stats(70)
    pstats.Stats(profiler, stream=handle).sort_stats("cumulative").print_stats("candidate|triton|RMSNorm|rms_norm")
args.output.with_suffix(".json").write_text(json.dumps({"scope": "cProfile diagnostic; not promotion timing", "model": result}, indent=2))
print(args.output, flush=True)
