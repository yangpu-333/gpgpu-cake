"""Small diagnostic for a newly recreated BI-V150 Pod; prints each completed step."""

import argparse
import time


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()


def step(name, fn):
    print("BEGIN", name, flush=True)
    start = time.monotonic()
    value = fn()
    print("END", name, "seconds=", round(time.monotonic() - start, 3),
          "value=", value, flush=True)
    return value


torch = step("import_torch", lambda: __import__("torch"))
step("torch_version", lambda: torch.__version__)
step("cuda_available", lambda: torch.cuda.is_available())
step("device_count", lambda: torch.cuda.device_count())
step("device_name", lambda: torch.cuda.get_device_name(args.device))
step("set_device", lambda: torch.cuda.set_device(args.device))
tensor = step("allocate_empty", lambda: torch.empty(16, device=f"cuda:{args.device}"))
step("zero", lambda: tensor.zero_())
step("add", lambda: tensor.add_(1))
step("synchronize", lambda: torch.cuda.synchronize())
step("copy_to_cpu", lambda: tensor.cpu().tolist())
print("HEALTH_PASS", flush=True)
