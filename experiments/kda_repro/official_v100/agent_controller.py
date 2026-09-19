"""Bounded OpenAI-compatible optimization loop for the official GDN Decode task.

The model can propose one isolated candidate module per iteration.  This controller
owns validation, scoring, promotion, budgets, and the immutable candidate ledger.
It intentionally exposes no shell or filesystem tool to the model.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import shutil
import ssl
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


ALLOWED_IMPORTS = {"math", "torch", "triton", "triton.language"}
FORBIDDEN_CALLS = {"eval", "exec", "open", "compile", "input", "__import__",
                   "globals", "locals", "vars", "getattr", "setattr", "delattr",
                   "exit", "quit", "breakpoint", "help"}
FORBIDDEN_TORCH_AREAS = {"distributed", "hub", "multiprocessing", "utils",
                         "package", "serialization", "save", "load", "ops", "classes"}
SMOKE_INDICES = "0,10,25,53"


class LLMResponseError(RuntimeError):
    """An API response was received but did not contain usable text."""

    def __init__(self, message: str, metadata: dict):
        super().__init__(message)
        self.metadata = metadata


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def attribute_parts(node: ast.AST) -> list[str]:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return list(reversed(parts))


def validate_source(source: str) -> None:
    if len(source.encode()) > 100_000:
        raise ValueError("candidate source exceeds 100 KB")
    tree = ast.parse(source, filename="generated_candidate.py")
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, ast.Import):
            if any(alias.name not in ALLOWED_IMPORTS for alias in node.names):
                raise ValueError("candidate imports a module outside the allowlist")
            continue
        if isinstance(node, ast.Assign):
            if not all(isinstance(target, ast.Name) and target.id.isupper() for target in node.targets):
                raise ValueError("top-level assignments must be uppercase constants")
            try:
                ast.literal_eval(node.value)
            except Exception as error:
                raise ValueError("top-level constants must be literals") from error
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                parts = attribute_parts(target)
                if parts not in (["torch", "no_grad"], ["triton", "jit"]):
                    raise ValueError(f"unapproved decorator: {'.'.join(parts)}")
            continue
        raise ValueError(f"unapproved top-level statement: {type(node).__name__}")

    exported = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "triton_candidate"]
    if len(exported) != 1:
        raise ValueError("candidate must define triton_candidate exactly once")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and (node.id in FORBIDDEN_CALLS or "__" in node.id):
            raise ValueError(f"forbidden name: {node.id}")
        if isinstance(node, ast.Attribute):
            parts = attribute_parts(node)
            if any("__" in part for part in parts):
                raise ValueError("dunder attribute access is forbidden")
            if len(parts) > 1 and parts[0] == "torch" and parts[1] in FORBIDDEN_TORCH_AREAS:
                raise ValueError(f"forbidden torch area: {parts[1]}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            raise ValueError(f"forbidden call: {node.func.id}")


def parse_json_content(content: str) -> dict:
    if not isinstance(content, str) or not content.strip():
        raise ValueError("API response content is empty")
    text = content.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3]
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("API response contains no JSON object")
        value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("API response must be a JSON object")
    return value


def api_endpoint() -> str:
    explicit = os.environ.get("KDA_LLM_ENDPOINT")
    if explicit:
        return explicit
    base = os.environ.get("KDA_LLM_BASE_URL", "").rstrip("/")
    if not base:
        raise RuntimeError("set KDA_LLM_BASE_URL or KDA_LLM_ENDPOINT")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def call_llm(messages: list[dict], json_mode: bool, *,
             max_tokens: int | None = None) -> tuple[str, dict]:
    key = os.environ.get("KDA_LLM_API_KEY")
    model = os.environ.get("KDA_LLM_MODEL")
    if not key or not model:
        raise RuntimeError("set KDA_LLM_API_KEY and KDA_LLM_MODEL")
    body = {
        "model": model,
        "messages": messages,
        "temperature": float(os.environ.get("KDA_LLM_TEMPERATURE", "0.2")),
        "max_tokens": max_tokens if max_tokens is not None else
                      int(os.environ.get("KDA_LLM_MAX_TOKENS", "8000")),
    }
    if json_mode and os.environ.get("KDA_LLM_JSON_MODE", "1") != "0":
        body["response_format"] = {"type": "json_object"}
    auth_header = os.environ.get("KDA_LLM_AUTH_HEADER", "Authorization")
    auth_scheme = os.environ.get("KDA_LLM_AUTH_SCHEME", "Bearer").strip()
    auth_value = f"{auth_scheme} {key}" if auth_scheme else key
    request = urllib.request.Request(
        api_endpoint(), data=json.dumps(body).encode(), method="POST",
        headers={auth_header: auth_value, "Content-Type": "application/json"})
    cafile = os.environ.get("SSL_CERT_FILE")
    context = ssl.create_default_context(cafile=cafile) if cafile else ssl.create_default_context()
    timeout = int(os.environ.get("KDA_LLM_TIMEOUT_SECONDS", "300"))
    with urllib.request.urlopen(request, context=context, timeout=timeout) as response:
        payload = json.loads(response.read())
    try:
        choice = payload["choices"][0]
        message = choice["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("unexpected OpenAI-compatible response shape") from error
    metadata = {"model": payload.get("model", model), "usage": payload.get("usage", {}),
                "id": payload.get("id"), "created": payload.get("created"),
                "finish_reason": choice.get("finish_reason"),
                "message_fields": sorted(message),
                "content_type": type(content).__name__,
                "content_chars": len(content) if isinstance(content, str) else None}
    if not isinstance(content, str) or not content.strip():
        completion_tokens = metadata["usage"].get("completion_tokens")
        raise LLMResponseError(
            "API returned empty content "
            f"(finish_reason={metadata['finish_reason']!r}, completion_tokens={completion_tokens!r})",
            metadata)
    return content, metadata


def call_candidate_llm(messages: list[dict]) -> tuple[str, dict]:
    """Retry one empty response without JSON mode and retain both attempt records."""
    limit = int(os.environ.get("KDA_LLM_CANDIDATE_MAX_TOKENS", "4000"))
    attempts = []
    last_error = None
    for attempt, json_mode in enumerate((True, False), start=1):
        request_messages = messages
        if attempt == 2:
            request_messages = messages + [{
                "role": "user",
                "content": ("The previous request reached the provider output limit. Return only the "
                            "requested compact JSON object. Keep rationale under 600 characters and "
                            "all replacement text under 2500 characters."),
            }]
        try:
            content, metadata = call_llm(request_messages, json_mode=json_mode,
                                         max_tokens=limit)
            attempts.append({"attempt": attempt, "json_mode": json_mode,
                             "status": "content_received", **metadata})
            return content, {**metadata, "attempts": attempts}
        except LLMResponseError as error:
            last_error = error
            attempts.append({"attempt": attempt, "json_mode": json_mode,
                             "status": "empty_content", "error": str(error),
                             **error.metadata})
    raise LLMResponseError(str(last_error), {"attempts": attempts})


def materialize_proposal(proposal: dict, current_source: str) -> tuple[str, str]:
    """Apply bounded exact replacements proposed by the model."""
    rationale = proposal.get("rationale")
    replacements = proposal.get("replacements")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("response requires a non-empty string rationale")
    if not isinstance(replacements, list) or not 1 <= len(replacements) <= 8:
        raise ValueError("response requires 1 to 8 exact replacements")
    result = current_source
    total_chars = 0
    for index, replacement in enumerate(replacements):
        if not isinstance(replacement, dict):
            raise ValueError(f"replacement {index} must be an object")
        old, new = replacement.get("old"), replacement.get("new")
        if not isinstance(old, str) or not old:
            raise ValueError(f"replacement {index} requires non-empty old text")
        if not isinstance(new, str):
            raise ValueError(f"replacement {index} requires string new text")
        total_chars += len(old) + len(new)
        if total_chars > 20_000:
            raise ValueError("replacement text exceeds 20000 characters")
        occurrences = result.count(old)
        if occurrences != 1:
            raise ValueError(
                f"replacement {index} old text must occur exactly once, found {occurrences}")
        result = result.replace(old, new, 1)
    if result == current_source:
        raise ValueError("proposal makes no source change")
    return result, rationale


def run_evaluation(evaluator: Path, data: Path, candidate: Path, output: Path,
                   indices: str | None, timeout: int, *, baseline: Path | None = None,
                   timing_repeats: int = 7) -> dict:
    command = [sys.executable, str(evaluator), "--data", str(data), "--output", str(output),
               "--candidate-source", str(candidate)]
    if baseline:
        command.extend(["--baseline-source", str(baseline)])
    if timing_repeats != 7:
        command.extend(["--timing-repeats", str(timing_repeats)])
    if indices:
        command.extend(["--indices", indices])
    child_env = os.environ.copy()
    for name in tuple(child_env):
        if name.startswith("KDA_LLM_") or name in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"}:
            child_env.pop(name, None)
    completed = subprocess.run(command, cwd=evaluator.parent, text=True, capture_output=True,
                               timeout=timeout, check=False, env=child_env)
    process_record = {"returncode": completed.returncode,
                      "stdout_tail": completed.stdout[-8000:], "stderr_tail": completed.stderr[-8000:]}
    (output.parent / (output.stem + "-process.json")).write_text(json.dumps(process_record, indent=2))
    if output.exists():
        report = json.loads(output.read_text())
    else:
        report = {"status": "missing_report"}
    if completed.returncode != 0:
        detail = report.get("error") or completed.stderr[-1000:] or completed.stdout[-1000:]
        raise RuntimeError(f"evaluation failed: {detail}")
    return report


def candidate_score(report: dict, expected_workloads: int, name: str = "candidate") -> float:
    if report.get("status") != "complete" or len(report.get("workloads", [])) != expected_workloads:
        raise ValueError("evaluation report is incomplete")
    values = []
    for workload in report["workloads"]:
        candidate = workload["candidates"].get(name, {})
        if candidate.get("status") != "correct" or "median_ms" not in candidate:
            raise ValueError(f"{name} is not correct for {workload.get('uuid')}")
        values.append(float(candidate["median_ms"]))
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("candidate latencies are invalid")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def recent_ledger(path: Path, count: int = 4) -> str:
    if not path.exists():
        return "No previous attempts."
    lines = path.read_text().splitlines()[-count:]
    compact = []
    for line in lines:
        item = json.loads(line)
        compact.append({key: item.get(key) for key in
                        ("candidate_id", "parent_id", "status", "score_ms", "reason")})
    return json.dumps(compact, indent=2)


def initialize_seed(workspace: Path, seed: Path, evaluator: Path, data: Path, timeout: int) -> tuple[str, float, Path]:
    candidates = workspace / "candidates"
    runs = workspace / "runs"
    ledger = workspace / "candidates.jsonl"
    seed_dir = candidates / "0000-seed"
    seed_dir.mkdir(parents=True, exist_ok=True)
    seed_copy = seed_dir / "candidate.py"
    if not seed_copy.exists():
        shutil.copy2(seed, seed_copy)
    report_path = runs / "0000-seed-full.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
    else:
        report = run_evaluation(evaluator, data, seed_copy, report_path, None, timeout)
    score = candidate_score(report, 54)
    if not ledger.exists():
        append_jsonl(ledger, {"candidate_id": "0000-seed", "parent_id": None,
                              "status": "promoted", "score_ms": score,
                              "source_sha256": sha256(seed_copy), "reason": "initial reviewed candidate"})
    return "0000-seed", score, seed_copy


def load_best(workspace: Path) -> tuple[str, float, Path] | None:
    ledger = workspace / "candidates.jsonl"
    if not ledger.exists():
        return None
    latest = {}
    for line in ledger.read_text().splitlines():
        record = json.loads(line)
        if record.get("candidate_id"):
            latest[record["candidate_id"]] = record
    promoted = [record for record in latest.values() if record.get("status") == "promoted"]
    if not promoted:
        return None
    best = min(promoted, key=lambda item: item["score_ms"])
    path = workspace / "candidates" / best["candidate_id"] / "candidate.py"
    return best["candidate_id"], float(best["score_ms"]), path


def write_agent_draft(workspace: Path, contract: str, plan: str,
                      best_id: str, best_score: float, best_source: Path) -> dict:
    prompt = f"""Write the initial KDA optimization draft in Markdown. Do not write code yet.

TASK CONTRACT
{contract}

EXECUTABLE CONTROLLER PLAN
{plan}

CURRENT REVIEWED BASELINE
candidate_id={best_id}
geometric_mean_ms={best_score:.9f}
source_sha256={sha256(best_source)}
```python
{best_source.read_text()}
```

Cover the current baseline, V100 risks, ranked candidate directions, first steps, exact validation gates, and promotion evidence."""
    content, metadata = call_llm([
        {"role": "system", "content": "You are a CUDA/Triton kernel optimization planner. Stay within the supplied contract."},
        {"role": "user", "content": prompt}], json_mode=False,
        max_tokens=int(os.environ.get("KDA_LLM_DRAFT_MAX_TOKENS", "8000")))
    docs = workspace / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "draft.md").write_text(content)
    (docs / "draft-api.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--min-improvement", type=float, default=0.01)
    parser.add_argument("--eval-timeout", type=int, default=1800)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--check-api", action="store_true")
    args = parser.parse_args()
    evaluator = here / "evaluate_decode.py"
    seed = here / "gdn_decode.py"
    contract_path = here / "agent" / "task_contract.md"
    plan_path = here / "agent" / "docs" / "plan.md"
    for required in (evaluator, seed, contract_path, plan_path,
                     args.data / "dataset-lock.json"):
        if not required.exists():
            raise SystemExit(f"missing required path: {required}")
    if args.iterations < 1 or not 0 < args.min_improvement < 1:
        raise SystemExit("invalid iteration or promotion budget")
    if args.check_config:
        print(json.dumps({"status": "config_valid", "endpoint": api_endpoint(),
                          "model": os.environ.get("KDA_LLM_MODEL"),
                          "api_key_present": bool(os.environ.get("KDA_LLM_API_KEY"))}))
        return
    if args.check_api:
        content, metadata = call_llm([
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": "Return exactly this object: {\"status\":\"ok\"}"}],
            json_mode=True, max_tokens=64)
        parsed = parse_json_content(content)
        if parsed.get("status") != "ok":
            raise SystemExit("API responded but did not follow the smoke-test contract")
        print(json.dumps({"status": "api_valid", "metadata": metadata}))
        return

    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    for name in ("candidates", "runs", "api"):
        (workspace / name).mkdir(exist_ok=True)
    shutil.copy2(contract_path, workspace / "task_contract.md")
    shutil.copy2(plan_path, workspace / "docs-plan.md")
    contract, plan = contract_path.read_text(), plan_path.read_text()

    existing = load_best(workspace)
    if existing is None:
        best_id, best_score, best_source = initialize_seed(
            workspace, seed, evaluator, args.data, args.eval_timeout)
    else:
        best_id, best_score, best_source = existing
    if not (workspace / "docs" / "draft.md").exists():
        write_agent_draft(workspace, contract, plan, best_id, best_score, best_source)

    ledger = workspace / "candidates.jsonl"
    used_numbers = [int(path.name.split('-', 1)[0]) for path in (workspace / "candidates").iterdir()
                    if path.is_dir() and path.name.split('-', 1)[0].isdigit()]
    next_number = max(used_numbers, default=0) + 1
    for offset in range(args.iterations):
        candidate_id = f"{next_number + offset:04d}-agent"
        candidate_dir = workspace / "candidates" / candidate_id
        candidate_dir.mkdir()
        current_source = best_source.read_text()
        prompt = f"""Propose exactly one small improvement to the source module below.
Return one compact JSON object with a string `rationale` and a `replacements` array. Each replacement is an object with exact string fields `old` and `new`; `old` must occur exactly once in CURRENT SOURCE. Do not return the complete module or a unified diff.
Do not claim success; the controller will apply the replacements and validate the complete result. Keep `rationale` under 600 characters, use at most 3 replacements, and keep all replacement text under 2500 characters. Output nothing outside the JSON object.

TASK CONTRACT
{contract}

CURRENT BEST
candidate_id={best_id}
geometric_mean_ms={best_score:.9f}
source_sha256={sha256(best_source)}

CURRENT SOURCE
```python
{current_source}
```

RECENT OUTCOMES
{recent_ledger(ledger)}
"""
        started = time.time()
        try:
            content, metadata = call_candidate_llm([
                {"role": "system", "content": "You optimize Triton kernels for sm70 under a strict immutable validator. Output valid JSON only."},
                {"role": "user", "content": prompt}])
            (workspace / "api" / f"{candidate_id}.json").write_text(json.dumps(
                {"metadata": metadata, "content": content}, indent=2, ensure_ascii=False))
            proposal = parse_json_content(content)
            source, rationale = materialize_proposal(proposal, current_source)
            validate_source(source)
            source_path = candidate_dir / "candidate.py"
            source_path.write_text(source)
            (candidate_dir / "api.json").write_text(json.dumps(
                {"metadata": metadata, "rationale": rationale}, indent=2, ensure_ascii=False))
            smoke_path = workspace / "runs" / f"{candidate_id}-smoke.json"
            smoke = run_evaluation(evaluator, args.data, source_path, smoke_path,
                                   SMOKE_INDICES, args.eval_timeout)
            candidate_score(smoke, 4)
            full_path = workspace / "runs" / f"{candidate_id}-full.json"
            full = run_evaluation(evaluator, args.data, source_path, full_path,
                                  None, args.eval_timeout, baseline=best_source,
                                  timing_repeats=21)
            score = candidate_score(full, 54)
            paired_baseline_score = candidate_score(full, 54, "baseline")
            threshold = paired_baseline_score * (1.0 - args.min_improvement)
            promoted = score < threshold
            record = {"candidate_id": candidate_id, "parent_id": best_id,
                      "status": "promoted" if promoted else "rejected",
                      "score_ms": score, "previous_best_ms": best_score,
                      "paired_baseline_ms": paired_baseline_score,
                      "paired_speedup": paired_baseline_score / score,
                      "source_sha256": sha256(source_path), "rationale": rationale,
                      "reason": "meets promotion threshold" if promoted else
                                f"requires score below {threshold:.9f} ms",
                      "elapsed_seconds": time.time() - started, "api": metadata}
            append_jsonl(ledger, record)
            if promoted:
                best_id, best_score, best_source = candidate_id, score, source_path
        except Exception as error:
            if isinstance(error, LLMResponseError):
                (workspace / "api" / f"{candidate_id}.json").write_text(json.dumps(
                    {"metadata": error.metadata, "content": None, "error": str(error)},
                    indent=2, ensure_ascii=False))
            reason = f"{type(error).__name__}: {error}"
            append_jsonl(ledger, {"candidate_id": candidate_id, "parent_id": best_id,
                                  "status": "rejected", "reason": reason[-4000:],
                                  "elapsed_seconds": time.time() - started})
        print(json.dumps({"iteration": offset + 1, "candidate_id": candidate_id,
                          "best_id": best_id, "best_score_ms": best_score}), flush=True)

    (workspace / "best.json").write_text(json.dumps(
        {"candidate_id": best_id, "score_ms": best_score,
         "source": str(best_source), "source_sha256": sha256(best_source)}, indent=2))
    print(json.dumps({"status": "agent_budget_complete", "best_id": best_id,
                      "best_score_ms": best_score, "workspace": str(workspace)}))


if __name__ == "__main__":
    main()
