"""Save an explicitly returned CLI plan or candidate; never execute response text."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("response", type=Path)
    parser.add_argument("kind", choices=("plan", "candidate"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite evidence")
    envelope = json.loads(args.response.read_text(encoding="utf-8-sig"))
    if envelope.get("is_error"):
        raise ValueError("CLI response is an error")
    result = envelope["result"]
    key = "plan_markdown" if args.kind == "plan" else "candidate_code"
    decoder = json.JSONDecoder()
    payload = None
    for i, char in enumerate(result):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(result[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and key in value:
            payload = value
            break
    if payload is None:
        raise ValueError("requested JSON payload not found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload[key].rstrip() + "\n", encoding="utf-8")
    receipt = args.output.with_suffix(args.output.suffix + ".receipt.json")
    receipt.write_text(json.dumps({"response": str(args.response),
        "response_sha256": hashlib.sha256(args.response.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "cli_model": list(envelope.get("modelUsage", {})),
        "source_ids": payload.get("source_ids", []),
        "rationale": payload.get("rationale"), "parent": payload.get("parent")}, indent=2) + "\n",
        encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
