"""Validate stage-ready JSONL and optionally resolve its media paths for Swift."""

import argparse
import json
import math
import tempfile
from collections import Counter
from pathlib import Path

from training.structured_cot import parse_structured_cot

EXPECTED = {
    "teacher-sft": {"rows": 30662},
    "sft": {"rows": 10113},
    "opd": {"rows": 40992, "padding": 54},
    "grpo": {
        "rows": 5376,
        "padding": 5,
        "rexvqa/rex_scalar": 4027,
        "3drad/open": 538,
        "3drad/choice": 537,
        "3drad/3d_measurement": 269,
    },
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve_media(value, root, check):
    require(isinstance(value, str) and value.strip(), "expected a local media path")
    path = (root / value).resolve()
    if check and not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def normalize_media(row, field, count, root, check):
    values = row.get(field)
    require(isinstance(values, list) and len(values) in count, f"invalid {field} list length")
    row[field] = [resolve_media(value, root, check) for value in values]

def validate_row(row, stage, root, check, stats):
    require(isinstance(row, dict), "expected a JSON object")
    messages = row.get("messages")
    roles = [] if not isinstance(messages, list) else [
        item.get("role") if isinstance(item, dict) else None for item in messages
    ]
    require(roles == (["user", "assistant"] if stage.endswith("sft") else ["user"]), "unexpected message roles")
    require(all(isinstance(msg.get("content"), str) and msg["content"].strip() for msg in messages),
            "each message needs nonempty text content")
    if stage == "teacher-sft":
        row["channel"] = resolve_media(row.get("channel"), root, check)
    else:
        normalize_media(row, "images", range(1, 7) if stage == "grpo" else (2,), root, check)
        require(messages[0]["content"].count("<image>") == len(row["images"]),
                "user message must contain one <image> per image")

    if stage in ("opd", "grpo"):
        padding = row.get("_native_opd_padding", False)
        require(isinstance(padding, bool), "_native_opd_padding must be a JSON boolean")
        stats["padding"] += padding
    if stage == "opd":
        normalize_media(row, "teacher_images", (1,), root, check)
        prompt = row.get("teacher_prompt")
        require(isinstance(prompt, str) and prompt.strip() and prompt.count("<image>") <= 1,
                "teacher_prompt needs CT question text and at most one <image> placeholder")
    if stage == "grpo":
        kind = row.get("reward_type")
        require(kind in ("rex_scalar", "3d_measurement", "choice", "open"), "unknown reward_type")
        solution = row.get("solution")
        require(isinstance(solution, str) and solution.strip(), "GRPO needs a solution string")
        if kind == "rex_scalar":
            try:
                mm = float(solution)
            except ValueError as exc:
                raise ValueError("rex_scalar solution must be a number in millimeters") from exc
            require(math.isfinite(mm) and mm >= 0, "rex_scalar solution must be a finite nonnegative number")
        if kind == "choice":
            require(row.get("gold_answer") in ("A", "B", "C", "D"), "choice rows need an A–D gold_answer")
        if kind == "open":
            question = row.get("question")
            require(isinstance(question, str) and question.strip(), "open rows need a question string")
        if not padding:
            stats[f"{row.get('source_dataset')}/{kind}"] += 1
    stats["rows"] += 1


def process(source, destination, stage, root, check, paper, structured):
    stats = Counter()
    with source.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
                validate_row(row, stage, root, check, stats)
                if structured:
                    require(parse_structured_cot(row["messages"][-1]["content"]) is not None,
                            "expected both reasoning sections and a boxed final answer")
            except (ValueError, OSError) as exc:
                raise ValueError(f"{source}:{number}: {exc}") from exc
            if destination is not None:
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    require(stats["rows"] > 0, "input JSONL is empty")
    if paper:
        for key, expected in EXPECTED[stage].items():
            require(stats[key] == expected, f"{stage} {key}: expected {expected}, found {stats[key]}")
    return dict(stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=EXPECTED)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, help="Write normalized JSONL to a new file")
    parser.add_argument("--media-root", type=Path, help="Resolve relative media paths from here")
    parser.add_argument("--skip-media-check", action="store_true", help="Validate schema without local media")
    parser.add_argument("--paper", action="store_true", help="Check paper dataset counts")
    parser.add_argument("--structured", action="store_true", help="Check structured student SFT targets")
    args = parser.parse_args()
    if args.structured and args.stage != "sft":
        parser.error("--structured is only for student SFT")

    processing = dict(source=args.input, stage=args.stage, root=args.media_root or args.input.parent,
                      check=not args.skip_media_check, paper=args.paper, structured=args.structured)
    try:
        if args.output is None:
            stats = process(destination=None, **processing)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent) as handle:
                stats = process(destination=handle, **processing)
                handle.flush()
                # Publish a complete file atomically and refuse to overwrite existing input or output.
                args.output.hardlink_to(handle.name)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(stats, sort_keys=True))


if __name__ == "__main__":
    main()
