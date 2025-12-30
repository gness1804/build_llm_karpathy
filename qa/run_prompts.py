#!/usr/bin/env python3
"""
Script that runs `qa/run_inference.py` multiple times and summarizes the results.

Usage example:
  python3 qa/run_prompts.py -p 'caretaking burnout' 'partners grief' 3

Additional options:
  --output-dir       Override where `run_inference.py` saves its files
  --model-type       Which model backend to target (default: openai_backend)
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RUN_INFERENCE_SCRIPT = SCRIPT_DIR / "run_inference.py"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "inference"


def build_command(prompt: str, model_type: str, output_dir: Optional[str]) -> list[str]:
    """Return the command line for a single inference run."""
    command = [
        sys.executable,
        str(RUN_INFERENCE_SCRIPT),
        "--prompt",
        prompt,
        "--model_type",
        model_type,
        "--save-output",
    ]
    if output_dir:
        command.extend(["--output-dir", output_dir])
    return command


def extract_output_path(log_text: str) -> Optional[Path]:
    """
    Look for the log line that announces where the output file was written.

    The child script prints either:
      ✅ Output written to: /full/path/to/file.txt
      ✅ Output saved to: /full/path/to/file.txt
    """
    match = re.search(r"Output (?:written|saved) to:\s*(.+)", log_text)
    if not match:
        return None
    output_path = match.group(1).strip()
    path = Path(output_path)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def parse_output_file(path: Path) -> tuple[Optional[str], Optional[float]]:
    """Return the timestamp and score stored inside the inference output."""
    generated_at = None
    score = None

    if not path.exists():
        print(f"⚠️  Unable to find output file at {path}", file=sys.stderr)
        return generated_at, score

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if generated_at is None and line.startswith("Generated at:"):
                generated_at = line.split(":", 1)[1].strip()
            if score is None:
                score_match = re.search(r"Score:\s*([0-9]+(?:\.[0-9]+)?)", line)
                if score_match:
                    score = float(score_match.group(1))

    if generated_at is None:
        generated_at = datetime.fromtimestamp(path.stat().st_mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    return generated_at, score


def echo_process_output(result: subprocess.CompletedProcess) -> str:
    """Print captured stdout/stderr and return the combined text."""
    combined = ""
    if result.stdout:
        print(result.stdout, end="")
        combined += result.stdout
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
        combined += result.stderr
    return combined


def escape_markdown(text: str) -> str:
    """Escape pipe characters so they do not break markdown tables."""
    return text.replace("|", "\\|")


def write_results(
    summary_path: Path,
    invocation: str,
    prompts: Iterable[str],
    runs: int,
    model_type: str,
    results: list[dict],
) -> None:
    """Persist the markdown summary with a table and averages."""
    prompt_list = ", ".join(prompts)
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Inference Prompt Results",
        "",
        f"**Generated at:** {now_iso}",
        f"**Invocation:** `{invocation}`",
        f"**Prompts:** {prompt_list}",
        f"**Runs per prompt:** {runs}",
        f"**Model type:** {model_type}",
        "",
        "## Run Results",
        "",
        "| Prompt | Run | Timestamp | Score |",
        "| --- | --- | --- | --- |",
    ]

    for entry in results:
        score_text = f"{entry['score']:.2f}" if entry["score"] is not None else "N/A"
        timestamp = entry["generated_at"] or "N/A"
        lines.append(
            f"| {escape_markdown(entry['prompt'])} | "
            f"{entry['run']} | "
            f"{timestamp} | "
            f"{score_text} |"
        )

    lines.append("")
    lines.append("## Averages")
    lines.append("")
    lines.append("| Prompt | Average Score |")
    lines.append("| --- | --- |")

    grouped = defaultdict(list)
    for entry in results:
        if entry["score"] is not None:
            grouped[entry["prompt"]].append(entry["score"])

    for prompt in sorted({entry["prompt"] for entry in results}):
        scores = grouped.get(prompt, [])
        if scores:
            avg = sum(scores) / len(scores)
            avg_text = f"{avg:.2f}"
        else:
            avg_text = "N/A"
        lines.append(f"| {escape_markdown(prompt)} | {avg_text} |")

    lines.append("")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n📊 Summary written to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multiple prompts through qa/run_inference.py."
    )
    parser.add_argument(
        "-p",
        "--prompts",
        nargs="+",
        required=True,
        help="One or more prompt shorthands (or direct text) to evaluate.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of times to run each prompt (same count for every prompt).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Optional override for where run_inference.py saves its outputs.",
    )
    parser.add_argument(
        "--model-type",
        choices=["gpt2", "openai_backend", "from_scratch"],
        default="openai_backend",
        help="Backend passed through to run_inference.py.",
    )

    args = parser.parse_args()

    if not RUN_INFERENCE_SCRIPT.exists():
        print(f"❌ Unable to find {RUN_INFERENCE_SCRIPT}", file=sys.stderr)
        sys.exit(1)

    invocation = " ".join(sys.argv)
    results: list[dict] = []

    for prompt in args.prompts:
        for run_index in range(1, args.runs + 1):
            print(
                f"\n🚀 Running prompt '{prompt}' (run {run_index} of {args.runs}) "
                f"with model_type={args.model_type}"
            )
            command = build_command(prompt, args.model_type, args.output_dir)

            result = subprocess.run(
                command,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
            )

            combined_output = echo_process_output(result)

            if result.returncode != 0:
                print(
                    f"❌ run_inference.py exited with {result.returncode}",
                    file=sys.stderr,
                )
                sys.exit(result.returncode)

            output_path = extract_output_path(combined_output)
            if output_path is None:
                print(
                    "⚠️  Unable to determine the output file that was written.",
                    file=sys.stderr,
                )
                continue

            generated_at, score = parse_output_file(output_path)
            results.append(
                {
                    "prompt": prompt,
                    "run": run_index,
                    "generated_at": generated_at,
                    "score": score,
                }
            )

    if not results:
        print("⚠️  No successful runs to summarize.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = RESULTS_DIR / f"results-{timestamp}.md"
    write_results(
        summary_path=summary_path,
        invocation=invocation,
        prompts=args.prompts,
        runs=args.runs,
        model_type=args.model_type,
        results=results,
    )


if __name__ == "__main__":
    main()

