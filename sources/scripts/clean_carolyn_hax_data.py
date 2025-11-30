#!/usr/bin/env python3
"""
Clean Carolyn Hax chat data by removing metadata and formatting noise.

This script removes:
- File headers and separators
- Timestamps
- "Press Enter to expand" and similar UI elements
- "Likes", "Post has X replies" metadata
- "avatar" markers
- Excessive whitespace
- Keeps only QUESTION: and ANSWER: content with clean formatting
"""

import re
import sys
from pathlib import Path


def clean_carolyn_hax_text(text: str) -> str:
    """
    Clean Carolyn Hax chat data by removing metadata and noise.

    Args:
        text: Raw text from merged chat files

    Returns:
        Cleaned text with only Q&A content
    """
    lines = text.split("\n")
    cleaned_lines = []
    # skip_until_question = False

    for line in lines:
        # Skip file headers and separators
        if line.strip().startswith("===") or line.strip().startswith("#"):
            continue

        # Skip empty lines at start
        if not cleaned_lines and not line.strip():
            continue

        # Remove timestamps (e.g., "11:05 a.m.", "Guest11:05 a.m.")
        line = re.sub(r"\d+:\d+ [ap]\.m\.", "", line)
        line = re.sub(r"Guest\d+:\d+ [ap]\.m\.", "", line)

        # Remove "Press Enter to expand" and similar
        line = re.sub(r"Press Enter to expand", "", line, flags=re.IGNORECASE)
        line = re.sub(r"\d+Press Enter to expand", "", line, flags=re.IGNORECASE)

        # Remove "Likes" and "Post has X replies" metadata
        line = re.sub(r"\d+ Likes?", "", line)
        line = re.sub(
            r"Post has \d+ replies? and \d+ Likes?", "", line, flags=re.IGNORECASE
        )
        line = re.sub(
            r"Post has undefined replies? and \d+ Likes?", "", line, flags=re.IGNORECASE
        )
        line = re.sub(r"Post has \d+ replies?", "", line, flags=re.IGNORECASE)
        line = re.sub(r"Post has undefined replies?", "", line, flags=re.IGNORECASE)
        line = re.sub(r"Post comments? expanded", "", line, flags=re.IGNORECASE)
        line = re.sub(r"Post expanded", "", line, flags=re.IGNORECASE)
        # Remove partial "Post has" lines
        if "Post has" in line and "replies" not in line and "Likes" not in line:
            continue

        # Remove standalone numbers (often like counts)
        if line.strip().isdigit() and len(line.strip()) < 4:
            continue

        # Remove lines that are just "and" or other single words that are likely fragments
        if line.strip() in [
            "and",
            "or",
            "the",
            "a",
            "an",
            "to",
            "of",
            "in",
            "is",
            "it",
        ]:
            continue

        # Remove "avatar" markers
        if line.strip() == "avatar":
            continue

        # Remove author headers that aren't part of content
        if (
            "Carolyn HaxAdvice Columnist" in line
            and not line.strip().startswith("QUESTION")
            and not line.strip().startswith("ANSWER")
        ):
            # Keep only if it's part of actual content, otherwise skip
            if ":" not in line or len(line.split(":")) < 2:
                continue

        # Remove "Guest" lines that are just metadata
        if (
            line.strip() == "Guest"
            or line.strip().startswith("Guest ")
            and ":" not in line
        ):
            continue

        # Remove "image" markers
        if line.strip() == "image":
            continue

        # Clean up the line
        line = line.strip()

        # Skip completely empty lines after cleaning
        if not line:
            # Only add empty line if previous line wasn't empty (avoid excessive blank lines)
            if cleaned_lines and cleaned_lines[-1].strip():
                cleaned_lines.append("")
            continue

        # Keep QUESTION: and ANSWER: lines as-is (they're important markers)
        if line.startswith("QUESTION:") or line.startswith("ANSWER:"):
            cleaned_lines.append(line)
        # Keep content lines that aren't just metadata
        elif not re.match(r"^[A-Z][a-z]+ \d+, \d{4}$", line):  # Skip date-only lines
            cleaned_lines.append(line)

    # Join and clean up excessive whitespace
    cleaned_text = "\n".join(cleaned_lines)

    # Remove excessive blank lines (more than 2 consecutive)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    # Remove leading/trailing whitespace
    cleaned_text = cleaned_text.strip()

    return cleaned_text


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean Carolyn Hax chat data by removing metadata"
    )
    parser.add_argument("input_file", help="Input file to clean")
    parser.add_argument(
        "-o", "--output", help="Output file (default: input_file with _cleaned suffix)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print verbose output"
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File '{args.input_file}' not found.", file=sys.stderr)
        sys.exit(1)

    # Read input
    with open(input_path, "r", encoding="utf-8") as f:
        original_text = f.read()

    if args.verbose:
        print(f"Original file size: {len(original_text):,} characters")
        print(f"Original lines: {len(original_text.splitlines()):,}")

    # Clean the text
    cleaned_text = clean_carolyn_hax_text(original_text)

    if args.verbose:
        print(f"Cleaned file size: {len(cleaned_text):,} characters")
        print(f"Cleaned lines: {len(cleaned_text.splitlines()):,}")
        print(
            f"Reduction: {len(original_text) - len(cleaned_text):,} characters removed"
        )

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Add _cleaned before extension
        stem = input_path.stem
        suffix = input_path.suffix
        output_path = input_path.parent / f"{stem}_cleaned{suffix}"

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print(f"✅ Cleaned data written to: {output_path}")
    print(
        f"   Original: {len(original_text):,} chars → Cleaned: {len(cleaned_text):,} chars"
    )


if __name__ == "__main__":
    main()
