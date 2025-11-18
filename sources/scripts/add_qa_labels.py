#!/usr/bin/env python3
"""
Add QUESTION and ANSWER labels to chat/forum transcripts for LLM fine-tuning.

This script processes markdown files containing Q&A conversations and adds
"QUESTION:" and "ANSWER:" labels to make them easier to use for LLM training.

Usage:
    python3 add_qa_labels.py <input_file> [output_file]

    If output_file is not specified, the input file will be modified in place.

Example:
    python3 add_qa_labels.py carolyn_hax_103125_chat.md
    python3 add_qa_labels.py input.md output_labeled.md
"""

import re
import sys
import argparse


def add_qa_labels(content):
    """
    Add QUESTION: and ANSWER: labels to a chat transcript.

    Args:
        content: String containing the file content

    Returns:
        String with labels added
    """
    lines = content.split("\n")
    result_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if this is a Guest line (question)
        # Matches patterns like "Guest11:05 a.m." or "Guest12:16 p.m." or "Guest Oct 24, 11:50 a.m."
        if re.match(r"^Guest\d+:\d+ [ap]\.m\.|Guest \w+ \d+, \d+:\d+ [ap]\.m\.", line):
            result_lines.append(line)
            i += 1
            # Skip empty lines and add QUESTION: before the first non-empty line.
            # BUT don't add question if it already exists. Only add it if it doesn't exist.
            while i < len(lines) and lines[i].strip() == "":
                result_lines.append(lines[i])
                i += 1
            if i < len(lines):
                # Check if line already starts with QUESTION: (handle multiple prefixes)
                line_content = lines[i]
                # Strip any existing QUESTION: prefixes
                while line_content.strip().startswith("QUESTION: "):
                    line_content = line_content.strip()[10:].strip()
                # Add QUESTION: prefix once
                result_lines.append("QUESTION: " + line_content)
                i += 1
        # Check if this is a Carolyn Hax answer (after "Advice Columnist" line)
        elif (
            line.strip() == "Advice Columnist"
            and i > 0
            and "Carolyn Hax" in lines[i - 1]
        ):
            result_lines.append(line)
            i += 1
            # Skip empty lines and add ANSWER: before the first non-empty line.
            # BUT don't add answer if it already exists. Only add it if it doesn't exist.
            while i < len(lines) and lines[i].strip() == "":
                result_lines.append(lines[i])
                i += 1
            if i < len(lines):
                # Check if line already starts with ANSWER: (handle multiple prefixes)
                line_content = lines[i]
                # Strip any existing ANSWER: prefixes
                while line_content.strip().startswith("ANSWER: "):
                    line_content = line_content.strip()[8:].strip()
                # Add ANSWER: prefix once
                result_lines.append("ANSWER: " + line_content)
                i += 1
        else:
            result_lines.append(line)
            i += 1

    return "\n".join(result_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Add QUESTION and ANSWER labels to chat transcripts for LLM fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input_file", help="Input markdown file to process")
    parser.add_argument(
        "output_file",
        nargs="?",
        help="Output file (default: modify input file in place)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print verbose output"
    )

    args = parser.parse_args()

    # Read the input file
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    # Process the content
    if args.verbose:
        print(f"Processing {args.input_file}...")

    labeled_content = add_qa_labels(content)

    # Determine output file
    output_file = args.output_file if args.output_file else args.input_file

    # Write the result
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(labeled_content)
        if args.verbose:
            print(f"Successfully wrote labeled content to {output_file}")
        else:
            print(f"File updated successfully: {output_file}")
    except Exception as e:
        print(f"Error writing file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
