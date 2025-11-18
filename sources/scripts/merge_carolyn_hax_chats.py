#!/usr/bin/env python3
"""
Merge multiple Carolyn Hax chat files into a single document for LLM training.

This script:
- Finds all chat files in the carolyn_hax_chats directory
- Sorts them from most recent to least recent (based on filename date)
- Filters out empty files
- Merges them into a single output file

Usage:
    python3 merge_carolyn_hax_chats.py [output_file]

    If output_file is not specified, defaults to 'carolyn_hax_merged.md' in the sources directory.

Example:
    python3 merge_carolyn_hax_chats.py
    python3 merge_carolyn_hax_chats.py merged_chats.md
"""

import re
import sys
from datetime import datetime
from pathlib import Path


def parse_date_from_filename(filename):
    """
    Extract date from filename like 'carolyn_hax_103125_chat.md'
    Format is MMDDYY (e.g., 103125 = October 31, 2025)

    Returns:
        datetime object or None if parsing fails
    """
    # Extract the date part (MMDDYY)
    match = re.search(r"carolyn_hax_(\d{6})_chat\.md", filename)
    if not match:
        return None

    date_str = match.group(1)
    try:
        # Parse MMDDYY format
        return datetime.strptime(date_str, "%m%d%y")
    except ValueError:
        return None


def get_chat_files(directory):
    """
    Get all chat files from the directory, sorted by date (most recent first).

    Args:
        directory: Path to directory containing chat files

    Returns:
        List of (filename, filepath, date) tuples, sorted by date descending
    """
    chat_dir = Path(directory)
    if not chat_dir.exists():
        print(f"Error: Directory '{directory}' does not exist.", file=sys.stderr)
        sys.exit(1)

    files_with_dates = []

    for filepath in chat_dir.glob("carolyn_hax_*_chat.md"):
        filename = filepath.name
        date = parse_date_from_filename(filename)

        if date is None:
            print(
                f"Warning: Could not parse date from '{filename}', skipping.",
                file=sys.stderr,
            )
            continue

        files_with_dates.append((filename, filepath, date))

    # Sort by date descending (most recent first)
    files_with_dates.sort(key=lambda x: x[2], reverse=True)

    return files_with_dates


def is_file_empty(filepath):
    """Check if a file is empty or contains only whitespace."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
            return len(content) == 0
    except Exception as e:
        print(f"Warning: Could not read '{filepath}': {e}", file=sys.stderr)
        return True  # Treat as empty if we can't read it


def merge_chat_files(files_with_dates, output_file):
    """
    Merge chat files into a single document.

    Args:
        files_with_dates: List of (filename, filepath, date) tuples
        output_file: Path to output file
    """
    merged_content = []
    files_merged = 0
    files_skipped = 0

    for filename, filepath, date in files_with_dates:
        # Skip empty files
        if is_file_empty(filepath):
            print(f"Skipping empty file: {filename}")
            files_skipped += 1
            continue

        # Read file content
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()

            # Add separator and filename header
            merged_content.append(f"\n\n{'='*80}\n")
            merged_content.append(f"# {filename}\n")
            merged_content.append(f"# Date: {date.strftime('%B %d, %Y')}\n")
            merged_content.append(f"{'='*80}\n\n")
            merged_content.append(content)

            files_merged += 1
            print(f"Merged: {filename} ({date.strftime('%B %d, %Y')})")

        except Exception as e:
            print(f"Error reading '{filepath}': {e}", file=sys.stderr)
            files_skipped += 1

    # Write merged content
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("".join(merged_content))

        print(f"\n{'='*80}")
        print(f"Successfully merged {files_merged} files into '{output_file}'")
        if files_skipped > 0:
            print(f"Skipped {files_skipped} empty or unreadable files")
        print(f"{'='*80}")

    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge Carolyn Hax chat files into a single document",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        default=None,
        help="Output file path (default: carolyn_hax_merged.md in sources directory)",
    )
    parser.add_argument(
        "--chats-dir",
        default=None,
        help="Directory containing chat files (default: sources/carolyn_hax_chats)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print verbose output"
    )

    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent
    sources_dir = script_dir  # Script is in sources directory

    if args.chats_dir:
        chats_directory = Path(args.chats_dir)
    else:
        chats_directory = sources_dir / "carolyn_hax_chats"

    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = sources_dir / "carolyn_hax_merged.md"

    # Get and sort chat files
    if args.verbose:
        print(f"Looking for chat files in: {chats_directory}")

    files_with_dates = get_chat_files(chats_directory)

    if not files_with_dates:
        print(
            f"Error: No valid chat files found in '{chats_directory}'", file=sys.stderr
        )
        sys.exit(1)

    if args.verbose:
        print(f"\nFound {len(files_with_dates)} chat files")
        print("Files will be merged in this order (most recent first):")
        for filename, _, date in files_with_dates:
            print(f"  - {filename} ({date.strftime('%B %d, %Y')})")
        print()

    # Merge files
    merge_chat_files(files_with_dates, output_file)


if __name__ == "__main__":
    main()
