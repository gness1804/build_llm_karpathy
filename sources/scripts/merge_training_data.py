#!/usr/bin/env python3
"""
Merge Reddit data with existing training data.

This script:
1. Reads the existing training_data_final.md
2. Reads new Reddit data files
3. Strips metadata headers from Reddit data
4. Merges everything into a new training file
"""

import argparse
import re
from pathlib import Path
from typing import List


def strip_reddit_metadata(content: str) -> str:
    """Remove Reddit post metadata headers, keeping only Q&A content"""
    lines = content.split("\n")
    cleaned_lines = []
    skip_until_question = False

    for i, line in enumerate(lines):
        # Skip metadata lines (lines starting with # or =)
        if line.strip().startswith("#") or line.strip().startswith("="):
            skip_until_question = True
            continue

        # Skip empty lines after metadata
        if skip_until_question and not line.strip():
            continue

        # Once we hit non-empty, non-metadata content, we're past the header
        if skip_until_question and line.strip():
            skip_until_question = False

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def extract_qa_pairs(content: str) -> List[str]:
    """Extract Q&A pairs from content, handling both formats"""
    # Split by double newlines to get sections
    sections = re.split(r"\n\n+", content)
    qa_pairs = []
    current_qa = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Check if this is a QUESTION
        if section.startswith("QUESTION:"):
            # Save previous Q&A if exists
            if current_qa:
                qa_pairs.append("\n\n".join(current_qa))
            current_qa = [section]
        elif section.startswith("ANSWER:"):
            if current_qa:
                current_qa.append(section)
            else:
                # Sometimes answers appear without questions (shouldn't happen, but handle it)
                current_qa = [section]
        else:
            # Continuation of current Q or A
            if current_qa:
                current_qa.append(section)

    # Add final Q&A pair
    if current_qa:
        qa_pairs.append("\n\n".join(current_qa))

    return qa_pairs


def merge_files(existing_file: Path, new_files: List[Path], output_file: Path):
    """Merge existing training data with new Reddit data"""
    print(f"📖 Reading existing training data: {existing_file}")
    with open(existing_file, "r", encoding="utf-8") as f:
        existing_content = f.read()

    # Extract Q&A pairs from existing data
    existing_pairs = extract_qa_pairs(existing_content)
    print(f"   Found {len(existing_pairs)} Q&A pairs in existing data")

    all_pairs = existing_pairs.copy()

    # Process each new file
    for new_file in new_files:
        if not new_file.exists():
            print(f"⚠️  Warning: {new_file} does not exist, skipping")
            continue

        print(f"📖 Reading new data: {new_file}")
        with open(new_file, "r", encoding="utf-8") as f:
            new_content = f.read()

        # Strip metadata
        cleaned_content = strip_reddit_metadata(new_content)

        # Extract Q&A pairs
        new_pairs = extract_qa_pairs(cleaned_content)
        print(f"   Found {len(new_pairs)} Q&A pairs in {new_file.name}")

        all_pairs.extend(new_pairs)

    # Write merged content
    print(f"\n💾 Writing merged data to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_pairs))
        f.write("\n")  # Final newline

    file_size = output_file.stat().st_size
    print(f"✅ Merged {len(all_pairs)} total Q&A pairs")
    print(
        f"   Output size: {file_size / 1024:.1f} KB ({file_size / (1024*1024):.2f} MB)"
    )

    return len(all_pairs), file_size


def main():
    parser = argparse.ArgumentParser(
        description="Merge Reddit data with existing training data"
    )
    parser.add_argument(
        "--existing",
        type=Path,
        default=Path("sources/training_data_final.md"),
        help="Existing training data file (default: sources/training_data_final.md)",
    )
    parser.add_argument(
        "--new",
        type=Path,
        nargs="+",
        required=True,
        help="New Reddit data file(s) to merge",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file (default: sources/training_data_final_merged.md)",
    )

    args = parser.parse_args()

    # Determine output file
    if args.output:
        output_file = args.output
    else:
        output_file = args.existing.parent / "training_data_final_merged.md"

    # Merge files
    total_pairs, file_size = merge_files(args.existing, args.new, output_file)

    print(f"\n✅ Done! Merged data saved to: {output_file}")


if __name__ == "__main__":
    main()
