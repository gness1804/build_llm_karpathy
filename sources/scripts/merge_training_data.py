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
import shutil
from datetime import datetime
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


def merge_files(
    existing_file: Path, new_files: List[Path], output_file: Path, archive: bool = False
):
    """Merge existing training data with new Reddit data"""
    print(f"📖 Reading existing training data: {existing_file}")
    with open(existing_file, "r", encoding="utf-8") as f:
        existing_content = f.read()

    # Extract Q&A pairs from existing data
    existing_pairs = extract_qa_pairs(existing_content)
    print(f"   Found {len(existing_pairs)} Q&A pairs in existing data")

    all_pairs = existing_pairs.copy()
    successfully_read_files = []

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
        successfully_read_files.append(new_file)

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

    # Archive source files if requested
    if archive and successfully_read_files:
        archive_source_files(successfully_read_files)

    return len(all_pairs), file_size


def archive_source_files(source_files: List[Path]):
    """Move source files to archive directory after successful merge"""
    if not source_files:
        return

    # Determine archive directory (sources/reddit/archive/)
    # Try to infer from first file's location
    first_file = source_files[0]
    if "reddit" in str(first_file.parent):
        archive_dir = first_file.parent / "archive"
    else:
        # Fallback: assume sources/reddit/archive
        script_dir = Path(__file__).parent
        sources_dir = script_dir.parent
        archive_dir = sources_dir / "reddit" / "archive"

    archive_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📦 Archiving {len(source_files)} source file(s) to {archive_dir}...")
    for source_file in source_files:
        try:
            archive_path = archive_dir / source_file.name
            # If file already exists in archive, add a timestamp
            if archive_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                stem = source_file.stem
                suffix = source_file.suffix
                archive_path = archive_dir / f"{stem}_{timestamp}{suffix}"

            shutil.move(str(source_file), str(archive_path))
            print(f"   ✅ Archived: {source_file.name} → {archive_path.name}")
        except Exception as e:
            print(f"   ⚠️  Failed to archive {source_file.name}: {e}")


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
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Archive source files after successful merge (moves them to sources/reddit/archive/)",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Skip archiving source files (overrides interactive prompt)",
    )

    args = parser.parse_args()

    # Determine output file
    if args.output:
        output_file = args.output
    else:
        output_file = args.existing.parent / "training_data_final_merged.md"

    # Determine if we should archive
    should_archive = args.archive
    if not args.archive and not args.no_archive:
        # Interactive prompt
        print("\n📦 Archive source files after merge?")
        print(f"   Files to archive: {len(args.new)} file(s)")
        for new_file in args.new:
            if new_file.exists():
                print(f"     - {new_file}")
        response = input("   Archive source files? [y/N]: ").strip().lower()
        should_archive = response in ("y", "yes")

    # Merge files
    total_pairs, file_size = merge_files(
        args.existing, args.new, output_file, archive=should_archive
    )

    print(f"\n✅ Done! Merged data saved to: {output_file}")


if __name__ == "__main__":
    main()
