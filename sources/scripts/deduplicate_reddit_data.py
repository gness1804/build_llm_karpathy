#!/usr/bin/env python3
"""
Deduplicate Reddit data file by removing duplicate posts.

This script reads a Reddit data file and removes duplicate posts,
keeping only the first occurrence of each unique post (identified by URL/post ID).

Usage:
    python3 deduplicate_reddit_data.py input_file.md [--output output_file.md]
"""

import argparse
from pathlib import Path
from typing import List, Tuple


def extract_post_id_from_url(url: str) -> str:
    """Extract post ID from Reddit URL"""
    try:
        # URL format: https://www.reddit.com/r/{subreddit}/comments/{post_id}/{title}/
        parts = url.split("/comments/")
        if len(parts) > 1:
            post_id = parts[1].split("/")[0]
            return post_id
    except Exception:
        pass
    return ""


def parse_reddit_file(file_path: Path) -> List[Tuple[str, List[str]]]:
    """Parse Reddit data file into posts

    Returns:
        List of (post_id, post_lines) tuples
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    posts = []
    current_post = []
    current_post_id = None
    i = 0

    while i < len(lines):
        line = lines[i].rstrip("\n")

        # Check if this is a separator line (starts a new post)
        if line.strip() and all(c == "=" for c in line.strip()):
            # Look ahead to see if this is followed by "# Reddit Post" or "# URL:"
            lookahead = i + 1
            is_post_start = False

            while lookahead < min(i + 5, len(lines)):
                next_line = lines[lookahead].strip()
                if next_line.startswith("# Reddit Post") or next_line.startswith(
                    "# URL:"
                ):
                    is_post_start = True
                    break
                elif next_line and not all(c == "=" for c in next_line):
                    # Not a separator, not a post header - probably content
                    break
                lookahead += 1

            if is_post_start:
                # Save previous post if we have one
                if current_post_id and current_post:
                    posts.append((current_post_id, current_post))

                # Start new post with this separator
                current_post = [line]
                i += 1

                # Read post header lines
                while i < len(lines):
                    header_line = lines[i].rstrip("\n")
                    current_post.append(header_line)

                    # Extract post ID from URL line
                    if header_line.startswith("# URL:"):
                        url = (
                            header_line.split(":", 1)[1].strip()
                            if ":" in header_line
                            else ""
                        )
                        current_post_id = extract_post_id_from_url(url)

                    # Check if we've hit the closing separator
                    if header_line.strip() and all(
                        c == "=" for c in header_line.strip()
                    ):
                        i += 1
                        break

                    i += 1

                # Read post content until next post separator
                while i < len(lines):
                    content_line = lines[i].rstrip("\n")

                    # Check if this is the start of a new post
                    if content_line.strip() and all(
                        c == "=" for c in content_line.strip()
                    ):
                        # Look ahead to confirm it's a new post
                        lookahead = i + 1
                        is_new_post = False
                        while lookahead < min(i + 5, len(lines)):
                            if lines[lookahead].strip().startswith(
                                "# Reddit Post"
                            ) or lines[lookahead].strip().startswith("# URL:"):
                                is_new_post = True
                                break
                            lookahead += 1

                        if is_new_post:
                            break

                    current_post.append(content_line)
                    i += 1

                continue

        i += 1

    # Don't forget the last post
    if current_post_id and current_post:
        posts.append((current_post_id, current_post))

    return posts


def deduplicate_posts(
    posts: List[Tuple[str, List[str]]],
) -> List[Tuple[str, List[str]]]:
    """Remove duplicate posts, keeping only the first occurrence"""
    seen_ids = set()
    unique_posts = []
    duplicates_removed = 0

    for post_id, post_lines in posts:
        if post_id and post_id in seen_ids:
            duplicates_removed += 1
            continue

        if post_id:
            seen_ids.add(post_id)
        unique_posts.append((post_id, post_lines))

    return unique_posts, duplicates_removed


def write_deduplicated_file(posts: List[Tuple[str, List[str]]], output_path: Path):
    """Write deduplicated posts to file"""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, (post_id, post_lines) in enumerate(posts, 1):
            # Update the post number in the header if it exists
            updated_lines = []
            for line in post_lines:
                if line.startswith("# Reddit Post"):
                    # Replace post number
                    updated_lines.append(f"# Reddit Post {i}")
                else:
                    updated_lines.append(line)

            # Write the post lines
            f.write("\n".join(updated_lines))
            f.write("\n\n")

    print(f"✅ Wrote {len(posts)} unique posts to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate Reddit data file by removing duplicate posts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Input Reddit data file to deduplicate",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file (default: input_file with _deduplicated suffix)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input file with deduplicated version (use with caution)",
    )

    args = parser.parse_args()

    input_file = Path(args.input_file)
    if not input_file.exists():
        print(f"❌ Error: File '{input_file}' not found.")
        return 1

    # Determine output file
    if args.in_place:
        output_file = input_file
        print(f"⚠️  Will overwrite {input_file} with deduplicated version")
    elif args.output:
        output_file = Path(args.output)
    else:
        # Add _deduplicated before extension
        stem = input_file.stem
        suffix = input_file.suffix
        output_file = input_file.parent / f"{stem}_deduplicated{suffix}"

    print(f"📖 Reading {input_file}...")
    posts = parse_reddit_file(input_file)
    print(f"   Found {len(posts)} posts in file")

    print("🔍 Deduplicating...")
    unique_posts, duplicates_removed = deduplicate_posts(posts)

    print(f"   Unique posts: {len(unique_posts)}")
    print(f"   Duplicates removed: {duplicates_removed}")

    if duplicates_removed == 0:
        print("   ✅ No duplicates found!")
        return 0

    print("💾 Writing deduplicated file...")
    write_deduplicated_file(unique_posts, output_file)

    # Show file size comparison
    input_size = input_file.stat().st_size
    output_size = output_file.stat().st_size
    reduction = ((input_size - output_size) / input_size) * 100

    print("\n📊 File size comparison:")
    print(f"   Original: {input_size / 1024:.1f} KB")
    print(f"   Deduplicated: {output_size / 1024:.1f} KB")
    print(f"   Reduction: {reduction:.1f}%")

    return 0


if __name__ == "__main__":
    exit(main())
