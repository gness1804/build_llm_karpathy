#!/usr/bin/env python3
"""
Clean up and deduplicate Reddit data files.

This script:
1. Analyzes all .md files to find duplicate posts (by URL/post ID)
2. Merges unique posts into consolidated files per subreddit
3. Keeps only the main log file
4. Moves old/duplicate files to archive
"""

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple


def extract_post_id_from_url(url: str) -> str | None:
    """Extract post ID from Reddit URL"""
    try:
        parts = url.split("/comments/")
        if len(parts) > 1:
            post_id_part = parts[1].split("/")[0]
            if post_id_part:
                return post_id_part
    except Exception:
        pass
    return None


def extract_posts_from_file(file_path: Path) -> Tuple[Dict[str, str], Set[str]]:
    """
    Extract posts from a Reddit data file.

    Returns:
        tuple: (dict mapping post_id to full post content, set of post IDs)
    """
    posts = {}
    post_ids = set()

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"  ⚠️  Error reading {file_path.name}: {e}")
        return posts, post_ids

    # Find all post sections (between ====== dividers)
    # Split by the divider pattern - this gives us sections between dividers
    sections = re.split(r"={80,}", content)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract URL from section to get post_id
        url_match = re.search(r"# URL: (https://www\.reddit\.com/r/[^\s]+)", section)
        if url_match:
            url = url_match.group(1)
            post_id = extract_post_id_from_url(url)

            if post_id:
                # Store the entire section (includes metadata + QUESTION + ANSWER)
                post_ids.add(post_id)
                posts[post_id] = section

    return posts, post_ids


def analyze_files(
    reddit_dir: Path,
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Set[str]]]:
    """
    Analyze all .md files in the directory.

    Returns:
        tuple: (
            dict mapping subreddit -> post_id -> content,
            dict mapping subreddit -> set of post_ids
        )
    """
    all_posts = defaultdict(dict)  # subreddit -> post_id -> content
    all_post_ids = defaultdict(set)  # subreddit -> set of post_ids
    file_info = []  # List of (file_path, subreddit, post_count)

    md_files = list(reddit_dir.glob("*.md"))

    print(f"📊 Analyzing {len(md_files)} data files...")

    for file_path in md_files:
        # Skip archive files and already consolidated files
        if "archive" in str(file_path) or "consolidated" in file_path.name:
            continue

        # Determine subreddit from filename
        filename = file_path.stem
        if "relationship_advice" in filename:
            subreddit = "relationship_advice"
        elif "relationships" in filename:
            subreddit = "relationships"
        else:
            subreddit = "unknown"

        posts, post_ids = extract_posts_from_file(file_path)

        if posts:
            # Merge into consolidated dict (later files overwrite earlier ones)
            for post_id, content in posts.items():
                all_posts[subreddit][post_id] = content
                all_post_ids[subreddit].add(post_id)

            file_info.append((file_path, subreddit, len(posts)))
            print(f"  📄 {file_path.name}: {len(posts)} posts ({subreddit})")

    return all_posts, all_post_ids, file_info


def write_consolidated_file(
    reddit_dir: Path, subreddit: str, posts: Dict[str, str], output_name: str
) -> Path:
    """Write consolidated posts to a single file"""
    output_path = reddit_dir / output_name

    print(f"\n💾 Writing consolidated file: {output_name}")
    print(f"   {len(posts)} unique posts")

    with open(output_path, "w", encoding="utf-8") as f:
        for i, (post_id, content) in enumerate(sorted(posts.items()), 1):
            f.write(f"\n{'='*80}\n")
            f.write(f"# Reddit Post {i}\n")
            f.write(content)
            f.write(f"\n{'='*80}\n\n")

    file_size = output_path.stat().st_size
    print(f"   File size: {file_size / 1024:.1f} KB ({file_size / (1024*1024):.2f} MB)")

    return output_path


def cleanup_reddit_directory(reddit_dir: Path, dry_run: bool = False):
    """Clean up and consolidate Reddit data files"""
    print("=" * 60)
    print("Reddit Data Cleanup")
    print("=" * 60)

    # Analyze all files
    all_posts, all_post_ids, file_info = analyze_files(reddit_dir)

    if not all_posts:
        print("\n❌ No posts found in any files!")
        return

    # Summary
    print("\n📊 Summary:")
    for subreddit, post_ids in all_post_ids.items():
        print(f"   r/{subreddit}: {len(post_ids)} unique posts")

    # Create archive directory
    archive_dir = reddit_dir / "archive"
    archive_dir.mkdir(exist_ok=True)

    # Consolidate files per subreddit
    consolidated_files = []

    for subreddit, posts in all_posts.items():
        if not posts:
            continue

        output_name = f"reddit_{subreddit}_consolidated.md"
        output_path = write_consolidated_file(reddit_dir, subreddit, posts, output_name)
        consolidated_files.append(output_path)

    # Identify files to archive
    files_to_archive = []
    log_files = list(reddit_dir.glob("*.txt"))

    for file_path, subreddit, post_count in file_info:
        if file_path.suffix != ".txt":
            files_to_archive.append(file_path)

    # Keep the most recent log file (relationship_advice if it exists)
    main_log = None
    for log_file in log_files:
        if "relationship_advice" in log_file.name:
            main_log = log_file
            break

    if not main_log and log_files:
        # Use the most recent log file
        main_log = max(log_files, key=lambda p: p.stat().st_mtime)

    print("\n📋 Log files:")
    for log_file in log_files:
        if log_file == main_log:
            print(f"   ✅ KEEPING: {log_file.name} (main log)")
        else:
            print(f"   📦 ARCHIVE: {log_file.name}")
            if not dry_run:
                shutil.move(str(log_file), str(archive_dir / log_file.name))

    # Archive old data files
    print("\n📦 Archiving old data files:")
    for file_path in files_to_archive:
        if file_path in consolidated_files:
            continue  # Don't archive the new consolidated files

        print(f"   📦 {file_path.name}")
        if not dry_run:
            shutil.move(str(file_path), str(archive_dir / file_path.name))

    # Clean up tmp directory if it exists
    tmp_dir = reddit_dir / "tmp"
    if tmp_dir.exists():
        print("\n🗑️  Cleaning up tmp directory...")
        if not dry_run:
            shutil.rmtree(tmp_dir)
        print("   ✅ Removed tmp/")

    print("\n✅ Cleanup complete!")
    print(f"   Consolidated files: {len(consolidated_files)}")
    print(
        f"   Archived files: {len(files_to_archive) + len(log_files) - (1 if main_log else 0)}"
    )

    if dry_run:
        print("\n⚠️  DRY RUN - No files were moved. Run without --dry-run to execute.")


def main():
    parser = argparse.ArgumentParser(
        description="Clean up and deduplicate Reddit data files"
    )
    parser.add_argument(
        "--reddit-dir",
        type=Path,
        default=Path("sources/reddit"),
        help="Reddit data directory (default: sources/reddit)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually moving files",
    )

    args = parser.parse_args()

    # Resolve path relative to script location
    script_dir = Path(__file__).parent
    if args.reddit_dir.is_absolute():
        reddit_dir = args.reddit_dir
    else:
        # Script is in sources/scripts/, reddit_dir is relative to sources/
        # So we go up one level from scripts/ to sources/, then use the path
        reddit_dir = script_dir.parent / args.reddit_dir

    if not reddit_dir.exists():
        print(f"❌ Error: Directory {reddit_dir} does not exist")
        return

    cleanup_reddit_directory(reddit_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
