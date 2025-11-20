#!/usr/bin/env python3
"""
Collect Reddit data WITHOUT requiring API credentials.

This script uses Reddit's public JSON endpoints (no authentication needed).
Slower than authenticated API but works immediately.

Usage:
    python3 collect_reddit_data_no_auth.py [--limit N] [--output FILE]
"""

import json
import argparse
import time
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Set
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError


class TeeOutput:
    """Write stdout/stderr data to multiple streams (e.g., console + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def format_question(title: str, selftext: str) -> str:
    """Format Reddit post as a QUESTION"""
    if selftext and selftext.strip():
        question = f"{title}\n\n{selftext}".strip()
    else:
        question = title.strip()

    # Clean up formatting
    question = question.replace("**", "")
    question = question.replace("*", "")
    question = question.replace("&amp;", "&")
    question = question.replace("&lt;", "<")
    question = question.replace("&gt;", ">")

    return f"QUESTION: {question}"


def format_answer(comment_body: str) -> str:
    """Format Reddit comment as an ANSWER"""
    answer = comment_body.strip()
    answer = answer.replace("**", "")
    answer = answer.replace("*", "")
    answer = answer.replace("&amp;", "&")
    answer = answer.replace("&lt;", "<")
    answer = answer.replace("&gt;", ">")
    answer = answer.replace("[deleted]", "")
    answer = answer.replace("[removed]", "")

    return f"ANSWER: {answer}"


def get_json(url: str, retries: int = 5) -> Optional[dict]:
    """Fetch JSON from URL with retries and rate limit handling"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    for attempt in range(retries):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))
                return data
        except HTTPError as e:
            # Handle 429 specifically
            if e.code == 429:
                wait_time = min(2**attempt * 10, 300)  # Max 5 minutes
                print(f"  ⚠️  Rate limited (429). Waiting {wait_time}s before retry...")
                if attempt < retries - 1:
                    time.sleep(wait_time)
                    continue
                else:
                    print("  ❌ Rate limit exceeded. Please wait and try again later.")
                    return None
            elif attempt < retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"  ⚠️  HTTP {e.code} error, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  ❌ Failed to fetch {url}: HTTP {e.code}")
                return None
        except (URLError, json.JSONDecodeError) as e:
            if attempt < retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"  ⚠️  Error fetching {url}, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  ❌ Failed to fetch {url}: {e}")
                return None

    return None


def get_posts(
    subreddit: str,
    limit: int = 25,
    sort: str = "top",
    time_filter: str = "all",
    after: Optional[str] = None,
) -> tuple[list[dict], Optional[str]]:
    """Get posts from subreddit using public JSON endpoint

    Args:
        subreddit: Subreddit name
        limit: Number of posts to fetch per request
        sort: Sort method - "top", "hot", or "new"
        time_filter: Time filter for "top" sort (all, year, month, week, day)
        after: Pagination token

    Returns:
        tuple: (list of posts, after token for pagination)
    """
    # Build URL based on sort method
    if sort == "top":
        url = f"https://www.reddit.com/r/{subreddit}/top.json?limit={limit}&t={time_filter}"
    elif sort == "hot":
        url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
    elif sort == "new":
        url = f"https://www.reddit.com/r/{subreddit}/new.json?limit={limit}"
    else:
        # Default to top
        url = f"https://www.reddit.com/r/{subreddit}/top.json?limit={limit}&t={time_filter}"

    if after:
        url += f"&after={after}"

    data = get_json(url)

    if not data or "data" not in data or "children" not in data["data"]:
        return [], None

    posts = []
    for child in data["data"]["children"]:
        post_data = child["data"]
        posts.append(
            {
                "title": post_data.get("title", ""),
                "selftext": post_data.get("selftext", ""),
                "score": post_data.get("score", 0),
                "created_utc": post_data.get("created_utc", 0),
                "permalink": f"https://www.reddit.com{post_data.get('permalink', '')}",
                "id": post_data.get("id", ""),
                "name": post_data.get(
                    "name", ""
                ),  # Reddit's unique identifier (e.g., "t3_abc123")
            }
        )

    # Get the 'after' token for pagination (None if no more pages)
    after_token = data["data"].get("after")

    return posts, after_token


def get_post_comments(post_id: str, subreddit: str) -> list[dict]:
    """Get comments for a specific post"""
    url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
    data = get_json(url)

    if not data or len(data) < 2:
        return []

    # Comments are in the second element
    comments_data = data[1]["data"]["children"]
    comments = []

    def extract_comments(children, depth=0):
        """Recursively extract comments"""
        for child in children:
            if child["kind"] == "t1":  # Comment
                comment_data = child["data"]
                body = comment_data.get("body", "")
                if body and body not in ["[deleted]", "[removed]"]:
                    comments.append(
                        {
                            "body": body,
                            "score": comment_data.get("score", 0),
                            "depth": depth,
                        }
                    )
                # Recursively get replies
                if "replies" in comment_data and comment_data["replies"]:
                    extract_comments(
                        comment_data["replies"]["data"]["children"], depth + 1
                    )

    extract_comments(comments_data)
    return comments


def get_answer_snippet(answer: str, words: int = 40) -> str:
    """Get first N words of answer for logging"""
    words_list = answer.split()[:words]
    snippet = " ".join(words_list)
    if len(answer.split()) > words:
        snippet += "..."
    return snippet


def extract_post_id_from_url(url: str) -> Optional[str]:
    """Extract post ID from Reddit URL

    URL format: https://www.reddit.com/r/{subreddit}/comments/{post_id}/{title}/
    Returns the short post ID (e.g., 'a3zvze') or None if not found
    """
    try:
        # Extract the post ID from URL path
        # URL: https://www.reddit.com/r/relationships/comments/a3zvze/title/
        parts = url.split("/comments/")
        if len(parts) > 1:
            post_id_part = parts[1].split("/")[0]
            if post_id_part:
                return post_id_part
    except Exception:
        pass
    return None


def load_post_ids_from_output_files(reddit_dir: Path, subreddit: str) -> Set[str]:
    """Load post IDs from existing output files in the reddit directory"""
    post_ids = set()

    # Look for all output files for this subreddit
    pattern = f"reddit_{subreddit}_*.md"
    output_files = list(reddit_dir.glob(pattern))

    if not output_files:
        return post_ids

    print(f"  📂 Scanning {len(output_files)} existing output file(s) for post IDs...")

    for output_file in output_files:
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                content = f.read()
                # Find all URLs in the file
                url_pattern = (
                    r"# URL: (https://www\.reddit\.com/r/[^/]+/comments/[^/]+/[^\s]+)"
                )
                urls = re.findall(url_pattern, content)

                for url in urls:
                    post_id = extract_post_id_from_url(url)
                    if post_id:
                        post_ids.add(post_id)
        except Exception as e:
            print(f"  ⚠️  Could not read {output_file.name}: {e}")

    return post_ids


def load_existing_post_ids(
    log_file: Optional[Path], reddit_dir: Path, subreddit: str
) -> Set[str]:
    """Load post IDs from existing log file and output files to avoid duplicates"""
    post_ids = set()

    # First, load from existing output files (for backwards compatibility)
    output_post_ids = load_post_ids_from_output_files(reddit_dir, subreddit)
    post_ids.update(output_post_ids)

    # Then, load from log file if it exists
    if log_file and log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("Post ID:"):
                        # Extract post ID from line like "Post ID: t3_abc123" or short ID
                        parts = line.split(":", 1)
                        if len(parts) > 1:
                            post_id = parts[1].strip()
                            post_ids.add(post_id)
                    elif line.startswith("URL:"):
                        # Also extract from URL lines for backwards compatibility
                        url = line.split(":", 1)[1].strip() if ":" in line else ""
                        extracted_id = extract_post_id_from_url(url)
                        if extracted_id:
                            post_ids.add(extracted_id)
        except Exception as e:
            print(f"  ⚠️  Could not read existing log file: {e}")

    return post_ids


def write_to_log(log_file: Path, post_id: str, question: str, answer: str, url: str):
    """Write post info to log file"""
    snippet = get_answer_snippet(answer, 40)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Post ID: {post_id}\n")
        f.write(f"URL: {url}\n")
        f.write(f"Question: {question[:200]}{'...' if len(question) > 200 else ''}\n")
        f.write(f"Answer snippet (first 40 words): {snippet}\n")
        f.write(f"{'='*80}\n")


def collect_posts(
    subreddit: str = "relationships",
    limit: int = 100,
    min_upvotes: int = 10,
    min_comment_length: int = 100,
    sort: str = "top",
    time_filter: str = "all",
    start_at: int = 1,
    delay: float = 3.0,
    batch_delay: float = 10.0,
    log_file: Optional[Path] = None,
    skip_duplicates: bool = True,
) -> list[dict]:
    """Collect posts and comments from subreddit"""
    print(f"\nCollecting from r/{subreddit}...")
    print(f"  Target: {limit} posts")
    print(f"  Sort method: {sort}")
    print(f"  Starting at post: {start_at}")
    print(f"  Min upvotes: {min_upvotes}")
    if sort == "top":
        print(f"  Time filter: {time_filter}")
    print("  Note: Using public API (slower, no auth needed)")

    # Load existing post IDs to avoid duplicates
    existing_post_ids = set()
    if skip_duplicates:
        # Need to get reddit_dir and subreddit - these should be passed or derived
        # For now, we'll get them from log_file path
        if log_file:
            reddit_dir = log_file.parent
            # Extract subreddit from log filename or use default
            subreddit_name = subreddit  # Use the parameter
        else:
            reddit_dir = Path("sources/reddit")  # Default fallback
            subreddit_name = subreddit

        existing_post_ids = load_existing_post_ids(log_file, reddit_dir, subreddit_name)
        if existing_post_ids:
            print(
                f"  📋 Loaded {len(existing_post_ids)} existing post IDs (will skip duplicates)"
            )

    collected = []
    skipped = 0
    duplicates_skipped = 0
    posts_seen = 0  # Track total posts seen (including skipped)
    batch_size = 25  # Reddit JSON API limit per request
    after_token = None  # For pagination

    batch_num = 0
    while len(collected) < limit:
        batch_num += 1
        print(f"\n  Batch {batch_num}...")

        # Get batch of posts with pagination
        posts, after_token = get_posts(
            subreddit,
            limit=batch_size,
            sort=sort,
            time_filter=time_filter,
            after=after_token,
        )

        if not posts:
            print("  ⚠️  No posts returned, stopping")
            break

        # Note: We'll process this batch even if after_token is None (last page)
        if after_token is None:
            print("  ℹ️  This is the last page of results")

        for post in posts:
            posts_seen += 1

            # Skip posts before start_at
            if posts_seen < start_at:
                if posts_seen % 10 == 0:
                    print(
                        f"    Skipping post {posts_seen} (before start_at={start_at})..."
                    )
                continue

            # Check for duplicates using post ID
            # Reddit provides both "name" (t3_xxx) and "id" (short ID)
            post_name = post.get("name", "")  # Full ID like "t3_abc123"
            post_short_id = post.get("id", "")  # Short ID like "abc123"

            # Check both formats for duplicates
            is_duplicate = False
            if skip_duplicates:
                if post_name and post_name in existing_post_ids:
                    is_duplicate = True
                elif post_short_id and post_short_id in existing_post_ids:
                    is_duplicate = True
                # Also check if URL matches (extract ID from permalink)
                elif post.get("permalink"):
                    url_id = extract_post_id_from_url(post["permalink"])
                    if url_id and url_id in existing_post_ids:
                        is_duplicate = True

            if is_duplicate:
                duplicates_skipped += 1
                display_id = post_short_id or post_name[:20] or "unknown"
                print(
                    f"    Post {posts_seen}: ⚠️  DUPLICATE (ID: {display_id}...) - skipping"
                )
                continue

            # Use short ID for tracking (more reliable from URLs)
            post_id = post_short_id or post_name

            # Filter by upvotes
            if post["score"] < min_upvotes:
                skipped += 1
                continue

            # Skip if no content
            if not post["selftext"] and len(post["title"]) < 50:
                skipped += 1
                continue

            # Get comments for this post
            print(
                f"    Post {posts_seen}: Processing: {post['title'][:50]}... (score: {post['score']})"
            )
            comments = get_post_comments(post["id"], subreddit)

            if not comments:
                skipped += 1
                continue

            # Find best comment (highest score, sufficient length)
            valid_comments = [
                c for c in comments if len(c["body"]) >= min_comment_length
            ]

            if not valid_comments:
                skipped += 1
                continue

            # Sort by score and get top one
            valid_comments.sort(key=lambda x: x["score"], reverse=True)
            best_comment = valid_comments[0]["body"]

            # Format as Q&A
            question = format_question(post["title"], post["selftext"])
            answer = format_answer(best_comment)

            # Get post ID for duplicate tracking (use short ID, fallback to name)
            post_id = post.get("id", "") or post.get("name", "")

            collected.append(
                {
                    "question": question,
                    "answer": answer,
                    "score": post["score"],
                    "created_utc": post["created_utc"],
                    "url": post["permalink"],
                    "post_id": post_id,
                }
            )

            # Add to existing IDs set to avoid duplicates within this run
            # Store both short ID and full name if available
            if post_id:
                existing_post_ids.add(post_id)
            if post.get("name") and post.get("name") != post_id:
                existing_post_ids.add(post.get("name"))

            # Write to log file
            if log_file:
                write_to_log(log_file, post_id, question, answer, post["permalink"])

            print(
                f"      ✅ Collected (total: {len(collected)}, skipped: {skipped}, duplicates: {duplicates_skipped})"
            )

            # Rate limiting - be more respectful to avoid 429 errors
            # Reddit's public API is stricter, so we need longer delays
            time.sleep(delay)

            # Stop if we have enough collected posts
            if len(collected) >= limit:
                break

        # Rate limiting between batches - longer delay to avoid rate limits
        # Only delay if we're continuing (have more posts to collect and after_token exists)
        if len(collected) < limit and after_token is not None:
            print(
                f"  ⏸️  Waiting {batch_delay} seconds before next batch (to avoid rate limits)..."
            )
            time.sleep(batch_delay)
        elif after_token is None:
            # No more pages, break after processing this batch
            print("\n  ℹ️  Reached end of available posts (after_token is None)")
            print(f"     Collected {len(collected)}/{limit} posts so far")
            if len(collected) < limit:
                print(f"     ⚠️  Could not reach target of {limit} posts.")
                print("     💡 Suggestions:")
                print(f"        - Lower --min-upvotes (currently {min_upvotes})")
                print(f"        - Use longer --time-filter (currently {time_filter})")
                print("        - Try --time-filter year or --time-filter all")
            break

    print(
        f"\n✅ Collected {len(collected)} posts (skipped {skipped}, duplicates skipped: {duplicates_skipped}, started at post {start_at})"
    )
    if log_file:
        print(f"   📋 Log file: {log_file}")
    return collected


def save_to_file(collected: list[dict], output_file: Path):
    """Save collected data to file"""
    with open(output_file, "w", encoding="utf-8") as f:
        for i, item in enumerate(collected, 1):
            f.write(f"\n{'='*80}\n")
            f.write(f"# Reddit Post {i}\n")
            f.write(
                f"# Score: {item['score']} | Date: {datetime.fromtimestamp(item['created_utc']).strftime('%Y-%m-%d')}\n"
            )
            f.write(f"# URL: {item['url']}\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"{item['question']}\n\n")
            f.write(f"{item['answer']}\n\n")

    print(f"✅ Saved {len(collected)} Q&A pairs to {output_file}")
    file_size = output_file.stat().st_size
    print(f"   File size: {file_size / 1024:.1f} KB ({file_size / (1024*1024):.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Collect Reddit data WITHOUT API credentials (uses public JSON)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of posts to collect (default: 50, note: slower without auth)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path",
    )
    parser.add_argument(
        "--min-upvotes",
        type=int,
        default=10,
        help="Minimum upvotes required (default: 10)",
    )
    parser.add_argument(
        "--min-comment-length",
        type=int,
        default=100,
        help="Minimum comment length (default: 100)",
    )
    parser.add_argument(
        "--sort",
        choices=["top", "hot", "new"],
        default="top",
        help="Sort method: 'top' (highest scoring), 'hot' (currently trending), or 'new' (most recent). Use 'hot' or 'new' to get different posts than previous runs. (default: top)",
    )
    parser.add_argument(
        "--time-filter",
        choices=["all", "year", "month", "week", "day"],
        default="all",
        help="Time filter for 'top' sort only (default: all)",
    )
    parser.add_argument(
        "--subreddit",
        default="relationships",
        help="Subreddit name (default: relationships)",
    )
    parser.add_argument(
        "--start-at",
        type=int,
        default=1,
        help="Start collecting from this post number (1-indexed, default: 1)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Delay between posts in seconds (default: 3.0, increase if getting 429 errors)",
    )
    parser.add_argument(
        "--batch-delay",
        type=float,
        default=10.0,
        help="Delay between batches in seconds (default: 10.0, increase if getting 429 errors)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Log file to track collected posts (default: reddit_{subreddit}_log.txt in reddit directory)",
    )
    parser.add_argument(
        "--no-skip-duplicates",
        action="store_true",
        help="Disable duplicate detection (not recommended)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for run logs (default: <repo_root>/logs)",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Disable writing console output to a log file",
    )

    args = parser.parse_args()

    # Determine output file
    script_dir = Path(__file__).parent
    sources_dir = script_dir.parent  # Go up from scripts/ to sources/
    reddit_dir = sources_dir / "reddit"
    reddit_dir.mkdir(exist_ok=True)

    if args.output:
        output_file = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d")
        output_file = reddit_dir / f"reddit_{args.subreddit}_{timestamp}.md"

    # Determine log file for collected post metadata
    if args.log_file:
        log_file = Path(args.log_file)
    else:
        log_file = reddit_dir / f"reddit_{args.subreddit}_log.txt"

    repo_root = script_dir.parent.parent
    if args.log_dir:
        log_dir = Path(args.log_dir)
        if not log_dir.is_absolute():
            log_dir = repo_root / log_dir
    else:
        log_dir = repo_root / "logs"

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    tee_log_handle = None

    try:
        if not args.no_log_file:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = (
                log_dir
                / f"reddit_collect_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            tee_log_handle = open(log_path, "w", encoding="utf-8")
            sys.stdout = TeeOutput(original_stdout, tee_log_handle)
            sys.stderr = TeeOutput(original_stderr, tee_log_handle)
            print(f"📝 Logging output to {log_path}")

        print("=" * 60)
        print("Reddit Data Collector (No Auth Required)")
        print("=" * 60)
        print("This script uses Reddit's public JSON endpoints.")
        print("No API credentials needed, but it's slower.")
        print("=" * 60)

        # Collect posts
        collected = collect_posts(
            subreddit=args.subreddit,
            limit=args.limit,
            min_upvotes=args.min_upvotes,
            min_comment_length=args.min_comment_length,
            sort=args.sort,
            time_filter=args.time_filter,
            start_at=args.start_at,
            delay=args.delay,
            batch_delay=args.batch_delay,
            log_file=log_file,
            skip_duplicates=not args.no_skip_duplicates,
        )

        if not collected:
            print("\n❌ No posts collected. Try:")
            print("  - Lowering --min-upvotes")
            print("  - Lowering --min-comment-length")
            print("  - Checking if subreddit exists")
            sys.exit(1)

        # Save to file
        save_to_file(collected, output_file)

        print(f"\n✅ Done! Collected {len(collected)} Q&A pairs")
        print(f"   Output: {output_file}")
        print("\nNote: This method is slower than authenticated API.")
        print("For faster collection, try getting Reddit API access.")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if tee_log_handle:
            tee_log_handle.close()


if __name__ == "__main__":
    main()
