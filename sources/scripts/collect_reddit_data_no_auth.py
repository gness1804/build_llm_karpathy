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
from pathlib import Path
from datetime import datetime
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError


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


def get_top_posts(
    subreddit: str, limit: int = 25, time_filter: str = "all"
) -> list[dict]:
    """Get top posts from subreddit using public JSON endpoint"""
    url = f"https://www.reddit.com/r/{subreddit}/top.json?limit={limit}&t={time_filter}"
    data = get_json(url)

    if not data or "data" not in data or "children" not in data["data"]:
        return []

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
            }
        )

    return posts


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


def collect_posts(
    subreddit: str = "relationships",
    limit: int = 100,
    min_upvotes: int = 10,
    min_comment_length: int = 100,
    time_filter: str = "all",
    start_at: int = 1,
    delay: float = 3.0,
    batch_delay: float = 10.0,
) -> list[dict]:
    """Collect posts and comments from subreddit"""
    print(f"\nCollecting from r/{subreddit}...")
    print(f"  Target: {limit} posts")
    print(f"  Starting at post: {start_at}")
    print(f"  Min upvotes: {min_upvotes}")
    print(f"  Time filter: {time_filter}")
    print("  Note: Using public API (slower, no auth needed)")

    collected = []
    skipped = 0
    posts_seen = 0  # Track total posts seen (including skipped)
    batch_size = 25  # Reddit JSON API limit per request

    # Calculate how many posts we need to fetch (accounting for start_at)
    # We need to fetch enough to get 'limit' posts after starting at 'start_at'
    total_posts_needed = start_at + limit - 1
    batches_needed = (total_posts_needed + batch_size - 1) // batch_size

    for batch_num in range(batches_needed):
        print(f"\n  Batch {batch_num + 1}/{batches_needed}...")

        # Get batch of posts
        posts = get_top_posts(subreddit, limit=batch_size, time_filter=time_filter)

        if not posts:
            print("  ⚠️  No posts returned, stopping")
            break

        for post in posts:
            posts_seen += 1

            # Skip posts before start_at
            if posts_seen < start_at:
                if posts_seen % 10 == 0:
                    print(
                        f"    Skipping post {posts_seen} (before start_at={start_at})..."
                    )
                continue

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

            collected.append(
                {
                    "question": question,
                    "answer": answer,
                    "score": post["score"],
                    "created_utc": post["created_utc"],
                    "url": post["permalink"],
                }
            )

            print(f"      ✅ Collected (total: {len(collected)}, skipped: {skipped})")

            # Rate limiting - be more respectful to avoid 429 errors
            # Reddit's public API is stricter, so we need longer delays
            time.sleep(delay)

            # Stop if we have enough collected posts
            if len(collected) >= limit:
                break

        # Rate limiting between batches - longer delay to avoid rate limits
        if batch_num < batches_needed - 1:
            print(
                f"  ⏸️  Waiting {batch_delay} seconds before next batch (to avoid rate limits)..."
            )
            time.sleep(batch_delay)

        # Stop if we have enough collected posts
        if len(collected) >= limit:
            break

    print(
        f"\n✅ Collected {len(collected)} posts (skipped {skipped}, started at post {start_at})"
    )
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
        "--time-filter",
        choices=["all", "year", "month", "week", "day"],
        default="all",
        help="Time filter (default: all)",
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

    args = parser.parse_args()

    # Determine output file
    if args.output:
        output_file = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d")
        script_dir = Path(__file__).parent
        sources_dir = script_dir.parent  # Go up from scripts/ to sources/
        reddit_dir = sources_dir / "reddit"
        reddit_dir.mkdir(exist_ok=True)
        output_file = reddit_dir / f"reddit_{args.subreddit}_{timestamp}.md"

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
        time_filter=args.time_filter,
        start_at=args.start_at,
        delay=args.delay,
        batch_delay=args.batch_delay,
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


if __name__ == "__main__":
    main()
