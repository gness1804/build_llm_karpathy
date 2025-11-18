#!/usr/bin/env python3
"""
Collect relationship advice data from Reddit r/relationships subreddit.

This script:
- Uses Reddit API (PRAW) to collect posts and top comments
- Filters for quality (upvotes, comment length)
- Converts to QUESTION/ANSWER format compatible with training
- Saves in format similar to Carolyn Hax chats

Requirements:
    pip install praw

Setup:
    1. Create Reddit app at https://www.reddit.com/prefs/apps
    2. Get client_id, client_secret, user_agent
    3. Set as environment variables or in .env file

Usage:
    python3 collect_reddit_data.py [--limit N] [--output FILE] [--min-upvotes N]
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

try:
    import praw
except ImportError:
    print("Error: praw library not installed.")
    print("Install with: pip install praw")
    sys.exit(1)


def get_reddit_client():
    """Initialize Reddit API client"""
    # Try environment variables first
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent = os.environ.get("REDDIT_USER_AGENT", "LLM Training Data Collector 1.0")

    if not client_id or not client_secret:
        print("Error: Reddit API credentials not found.")
        print("\nTo get credentials:")
        print("1. Go to https://www.reddit.com/prefs/apps")
        print("2. Click 'create another app...'")
        print("3. Choose 'script' type")
        print("4. Set redirect URI to http://localhost:8080")
        print("5. Copy client_id (under the app name) and client_secret")
        print("\nThen set environment variables:")
        print("  export REDDIT_CLIENT_ID='your_client_id'")
        print("  export REDDIT_CLIENT_SECRET='your_client_secret'")
        print("  export REDDIT_USER_AGENT='Your App Name 1.0'")
        sys.exit(1)

    reddit = praw.Reddit(
        client_id=client_id, client_secret=client_secret, user_agent=user_agent
    )

    # Test connection
    try:
        reddit.user.me()  # This will be None for read-only, but tests connection
        print("✅ Connected to Reddit API")
    except Exception as e:
        print(f"⚠️  Warning: Could not verify Reddit connection: {e}")
        print("   Continuing anyway (read-only access should work)...")

    return reddit


def format_question(title: str, selftext: str) -> str:
    """Format Reddit post as a QUESTION"""
    # Combine title and selftext
    if selftext and selftext.strip():
        question = f"{title}\n\n{selftext}".strip()
    else:
        question = title.strip()

    # Clean up common Reddit formatting
    question = question.replace("**", "")  # Remove bold markers
    question = question.replace("*", "")  # Remove italics
    question = question.replace("&amp;", "&")
    question = question.replace("&lt;", "<")
    question = question.replace("&gt;", ">")

    return f"QUESTION: {question}"


def format_answer(comment_body: str) -> str:
    """Format Reddit comment as an ANSWER"""
    # Clean up Reddit formatting
    answer = comment_body.strip()
    answer = answer.replace("**", "")  # Remove bold markers
    answer = answer.replace("*", "")  # Remove italics
    answer = answer.replace("&amp;", "&")
    answer = answer.replace("&lt;", "<")
    answer = answer.replace("&gt;", ">")

    # Remove common Reddit phrases
    answer = answer.replace("[deleted]", "")
    answer = answer.replace("[removed]", "")

    return f"ANSWER: {answer}"


def get_best_advice_comment(post, min_length: int = 100) -> Optional[str]:
    """Get the best advice comment from a post"""
    # Sort comments by score (upvotes)
    post.comments.replace_more(limit=0)  # Remove "load more comments" placeholders
    comments = post.comments.list()

    # Filter and sort comments
    valid_comments = []
    for comment in comments:
        # Skip deleted/removed
        if hasattr(comment, "body") and comment.body not in ["[deleted]", "[removed]"]:
            # Filter by length (substantial advice)
            if len(comment.body) >= min_length:
                valid_comments.append((comment.score, comment.body))

    if not valid_comments:
        return None

    # Return highest-scored comment
    valid_comments.sort(reverse=True, key=lambda x: x[0])
    return valid_comments[0][1]


def collect_posts(
    reddit: praw.Reddit,
    subreddit_name: str = "relationships",
    limit: int = 100,
    min_upvotes: int = 10,
    min_comment_length: int = 100,
    sort_by: str = "top",
    time_filter: str = "all",
) -> list[dict]:
    """Collect posts from subreddit"""
    print(f"\nCollecting from r/{subreddit_name}...")
    print(f"  Limit: {limit} posts")
    print(f"  Min upvotes: {min_upvotes}")
    print(f"  Sort by: {sort_by}")
    print(f"  Time filter: {time_filter}")

    subreddit = reddit.subreddit(subreddit_name)

    # Get posts based on sort method
    if sort_by == "top":
        posts = subreddit.top(limit=limit, time_filter=time_filter)
    elif sort_by == "hot":
        posts = subreddit.hot(limit=limit)
    elif sort_by == "new":
        posts = subreddit.new(limit=limit)
    else:
        posts = subreddit.top(limit=limit, time_filter=time_filter)

    collected = []
    skipped = 0

    for i, post in enumerate(posts, 1):
        # Filter by upvotes
        if post.score < min_upvotes:
            skipped += 1
            continue

        # Skip stickied posts (usually mod announcements)
        if post.stickied:
            skipped += 1
            continue

        # Skip if no selftext and title is too short
        if not post.selftext and len(post.title) < 50:
            skipped += 1
            continue

        # Get best advice comment
        advice = get_best_advice_comment(post, min_comment_length)
        if not advice:
            skipped += 1
            continue

        # Format as Q&A
        question = format_question(post.title, post.selftext)
        answer = format_answer(advice)

        collected.append(
            {
                "question": question,
                "answer": answer,
                "score": post.score,
                "created_utc": post.created_utc,
                "url": post.url,
            }
        )

        if i % 10 == 0:
            print(f"  Collected {len(collected)} posts (skipped {skipped})...")

    print(f"\n✅ Collected {len(collected)} posts (skipped {skipped})")
    return collected


def save_to_file(collected: list[dict], output_file: Path):
    """Save collected data to file in training format"""
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
        description="Collect relationship advice from Reddit r/relationships",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of posts to collect (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: reddit_relationships_YYYYMMDD.md)",
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
        help="Minimum comment length in characters (default: 100)",
    )
    parser.add_argument(
        "--sort",
        choices=["top", "hot", "new"],
        default="top",
        help="Sort method (default: top)",
    )
    parser.add_argument(
        "--time-filter",
        choices=["all", "year", "month", "week", "day"],
        default="all",
        help="Time filter for 'top' sort (default: all)",
    )
    parser.add_argument(
        "--subreddit",
        default="relationships",
        help="Subreddit name (default: relationships)",
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

    # Initialize Reddit client
    reddit = get_reddit_client()

    # Collect posts
    collected = collect_posts(
        reddit=reddit,
        subreddit_name=args.subreddit,
        limit=args.limit,
        min_upvotes=args.min_upvotes,
        min_comment_length=args.min_comment_length,
        sort_by=args.sort,
        time_filter=args.time_filter,
    )

    if not collected:
        print("\n❌ No posts collected. Try:")
        print("  - Lowering --min-upvotes")
        print("  - Lowering --min-comment-length")
        print("  - Increasing --limit")
        sys.exit(1)

    # Save to file
    save_to_file(collected, output_file)

    print(f"\n✅ Done! Collected {len(collected)} Q&A pairs")
    print(f"   Output: {output_file}")
    print("\nNext steps:")
    print("  1. Review the output file")
    print(
        f"  2. Optionally clean with: python3 clean_carolyn_hax_data.py {output_file.name}"
    )
    print("  3. Merge with Carolyn Hax data or use separately")


if __name__ == "__main__":
    main()
