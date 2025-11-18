#!/usr/bin/env python3
"""
Find missing Carolyn Hax chat dates to help identify what data to collect.

This script:
- Analyzes existing chat files
- Identifies gaps in the date sequence
- Suggests dates to collect
- Estimates how much data you have vs. need

Usage:
    python3 find_missing_dates.py [--chats-dir DIR] [--target-size MB]
"""

import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict


def parse_date_from_filename(filename: str) -> datetime | None:
    """Extract date from filename like 'carolyn_hax_103125_chat.md'"""
    match = re.search(r"carolyn_hax_(\d{6})_chat\.md", filename)
    if not match:
        return None
    
    date_str = match.group(1)
    try:
        return datetime.strptime(date_str, "%m%d%y")
    except ValueError:
        return None


def get_file_size(filepath: Path) -> int:
    """Get file size in bytes"""
    try:
        return filepath.stat().st_size
    except OSError:
        return 0


def analyze_chats(chats_dir: Path) -> dict:
    """Analyze existing chat files and return statistics"""
    stats = {
        'files': [],
        'dates': [],
        'total_size': 0,
        'non_empty_count': 0,
        'empty_count': 0,
        'by_year': defaultdict(list),
        'by_month': defaultdict(list),
    }
    
    for filepath in sorted(chats_dir.glob("carolyn_hax_*_chat.md")):
        filename = filepath.name
        date = parse_date_from_filename(filename)
        size = get_file_size(filepath)
        
        file_info = {
            'filename': filename,
            'filepath': filepath,
            'date': date,
            'size': size,
            'is_empty': size == 0,
        }
        
        stats['files'].append(file_info)
        stats['total_size'] += size
        
        if size > 0:
            stats['non_empty_count'] += 1
            if date:
                stats['dates'].append(date)
                stats['by_year'][date.year].append(file_info)
                stats['by_month'][f"{date.year}-{date.month:02d}"].append(file_info)
        else:
            stats['empty_count'] += 1
    
    stats['dates'].sort()
    return stats


def find_friday_dates(start_date: datetime, end_date: datetime) -> list[datetime]:
    """Find all Fridays between start and end dates"""
    fridays = []
    current = start_date
    
    # Find first Friday
    while current.weekday() != 4:  # 4 = Friday
        current += timedelta(days=1)
    
    # Collect all Fridays
    while current <= end_date:
        fridays.append(current)
        current += timedelta(days=7)
    
    return fridays


def find_missing_dates(existing_dates: list[datetime], start_date: datetime, end_date: datetime) -> list[datetime]:
    """Find Friday dates that are missing from the collection"""
    existing_set = set(existing_dates)
    all_fridays = find_friday_dates(start_date, end_date)
    
    missing = [d for d in all_fridays if d not in existing_set]
    return sorted(missing)


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Find missing Carolyn Hax chat dates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--chats-dir",
        type=Path,
        default=None,
        help="Directory containing chat files (default: sources/carolyn_hax_chats)",
    )
    parser.add_argument(
        "--target-size",
        type=float,
        default=5.0,
        help="Target dataset size in MB (default: 5.0)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2024,
        help="Start year for missing date search (default: 2024)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="End year for missing date search (default: 2025)",
    )
    
    args = parser.parse_args()
    
    # Determine chats directory
    script_dir = Path(__file__).parent
    if args.chats_dir:
        chats_dir = args.chats_dir
    else:
        chats_dir = script_dir / "carolyn_hax_chats"
    
    if not chats_dir.exists():
        print(f"Error: Directory '{chats_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    # Analyze existing chats
    print("Analyzing existing chat files...")
    stats = analyze_chats(chats_dir)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total files: {len(stats['files'])}")
    print(f"Non-empty files: {stats['non_empty_count']}")
    print(f"Empty files: {stats['empty_count']}")
    print(f"Total size: {format_size(stats['total_size'])}")
    print(f"Average file size: {format_size(stats['total_size'] / max(stats['non_empty_count'], 1))}")
    
    if stats['dates']:
        print("\nDate range:")
        print(f"  Earliest: {min(stats['dates']).strftime('%B %d, %Y')}")
        print(f"  Latest: {max(stats['dates']).strftime('%B %d, %Y')}")
    
    # Calculate gap to target
    current_mb = stats['total_size'] / (1024 * 1024)
    target_mb = args.target_size
    gap_mb = max(0, target_mb - current_mb)
    
    print("\nSize analysis:")
    print(f"  Current: {current_mb:.2f} MB")
    print(f"  Target: {target_mb:.2f} MB")
    print(f"  Gap: {gap_mb:.2f} MB")
    
    if stats['non_empty_count'] > 0:
        avg_mb_per_file = current_mb / stats['non_empty_count']
        files_needed = int(gap_mb / avg_mb_per_file) if avg_mb_per_file > 0 else 0
        print(f"  Estimated files needed: ~{files_needed} (based on current average)")
    
    # Find missing dates
    if stats['dates']:
        start_date = datetime(args.start_year, 1, 1)
        end_date = datetime(args.end_year, 12, 31)
        missing = find_missing_dates(stats['dates'], start_date, end_date)
        
        print("\n" + "=" * 60)
        print(f"MISSING FRIDAY CHATS ({args.start_year}-{args.end_year})")
        print("=" * 60)
        
        if missing:
            print(f"Found {len(missing)} missing Friday dates:\n")
            
            # Group by year/month for readability
            by_year_month = defaultdict(list)
            for date in missing:
                key = f"{date.year}-{date.month:02d}"
                by_year_month[key].append(date)
            
            for year_month in sorted(by_year_month.keys()):
                dates = by_year_month[year_month]
                print(f"{year_month}: {len(dates)} missing")
                for date in dates[:5]:  # Show first 5
                    filename = f"carolyn_hax_{date.strftime('%m%d%y')}_chat.md"
                    print(f"  - {date.strftime('%B %d, %Y')} ({filename})")
                if len(dates) > 5:
                    print(f"  ... and {len(dates) - 5} more")
                print()
            
            # Show empty files that could be filled
            empty_files = [f for f in stats['files'] if f['is_empty']]
            if empty_files:
                print(f"\nEmpty files that need content ({len(empty_files)}):")
                for file_info in empty_files[:10]:
                    print(f"  - {file_info['filename']}")
                if len(empty_files) > 10:
                    print(f"  ... and {len(empty_files) - 10} more")
        else:
            print("No missing Friday dates found in the specified range!")
            print("(All Fridays in range are already collected)")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if gap_mb > 0:
        print(f"1. Collect {gap_mb:.1f} MB more data to reach target")
        if missing:
            print(f"2. Start with {min(20, len(missing))} missing chats (easiest to find)")
        if empty_files:
            print(f"3. Fill {len(empty_files)} empty files first")
        print("4. Focus on recent dates (easier to find online)")
        print("5. After collecting, run merge and clean scripts")
    else:
        print("✅ You've reached your target size!")
        print("Consider collecting more for even better results (10MB+ ideal)")
    
    print("\nNext steps:")
    print("1. Visit: https://www.washingtonpost.com/advice/ask-carolyn-hax/")
    print("2. Find missing chat dates from the list above")
    print("3. Copy chat content and save as carolyn_hax_MMDDYY_chat.md")
    print("4. Run: python3 merge_carolyn_hax_chats.py")
    print("5. Run: python3 clean_carolyn_hax_data.py")


if __name__ == "__main__":
    main()

