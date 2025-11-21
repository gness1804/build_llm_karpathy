# Handoff: Reddit Data Collection and Dataset Expansion

**Date**: November 20, 2025  
**Status**: Dataset Expanded to 4.3MB, Data Collection Infrastructure Complete  
**Working Directory**: `~/Desktop/build_llm_karpathy`

## Project Overview

This project fine-tunes GPT-2 to generate advice in the style of Carolyn Hax. The focus has shifted from training to **expanding the dataset** from ~1MB to 5-10MB to improve model performance. We've built a comprehensive data collection and management pipeline.

**Key Objectives:**
- Expand training dataset from 1MB to 5-10MB
- Collect Reddit relationship advice posts (Q&A format)
- Maintain clean, deduplicated dataset
- Prepare for fresh training run with larger dataset

## Current State

### ✅ Completed

1. **Fixed Critical Bugs**:
   - **Pagination Bug**: Fixed Reddit collector to properly use `after` token for pagination (was stuck at 25 posts)
   - **Duplicate Detection**: Improved duplicate detection across multiple collection runs
   - **Sort Methods**: Added support for `hot`, `new`, and `top` sort methods to avoid duplicate posts

2. **Data Collection Infrastructure**:
   - **Collection Script** (`sources/scripts/collect_reddit_data_no_auth.py`):
     - Supports `--sort hot/new/top` to get different posts each run
     - Automatic logging to `logs/` directory (tee-style, shows in terminal + saves to file)
     - Proper pagination with `after` token
     - Duplicate detection based on post IDs
   - **Merge Script** (`sources/scripts/merge_training_data.py`):
     - Merges Reddit data with training dataset
     - Strips metadata headers automatically
     - Interactive archiving prompt (moves source files to archive after merge)
     - Supports `--archive` and `--no-archive` flags
   - **Cleanup Script** (`sources/scripts/cleanup_reddit_data.py`):
     - Consolidates duplicate posts across multiple Reddit files
     - Archives old files automatically
     - Keeps directory clean with only unique consolidated files

3. **Dataset Expansion**:
   - **Original**: `sources/training_data_final.md` (1.0 MB, 750 Q&A pairs)
   - **Current**: `sources/training_data_final_merged.md` (4.3 MB, ~1,486 Q&A pairs)
   - **Progress**: Expanded from 1MB → 4.3MB (target: 5-10MB)

4. **Directory Organization**:
   - Cleaned up `sources/reddit/` directory
   - Consolidated files: `reddit_relationship_advice_consolidated.md`, `reddit_relationships_consolidated.md`
   - Archived old/duplicate files to `sources/reddit/archive/`
   - Main log file: `sources/reddit/reddit_relationship_advice_log.txt`

### 🚧 In Progress / Current Challenges

1. **Data Collection Efficiency**:
   - Most posts are being filtered out by `--min-upvotes` and `--min-comment-length`
   - With `--sort new`, posts are very recent and have low upvotes (0-5)
   - With `--sort hot`, many posts still don't meet `min_upvotes=10`
   - **Solution**: Lower filters when using `new` or `hot` sort

2. **Dataset Size**:
   - Current: 4.3 MB (target: 5-10 MB)
   - Need ~1-6 MB more data
   - Continue collecting with adjusted filters

## Next Steps (Immediate)

### Priority 1: Collect More Data with Adjusted Filters

**For "new" sort** (most recent posts):
```bash
python3 sources/scripts/collect_reddit_data_no_auth.py \
  --subreddit relationship_advice \
  --limit 200 \
  --sort new \
  --min-upvotes 1 \
  --min-comment-length 50 \
  --output sources/reddit/reddit_relationship_advice_new_200.md
```

**For "hot" sort** (trending posts):
```bash
python3 sources/scripts/collect_reddit_data_no_auth.py \
  --subreddit relationship_advice \
  --limit 200 \
  --sort hot \
  --min-upvotes 5 \
  --min-comment-length 75 \
  --output sources/reddit/reddit_relationship_advice_hot_200.md
```

**Why these settings:**
- `--sort new`: Posts are minutes/hours old, so `min_upvotes=1` is appropriate
- `--sort hot`: Posts have more upvotes but `min_upvotes=5` catches more posts
- Lower `min_comment_length` (50-75) captures shorter but valid advice comments

### Priority 2: Merge New Data

After collecting, merge with existing dataset:
```bash
python3 sources/scripts/merge_training_data.py \
  --existing sources/training_data_final_merged.md \
  --new sources/reddit/reddit_relationship_advice_new_200.md \
  --output sources/training_data_final_merged.md
```

The script will prompt to archive source files (recommended to keep directory clean).

### Priority 3: Collect from Multiple Subreddits

To reach 5-10MB target, consider collecting from:
- `r/relationship_advice` (current)
- `r/relationships` (already have some)
- `r/dating_advice` (related content)
- `r/AmItheAsshole` (filter for relationship posts)

### Priority 4: Start Fresh Training Run

Once dataset reaches 5-10MB:
```bash
TRAINING_DATA_SOURCE=sources/training_data_final_merged.md \
MODEL_TYPE=gpt2 \
USE_LORA=False \
TRAINING_STEPS=5000 \
ENABLE_CHECKPOINTS=True \
LEARNING_RATE=1e-5 \
python3 training.py
```

**Important**: Start a **FRESH** training run (don't resume from old checkpoint) since the dataset has changed significantly.

## Key Implementation Details

### Reddit Collection Script Features

1. **Sort Methods**:
   - `top`: Highest scoring (same results each run → many duplicates)
   - `hot`: Currently trending (changes frequently, good for new content)
   - `new`: Most recent (always different, best for avoiding duplicates)

2. **Pagination**:
   - Uses Reddit's `after` token for proper pagination
   - Continues until `after_token is None` (end of results)
   - Handles rate limiting with configurable delays

3. **Duplicate Detection**:
   - Loads existing post IDs from log files and output files
   - Checks both short IDs and full Reddit IDs
   - Skips duplicates automatically

4. **Logging**:
   - Automatic tee-style logging to `logs/reddit_collect_YYYYMMDD_HHMMSS.log`
   - Shows output in terminal AND saves to file
   - Use `--no-log-file` to disable file logging

### Merge Script Features

1. **Metadata Stripping**:
   - Automatically removes Reddit post headers (lines starting with `#` or `=`)
   - Extracts only Q&A pairs (QUESTION: ... ANSWER: ...)

2. **Archiving**:
   - Interactive prompt: "Archive source files after merge? [y/N]"
   - `--archive` flag: Automatically archive without prompt
   - `--no-archive` flag: Skip archiving
   - Moves files to `sources/reddit/archive/` with timestamp if duplicate exists

### Cleanup Script Features

1. **Deduplication**:
   - Analyzes all `.md` files in `sources/reddit/`
   - Extracts post IDs from URLs
   - Consolidates unique posts per subreddit

2. **Organization**:
   - Creates consolidated files: `reddit_{subreddit}_consolidated.md`
   - Archives old files to `archive/` directory
   - Keeps main log file (relationship_advice if available)

## Project Structure

```
build_llm_karpathy/
├── sources/
│   ├── training_data_final.md              # Original base dataset (1.0 MB)
│   ├── training_data_final_merged.md       # Current working dataset (4.3 MB) ⭐ USE THIS
│   ├── reddit/
│   │   ├── reddit_relationship_advice_consolidated.md  # Unique posts
│   │   ├── reddit_relationships_consolidated.md        # Unique posts
│   │   ├── reddit_relationship_advice_log.txt          # Main log file
│   │   └── archive/                        # Archived old files
│   └── scripts/
│       ├── collect_reddit_data_no_auth.py  # Reddit collection script
│       ├── merge_training_data.py          # Merge Reddit data with training data
│       ├── cleanup_reddit_data.py          # Clean up and deduplicate Reddit files
│       └── README_REDDIT_WORKFLOW.md       # Workflow documentation
├── training.py                              # Main training script
├── training_resume.py                       # Resume training from checkpoint
├── logs/                                    # Collection run logs
└── checkpoints/                             # Model checkpoints
```

## Known Issues

1. **Filter Settings Too Restrictive**:
   - `--min-upvotes 10` filters out most "new" posts (they have 0-5 upvotes)
   - `--min-comment-length 100` filters out shorter but valid comments
   - **Solution**: Lower filters based on sort method (see Next Steps)

2. **Duplicate Posts**:
   - Using `--sort top` always returns same posts → all duplicates
   - **Solution**: Use `--sort hot` or `--sort new` for different posts

3. **Data Exhaustion**:
   - After collecting from `r/relationship_advice` with current filters, running out of qualifying posts
   - **Solution**: Lower filters OR collect from additional subreddits

## Questions for Next Agent

1. **Filter Strategy**: Should we use different default filters based on sort method? (e.g., auto-adjust `min_upvotes` based on `--sort`)

2. **Data Quality vs Quantity**: Is it better to:
   - Lower filters significantly to get more data (may include lower quality posts)
   - Keep filters higher and collect from multiple subreddits
   - Use a combination approach

3. **Training Strategy**: Once we reach 5-10MB:
   - Start fresh training run immediately?
   - Continue collecting to 10MB+ for better results?
   - Test with current 4.3MB first?

## Resources

### Documentation
- `sources/scripts/README_REDDIT_WORKFLOW.md` - Complete workflow guide
- `.cursor/progress/2-handoff-gpt2-finetuning-20251119.md` - Previous handoff (training focus)

### Key Scripts
- **Collection**: `sources/scripts/collect_reddit_data_no_auth.py`
- **Merging**: `sources/scripts/merge_training_data.py`
- **Cleanup**: `sources/scripts/cleanup_reddit_data.py`

### Current Dataset
- **Working File**: `sources/training_data_final_merged.md` (4.3 MB, 1,486 Q&A pairs)
- **Base File**: `sources/training_data_final.md` (1.0 MB, 750 Q&A pairs) - DO NOT MODIFY

### Logs
- Collection logs: `logs/reddit_collect_*.log`
- Reddit metadata log: `sources/reddit/reddit_relationship_advice_log.txt`

## Workflow Summary

**To add more Reddit data:**
1. Collect → `sources/reddit/new_file.md` (use appropriate filters for sort method)
2. Merge → `sources/training_data_final_merged.md` (use merged file as `--existing`)
3. Archive → Source files moved to `sources/reddit/archive/` (optional but recommended)

**Example complete workflow:**
```bash
# 1. Collect with adjusted filters
python3 sources/scripts/collect_reddit_data_no_auth.py \
  --subreddit relationship_advice \
  --limit 200 \
  --sort hot \
  --min-upvotes 5 \
  --min-comment-length 75 \
  --output sources/reddit/reddit_relationship_advice_hot_200.md

# 2. Merge with existing dataset
python3 sources/scripts/merge_training_data.py \
  --existing sources/training_data_final_merged.md \
  --new sources/reddit/reddit_relationship_advice_hot_200.md \
  --output sources/training_data_final_merged.md
# (Will prompt to archive - say 'y' to keep directory clean)

# 3. Verify size
wc -c sources/training_data_final_merged.md
```

## Environment Variables for Training (When Ready)

```bash
TRAINING_DATA_SOURCE=sources/training_data_final_merged.md
MODEL_TYPE=gpt2
USE_LORA=False
TRAINING_STEPS=5000
ENABLE_CHECKPOINTS=True
LEARNING_RATE=1e-5
```

**Note**: Start a FRESH training run (not resuming) since the dataset has changed significantly.

