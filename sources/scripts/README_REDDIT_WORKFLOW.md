# Reddit Data Collection & Merging Workflow

## Overview

There are two separate processes:
1. **Collection**: Gather Reddit posts → creates files in `sources/reddit/`
2. **Merging**: Add Reddit data to training file → updates `training_data_final_merged.md`

The cleanup script is optional and only for organizing the `sources/reddit/` directory.

## Workflow for Adding More Reddit Posts

### Step 1: Collect New Reddit Data

Run the collection script to gather new posts:

```bash
python3 sources/scripts/collect_reddit_data_no_auth.py \
  --subreddit relationship_advice \
  --limit 400 \
  --time-filter year \
  --min-upvotes 50 \
  --output sources/reddit/reddit_relationship_advice_year_400_new.md \
  --delay 3.5 \
  --batch-delay 12.0
```

This creates a new file in `sources/reddit/` (NOT in archive).

### Step 2: Merge with Training Data

Merge the new Reddit file(s) with your existing merged training data:

```bash
python3 sources/scripts/merge_training_data.py \
  --existing sources/training_data_final_merged.md \
  --new sources/reddit/reddit_relationship_advice_year_400_new.md \
  --output sources/training_data_final_merged.md
```

**Important**: Use `training_data_final_merged.md` as the `--existing` file (not `training_data_final.md`), so you're adding to your already-merged dataset.

### Step 3 (Optional): Clean Up Reddit Directory

Periodically, you can clean up the `sources/reddit/` directory:

```bash
python3 sources/scripts/cleanup_reddit_data.py --reddit-dir reddit
```

This will:
- Consolidate duplicate posts across files
- Move old files to `archive/`
- Keep only unique consolidated files

## Quick Reference

**To add more data:**
1. Collect → `sources/reddit/new_file.md`
2. Merge → `sources/training_data_final_merged.md` (use merged file as existing)

**Files:**
- `sources/training_data_final.md` - Original base dataset (don't modify)
- `sources/training_data_final_merged.md` - Your working training file (this is what you train on)
- `sources/reddit/*.md` - New Reddit data files
- `sources/reddit/archive/*.md` - Archived/old Reddit files

## Example: Adding More Data

```bash
# 1. Collect new posts
python3 sources/scripts/collect_reddit_data_no_auth.py \
  --subreddit relationship_advice \
  --limit 500 \
  --time-filter all \
  --min-upvotes 30 \
  --output sources/reddit/reddit_relationship_advice_all_500.md

# 2. Merge with existing merged training data
python3 sources/scripts/merge_training_data.py \
  --existing sources/training_data_final_merged.md \
  --new sources/reddit/reddit_relationship_advice_all_500.md \
  --output sources/training_data_final_merged.md

# 3. (Optional) Clean up reddit directory
python3 sources/scripts/cleanup_reddit_data.py --reddit-dir reddit
```

