# Training Data Files - Explanation

## Current Files

1. **`training_data_final.md`** (1.0 MB, 750 Q&A pairs)
   - Original training data
   - Contains Carolyn Hax chat transcripts and original Reddit data
   - **This is the base dataset**

2. **`training_data_final_merged.md`** (1.0 MB, 750 Q&A pairs)
   - **CURRENTLY IDENTICAL to `training_data_final.md`**
   - The merge script was run but didn't successfully add Reddit data
   - The consolidated Reddit files are empty (bug in cleanup script)
   - **DO NOT USE THIS FILE** - it's a duplicate

## What Happened

The `cleanup_reddit_data.py` script created consolidated Reddit files, but they're empty (only metadata, no Q&A content). The merge script then tried to merge these empty files, resulting in a file identical to the original.

## Solution

To properly merge Reddit data, you need to:

1. **Use the archived Reddit files** (they have the actual content):
   ```bash
   python3 sources/scripts/merge_training_data.py \
     --existing sources/training_data_final.md \
     --new sources/reddit/archive/reddit_relationship_advice_month_400.md \
            sources/reddit/archive/reddit_relationship_advice_year_400.md \
            sources/reddit/archive/reddit_relationships_*.md \
     --output sources/training_data_final_merged.md
   ```

2. **OR fix the consolidated files** by restoring from archive and re-running cleanup

## Recommendation

- **Use `training_data_final.md`** for training (it's the complete, working dataset)
- **Delete `training_data_final_merged.md`** (it's a duplicate)
- Re-run the merge with the archived files to create a proper merged dataset

