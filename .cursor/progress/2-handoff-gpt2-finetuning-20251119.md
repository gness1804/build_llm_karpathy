# Handoff: GPT-2 Fine-Tuning for Carolyn Hax-Style Advice Generation

**Date**: November 19, 2025
**Status**: Training Plateaued (Loss ~5.0), Need More Data
**Working Directory**: `~/Desktop/build_llm_karpathy`

## Project Overview

This project fine-tunes GPT-2 to generate advice in the style of Carolyn Hax. We are currently using a **Full Fine-Tuning** approach (LoRA was tested but underperformed).

**Key Objectives:**
- Fine-tune GPT-2 on Q&A format data
- Generate coherent advice responses to relationship questions
- Achieve training loss < 3.0 (currently stuck at ~5.0)

## Current State

### ✅ Completed
1.  **Data Collection**:
    - Collected ~90 posts from `r/relationships`.
    - Merged with ~23 Carolyn Hax chat transcripts.
    - Total dataset size: ~837KB (Too small for high-quality generation).
    - **Cleaned Data**: Removed metadata headers from training files.
2.  **Training Infrastructure**:
    - `training.py`: Supports full fine-tuning and LoRA.
    - `training_resume.py`: Validated and fixed for resuming training runs.
    - **Fixed**: Indentation and variable scope issues in `training_resume.py`.
3.  **Experiments**:
    - **LoRA Run (10k steps)**: Loss ~7.3 (High, gibberish output).
    - **Full Fine-Tuning Run (2k steps)**: Loss dropped to ~5.2 (Better, coherent words but broken grammar).
    - **Resume Run (+3k steps)**: Loss plateaued at ~5.08.
    - **High LR Test (+1k steps)**: Loss stayed flat at ~5.00 (Confirmed data saturation).

### 🚧 In Progress / Blockers
1.  **Data Saturation**: The current dataset (~1MB) is too small. The model has learned everything it can from it without overfitting, but it's not enough to learn proper grammar and style generalization.
2.  **Loss Plateau**: Training loss is stuck around 5.0. We need it below 3.0 for good results.

## Next Steps (Immediate)

**CRITICAL PRIORITY: Expand Dataset to 5-10MB**

1.  **Collect Massive Data**:
    - Run the Reddit collector on `r/relationship_advice` (larger subreddit).
    - Target: 300-500 new posts.
    - Command:
      ```bash
      python3 build_llm_karpathy/sources/scripts/collect_reddit_data_no_auth.py \
        --subreddit relationship_advice \
        --limit 300 \
        --time-filter month \
        --min-upvotes 50 \
        --output sources/reddit/reddit_relationship_advice_month_300.md
      ```

2.  **Merge & Retrain**:
    - Concatenate the new data with `sources/training_data_final.md`.
    - Start a **FRESH** full fine-tuning run (don't resume) on the larger dataset.
    - Expect loss to drop significantly lower with more data.

## Key Files

- **Training Script**: `build_llm_karpathy/training.py`
- **Resume Script**: `build_llm_karpathy/training_resume.py`
- **Current Dataset**: `build_llm_karpathy/sources/training_data_final.md` (DO NOT use `training_data_combined.md` - it was corrupted).
- **Collector Script**: `build_llm_karpathy/sources/scripts/collect_reddit_data_no_auth.py`
- **Latest Checkpoint**: `checkpoints/checkpoint_resumed_step006000_11192025_192528.pt`

## Environment Variables for Training
```bash
TRAINING_DATA_SOURCE=sources/training_data_final.md
MODEL_TYPE=gpt2
USE_LORA=False
TRAINING_STEPS=5000
ENABLE_CHECKPOINTS=True
LEARNING_RATE=1e-5
```

## Context for Next Agent
- We switched from LoRA to Full Fine-Tuning because LoRA loss was too high (~7.3).
- Full FT got us to ~5.0 but stalled.
- We confirmed the plateau isn't just learning rate by testing a higher LR (5e-5), which didn't help.
- **Conclusion**: We need significantly more data. Focus on data collection first.

