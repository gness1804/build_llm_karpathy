# Handoff: GPT-2 Fine-Tuning for Carolyn Hax-Style Advice Generation

**Date**: November 18, 2025  
**Status**: Data Collection & Preprocessing Complete, Training in Progress  
**Working Directory**: `~/Desktop/build_llm_karpathy`

## Project Overview

This project fine-tunes GPT-2 to generate advice in the style of Carolyn Hax, a Washington Post advice columnist. The goal is to create a language model that can provide coherent, empathetic relationship advice similar to Carolyn Hax's writing style.

**Key Objectives:**
- Fine-tune GPT-2 (using LoRA for efficiency) on Q&A format data
- Generate coherent advice responses to relationship questions
- Maintain Carolyn Hax's empathetic, thoughtful tone

**Current Approach:**
- Using GPT-2 base model with LoRA (Low-Rank Adaptation) for efficient fine-tuning
- Training on formatted Q&A pairs (QUESTION: / ANSWER: format)
- Collecting data from both Carolyn Hax chats and Reddit r/relationships

## Current State

### ✅ Completed

1. **Data Collection Infrastructure**
   - **Carolyn Hax Data Collection**: Manual collection scripts and guides
     - `sources/scripts/create_carolyn_hax_chat_files.py` - Helper script for creating chat files
     - `sources/docs/COLLECTING_MORE_CAROLYN_HAX_DATA.md` - Collection guide
     - `sources/docs/QUICK_START_COLLECTING.md` - Quick reference
   - **Reddit Data Collection**: Automated collection without API credentials
     - `sources/scripts/collect_reddit_data_no_auth.py` - Public JSON endpoint scraper
     - `sources/scripts/collect_reddit_data.py` - Authenticated PRAW version (requires API setup)
     - Both scripts include duplicate detection and logging
     - Rate limiting with exponential backoff for 429 errors

2. **Data Preprocessing Scripts**
   - `sources/scripts/add_qa_labels.py` - Adds "QUESTION:" and "ANSWER:" labels to chat transcripts (idempotent)
   - `sources/scripts/merge_carolyn_hax_chats.py` - Merges multiple chat files into one document
   - `sources/scripts/clean_carolyn_hax_data.py` - Removes metadata (timestamps, "Likes", "Press Enter", etc.)
   - `sources/scripts/deduplicate_reddit_data.py` - Removes duplicate posts from Reddit data files

3. **Training Infrastructure**
   - `training.py` - Main training script with:
     - GPT-2 fine-tuning support (LoRA and full fine-tuning)
     - Learning rate scheduler (warmup + cosine decay)
     - Gradient clipping
     - Checkpoint saving
     - Output to file functionality
     - Configurable via environment variables
   - `training_resume.py` - Resume training from checkpoint
   - `load_checkpoint.py` - Load model for inference with configurable temperature/top_k

4. **Hyperparameter Optimization**
   - Learning rate: `1e-5` (reduced from `3e-4` for fine-tuning)
   - Block size: `128` (reduced from `512` for speed, balanced with context)
   - Batch size: `16` (optimized for speed/quality balance)
   - LoRA rank: `16`, alpha: `32.0` (for GPT-2 fine-tuning)
   - Learning rate scheduler: 10% warmup, 90% cosine decay
   - Gradient clipping: `max_norm=1.0`
   - Generation parameters: `temperature=0.7`, `top_k=50`

5. **Data Files**
   - **Carolyn Hax**: 
     - `sources/carolyn_hax/carolyn_hax_chats/` - 23 individual chat files
     - `sources/carolyn_hax/carolyn_hax_merged_cleaned.md` - Merged and cleaned dataset
   - **Reddit**:
     - `sources/reddit/reddit_relationships_20251118_deduplicated.md` - 25 unique posts (deduplicated from 150)
     - `sources/archive/` - Archived original files with duplicates

6. **Documentation**
   - `FIXING_GIBBERISH_OUTPUT.md` - Analysis of output quality issues and solutions
   - `docs/PRETRAINED_VS_YOUR_MODEL.md` - Comparison of training approaches
   - `sources/docs/REDDIT_DATA_GUIDE.md` - Guide for using Reddit data
   - `sources/docs/REDDIT_SETUP.md` - Reddit API setup instructions

### 🔧 In Progress

1. **Data Quality Improvement**
   - Successfully deduplicated Reddit data (150 → 25 unique posts)
   - Need to collect more unique Reddit posts to expand dataset
   - Current dataset size: ~749KB (Carolyn Hax) + ~88KB (Reddit) = ~837KB total

2. **Training Optimization**
   - Loss reduced from 9.6 to 7.2 after data cleaning and hyperparameter adjustments
   - Target loss: 2-3 for coherent GPT-2 output
   - Still experiencing some gibberish output, likely due to:
     - Insufficient training data
     - Loss still too high (needs to drop further)
     - May need more training steps or lower learning rate

## Next Steps

### Immediate Priority

1. **Expand Dataset**
   - Collect more Reddit r/relationships posts using `collect_reddit_data_no_auth.py`
   - The script now has duplicate detection, so new runs will only collect unique posts
   - Target: 100+ unique posts for better model performance
   - Command:
     ```bash
     python3 sources/scripts/collect_reddit_data_no_auth.py \
       --subreddit relationships \
       --limit 100 \
       --output sources/reddit/reddit_relationships_$(date +%Y%m%d).md
     ```

2. **Merge and Clean New Data**
   - After collecting more Reddit data, merge with existing Carolyn Hax data
   - Ensure all data is properly formatted with QUESTION:/ANSWER: labels
   - Clean any remaining metadata

3. **Continue Training**
   - Retrain with expanded dataset
   - Monitor loss - should continue decreasing
   - If loss plateaus, consider:
     - Lower learning rate (5e-6)
     - Full fine-tuning instead of LoRA
     - More training steps (20,000+)

### Future Enhancements

1. **Model Evaluation**
   - Create evaluation metrics beyond loss
   - Test on held-out questions
   - Compare output quality to original Carolyn Hax responses

2. **Data Quality**
   - Filter Reddit posts for quality (upvotes, engagement)
   - Prioritize longer, more detailed responses
   - Consider adding more Carolyn Hax data if available

3. **Inference Improvements**
   - Experiment with different temperature/top_k values
   - Add prompt engineering for better context
   - Implement response length control

## Key Implementation Details

### Training Configuration

**Environment Variables:**
```bash
MODEL_TYPE=gpt2                    # Use GPT-2 instead of training from scratch
USE_LORA=True                      # Use LoRA for efficient fine-tuning
LORA_RANK=16                       # LoRA rank (higher = more parameters)
LORA_ALPHA=32.0                    # LoRA alpha (scaling factor)
TRAINING_STEPS=10000               # Number of training steps
ENABLE_CHECKPOINTS=True            # Save checkpoints during training
CHECKPOINT_INTERVAL=500            # Save every N steps
TRAINING_DATA_SOURCE=sources/carolyn_hax/carolyn_hax_merged_cleaned.md
```

**Training Command:**
```bash
TRAINING_DATA_SOURCE=sources/carolyn_hax/carolyn_hax_merged_cleaned.md \
MODEL_TYPE=gpt2 \
USE_LORA=True \
LORA_RANK=16 \
LORA_ALPHA=32.0 \
TRAINING_STEPS=10000 \
ENABLE_CHECKPOINTS=True \
python3 training.py
```

### Data Format

All training data should follow this format:
```
QUESTION: [The question text here]

ANSWER: [The answer text here]

QUESTION: [Next question...]

ANSWER: [Next answer...]
```

### File Structure

```
build_llm_karpathy/
├── training.py                    # Main training script
├── training_resume.py             # Resume from checkpoint
├── load_checkpoint.py             # Load for inference
├── models/
│   ├── gpt2_wrapper.py           # GPT-2 model wrapper
│   └── bigram_lm_v2_lora.py      # LoRA implementation
├── sources/
│   ├── carolyn_hax/
│   │   ├── carolyn_hax_chats/    # Individual chat files
│   │   └── carolyn_hax_merged_cleaned.md
│   ├── reddit/
│   │   └── reddit_relationships_20251118_deduplicated.md
│   └── scripts/
│       ├── add_qa_labels.py
│       ├── merge_carolyn_hax_chats.py
│       ├── clean_carolyn_hax_data.py
│       ├── collect_reddit_data_no_auth.py
│       ├── collect_reddit_data.py
│       └── deduplicate_reddit_data.py
└── outputs/                       # Training output logs
```

### Important Scripts

1. **`collect_reddit_data_no_auth.py`**
   - Collects Reddit posts without API credentials
   - Automatically detects and skips duplicates
   - Logs collected posts to prevent re-collection
   - Rate limiting with exponential backoff
   - Usage:
     ```bash
     python3 sources/scripts/collect_reddit_data_no_auth.py \
       --subreddit relationships \
       --limit 50 \
       --delay 3 \
       --batch-delay 10
     ```

2. **`deduplicate_reddit_data.py`**
   - Removes duplicate posts from Reddit data files
   - Preserves original format with headers
   - Renumbers posts sequentially
   - Usage:
     ```bash
     python3 sources/scripts/deduplicate_reddit_data.py \
       sources/reddit/reddit_relationships_20251118.md \
       --output sources/reddit/reddit_relationships_20251118_deduplicated.md
     ```

3. **`clean_carolyn_hax_data.py`**
   - Removes metadata from Carolyn Hax chats
   - Removes timestamps, "Likes", "Press Enter", etc.
   - Keeps only QUESTION:/ANSWER: content
   - Usage:
     ```bash
     python3 sources/scripts/clean_carolyn_hax_data.py \
       sources/carolyn_hax/carolyn_hax_merged.md \
       -o sources/carolyn_hax/carolyn_hax_merged_cleaned.md
     ```

## Known Issues

1. **High Loss Values**
   - Current loss: ~7.2 (target: 2-3)
   - Likely causes:
     - Insufficient training data (only ~837KB)
     - May need more training steps
     - Learning rate might need further adjustment

2. **Gibberish Output**
   - Model sometimes generates incoherent text
   - Related to high loss values
   - Partially addressed by:
     - Data cleaning (removed metadata)
     - Lower temperature (0.7)
     - Better hyperparameters
   - Still needs improvement through more data and training

3. **Dataset Size**
   - Current dataset is relatively small (~837KB)
   - GPT-2 fine-tuning typically benefits from larger datasets
   - Solution: Collect more Reddit data

4. **Duplicate Detection**
   - Reddit collection script now has duplicate detection
   - Existing file had 150 posts but only 25 unique (deduplicated)
   - Future runs will automatically skip duplicates

## Questions for Next Agent

1. **Data Collection Strategy**
   - Should we focus on collecting more Reddit data or more Carolyn Hax data?
   - What's the target dataset size for good GPT-2 fine-tuning?
   - Should we filter Reddit posts by quality (upvotes, length)?

2. **Training Approach**
   - Should we try full fine-tuning instead of LoRA for better quality?
   - What's the optimal learning rate for this dataset size?
   - How many training steps are needed to reach loss < 3?

3. **Model Evaluation**
   - How should we evaluate output quality beyond loss?
   - Should we create a test set of questions to evaluate on?
   - What metrics matter most for advice generation?

4. **Data Format**
   - Should we include metadata like post scores/dates in training?
   - Should we format Reddit data differently than Carolyn Hax data?
   - Should we add context about the question (e.g., relationship type)?

## Resources

### Key Files
- `training.py` - Main training script
- `FIXING_GIBBERISH_OUTPUT.md` - Detailed analysis of output issues
- `sources/carolyn_hax/carolyn_hax_merged_cleaned.md` - Current training data
- `sources/reddit/reddit_relationships_20251118_deduplicated.md` - Reddit dataset

### Documentation
- `docs/PRETRAINED_VS_YOUR_MODEL.md` - Training approach comparison
- `sources/docs/REDDIT_DATA_GUIDE.md` - Reddit data usage guide
- `sources/docs/COLLECTING_MORE_CAROLYN_HAX_DATA.md` - Carolyn Hax collection guide

### Scripts
- `sources/scripts/collect_reddit_data_no_auth.py` - Reddit collection (no auth)
- `sources/scripts/deduplicate_reddit_data.py` - Deduplication tool
- `sources/scripts/clean_carolyn_hax_data.py` - Data cleaning

### Checkpoints
- Checkpoints are saved in `checkpoints/` directory
- Use `training_resume.py` to continue from a checkpoint
- Use `load_checkpoint.py` for inference

### Output Logs
- Training outputs are saved in `outputs/` directory
- Filenames include model type, data source, steps, and timestamp
- Example: `build_llm_output_gpt2_carolyn_hax_merged_cleaned_50257_10000_test=false_gpt2_gpt2_lora_r16_a32.0_OUTPUT_11142025_140620.txt`

## Technical Notes

### LoRA Configuration
- **Rank (r)**: 16 - Controls the number of trainable parameters
- **Alpha (α)**: 32.0 - Scaling factor for LoRA weights
- **Dropout**: 0.0 - Currently no dropout in LoRA layers
- Higher rank/alpha = more parameters, potentially better quality but slower training

### Learning Rate Schedule
- **Warmup**: 10% of training steps (linear increase)
- **Decay**: 90% of training steps (cosine annealing)
- Helps prevent catastrophic forgetting during fine-tuning

### Generation Parameters
- **Temperature**: 0.7 (lower = more focused, less random)
- **Top-k**: 50 (only consider top 50 tokens for sampling)
- Adjust via environment variables: `TEMPERATURE=0.6 TOP_K=40 python3 load_checkpoint.py`

### Device Support
- Supports CPU, CUDA, and MPS (Apple Silicon)
- Automatically detects and uses best available device
- MPS prioritized on Apple Silicon for better performance

## Recent Changes

1. **November 18, 2025**: Created deduplication script and cleaned Reddit data (150 → 25 unique posts)
2. **November 14, 2025**: Implemented data cleaning, improved hyperparameters, added learning rate scheduler
3. **November 13, 2025**: Added Reddit data collection scripts with duplicate detection
4. **November 10-12, 2025**: Initial GPT-2 fine-tuning setup, LoRA integration, checkpoint system

## Next Session Checklist

- [ ] Collect more Reddit data (target: 100+ unique posts)
- [ ] Merge new Reddit data with existing dataset
- [ ] Verify data format (QUESTION:/ANSWER: labels)
- [ ] Retrain model with expanded dataset
- [ ] Monitor loss - should continue decreasing
- [ ] Evaluate output quality on test questions
- [ ] Adjust hyperparameters if loss plateaus
- [ ] Consider full fine-tuning if LoRA results are insufficient

---

**To pick up this handoff:**
```bash
cd ~/Desktop/build_llm_karpathy
cfs instructions handoff pickup
```

