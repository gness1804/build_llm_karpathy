# Resume Improving The Fine Tuning Post Merging Training Scripts

## Working directory

`~/Desktop/build_llm_karpathy`

## Contents

Now that the training scripts have been merged successfully, it's time to return to fine-tuning. I'm getting reasonable results now after 2100 steps, but I want to further improve. Some of the problems include repetition and some nonsensical statements. I'm not getting gibberish, but the output isn't that smart either. I want to work with an agent to optimize this further. 

I should show the agent the last few inference outputs which are now available in outputs/inference. This should allow the agent to have a sense of what the output is looking like. 

For reference, my last successful training run pre-merge of the training scripts was: 

```bash
TRAINING_DATA_SOURCE=sources/training_data_final_merged.md \
MODEL_TYPE=gpt2 \
USE_LORA=False \
TRAINING_STEPS=2000 \
LEARNING_RATE=5e-6 \
BLOCK_SIZE=128 \
BATCH_SIZE=8 \
ENABLE_CHECKPOINTS=True \
USE_LR_WARMUP=False \
ENABLE_OUTPUT_TO_FILE=True \
python3 training.py
```

I need to work with the agent to figure out what to do next. Maybe changing the learning rate, for instance. 
