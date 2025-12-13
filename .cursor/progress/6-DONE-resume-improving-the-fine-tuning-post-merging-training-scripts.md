# Resume Improving The Fine Tuning Post Merging Training Scripts

## Working directory

`~/Desktop/build_llm_karpathy`

## Contents

I have been working on training and fine-tuning a large language model that uses information about relationships taken from advice columns and Reddit. After lots of trial and error, I was finally able to get some decent results. But as you will see from the log files that I mention below, the model still has a ways to go in training.

I'm getting reasonable results now after 2100 steps, but I want to further improve. Some of the problems include repetition and some nonsensical statements. I'm not getting gibberish, but the output isn't that smart either. I want to work with an agent to optimize this further. 

Please review the inference outputs in the `outputs/inference` directory. Note that it's coherent English, but it suffers from repetition and logical contradictions. 

For reference, my last successful training run of the training scripts was: 

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

My most recent checkpoint is `checkpoints/checkpoint_gpt2_training_data_final_merged_step002100_11252025_194715.pt`.

Again, I went through a lot of trial and error with other agents dealing with problems like gibberish output. I had to fix some critical bugs and tweak some hyperparameters, but now things are working much better. But now I want to optimize further to improve the output from what we see in these output logs. Can you help me make some optimizations so that we can continue to train our model to output strong, intelligent text?

<!-- DONE -->
