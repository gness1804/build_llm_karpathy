# Training runs.

## Training run that led to a good result. 
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

## Training script to run to test the new @training.py. Note the smaller number of training steps.

TRAINING_DATA_SOURCE=sources/training_data_final_merged.md \
MODEL_TYPE=gpt2 \
USE_LORA=False \
TRAINING_STEPS=200 \
LEARNING_RATE=5e-6 \
BLOCK_SIZE=128 \
BATCH_SIZE=8 \
ENABLE_CHECKPOINTS=True \
USE_LR_WARMUP=False \
ENABLE_OUTPUT_TO_FILE=True \
python3 training.py