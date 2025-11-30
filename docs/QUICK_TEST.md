# Quick Test Commands

## Minimal GPT-2 Test (Fastest)

Test GPT-2 integration with minimal training:

```bash
MODEL_TYPE=gpt2 USE_LORA=True TRAINING_STEPS=10 TEST_MODE=True python3 training.py
```

**What this does:**
- Loads GPT-2 Small (124M params)
- Applies LoRA adapters
- Trains for only 10 steps (just to verify it works)
- Uses test mode (smaller batch size)

**Expected time:** 1-2 minutes (mostly model loading/download)

## Even Shorter Test

If you just want to verify the model loads and tokenizes correctly:

```bash
MODEL_TYPE=gpt2 USE_LORA=True TRAINING_STEPS=5 TEST_MODE=True python3 training.py
```

**Expected time:** < 1 minute

## Notes

1. **TOKENIZATION_METHOD is ignored** when `MODEL_TYPE=gpt2` - GPT-2 uses its own tokenizer automatically
2. **First run downloads GPT-2** (~500MB) - this takes time but is cached
3. **GPT-2 is slower** than from-scratch models even with LoRA (it's 124M params vs your ~780K params)

## Full Test Command (What You Ran)

```bash
MODEL_TYPE=gpt2 USE_LORA=True GPT2_MODEL_NAME=gpt2 TEST_MODE=True python3 training.py
```

This runs 1000 training steps (TEST_MODE default), which can take 10-30 minutes depending on your hardware.

## Recommended Quick Test

```bash
MODEL_TYPE=gpt2 USE_LORA=True TRAINING_STEPS=50 TEST_MODE=True python3 training.py
```

This gives you:
- Enough steps to see training progress
- Quick verification (2-5 minutes)
- Can see if loss is decreasing

