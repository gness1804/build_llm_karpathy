# Write Script To Run Inference Prompts

## Working directory

`~/Desktop/build_llm_karpathy`

## Contents

I have a group of easy, medium, and hard inference prompts to train my interpersonal relationships LLM at qa/test_prompts.md. I would like you to write a Python script in the qa/ directory (already exists). This script should enable the user to select one of these prompts to run inference against.

The full command that I have been using to run inference is:

```sh
CHECKPOINT_PATH=path/to/checkpoint.pt \
MODE=inference \
TOP_K=40 \
TOP_P=0.9 \
REPETITION_PENALTY=1.2 \
NO_REPEAT_NGRAM_SIZE=3 \
MAX_NEW_TOKENS=300 \
SAVE_OUTPUT=True \
TEMPERATURE=1.0 \
PROMPT="QUESTION: <prompt>\n\nANSWER:" \
python3 scripts/load_checkpoint.py
```

I have a .env file saved in the qa/ directory that contains all of these values EXCEPT for CHECKPOINT_PATH and PROMPT. I want the Python script to take in these latter two items as inputs but to delegate to the .env file the setting of all the other env vars for each inference run. 

The script should:
- Allow the user to select a prompt to run from the list in the test_prompts.md file noted above. 
- Selection of a prompt should use a logical shorthand. (Example: `qa/run_inference.py --prompt 'parent overstepping' --checkpoint 'path/to/checkpoint.pt' )
- As noted above, the other flag to be passed in should be the path to the checkpoint file to run inference against. 
- The script should run the inference process as normal, including outputting the result in the appropriate place.

## Some more examples

- `qa/run_inference.py --prompt 'simple romantic miscommunication' --checkpoint 'path/to/checkpoint1.pt'
- `qa/run_inference.py --prompt 'value difference around ambition' --checkpoint 'path/to/checkpoint2.pt'


