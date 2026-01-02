# Create A New Script To Run Multiple Prompts

## Working directory

`~/Desktop/build_llm_karpathy`

## Contents

Now can you create a script that runs inference against multiple prompts? The new script should run the script @build_llm_karpathy/qa/run_inference.py  against these prompts. More specifically, The user will supply a list of prompts in the CLI using the shorthands that we've already established in @build_llm_karpathy/qa/test_prompts.md . The user will also pass in the number of runs for each prompt. So for example: `python3 qa/run_prompts.py -p 'caretaking burnout in a family system' 'partners grief and emotional distance' 3`. This would instruct the script to run each of these prompts (the one about caregiving burnout and the one about partner's grief) 3 times each. The script will run the child script `qa/run_inference.py` with a given prompt for each of these runs. This will create the documentation for each run as before. So the `--save-output` flag will be passed in by the parent script to each call of @run_inference.py . The new script will also take an output directory argument, falling back to the default of @run_inference.py    if the user doesn't pass an output directory in. The model_type for the new script will be `openai_backend` (this can be changed later if desired.)

This new script should also create a second document, `outputs/inference/results-{timestamp}.md`. This new results file will be a table of all the prompts with the resulting score and a timestamp of when the prompt was run. Below this table, there should be the average for each prompt.It should simply be the average of all the runs of a specific prompt.

The purpose of this script is to help better benchmark different models that I've created via fine-tune.

<!-- DONE -->
