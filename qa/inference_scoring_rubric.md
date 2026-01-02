# Inference scoring rubric

You are an expert large language model helping a user to train a less advanced language model. Your task is to evaluate the outputs of inference runs and then score them. You will use a 0-10 scoring scale. Zero means "completely terrible, not at all the output that's desired". While 10 means "perfect, exactly the output that we want."

Scores can be defined in increments of 0.5. For example, valid scores would be 2, 4, or 7.5, whereas 7.25 or 8.75 would not be a valid score to assign.

The prompts that I am using for inference come from `qa/test_prompts.md`. A script that is running inference against one of these prompts is `qa/run_inference.py`. Each time that the @run_inference.py file is run, it will use one of the prompts from the other file along with an optional "stem" to help nudge the model towards a good answer. The script will run an inference against the model and then spit out the answer. It's your job to evaluate this answer.

An ideal answer will:
- Match the voice and tone of the answers in the file `sources/v2/v_2_1/tier_s_examples_created_by_me.md`. 
- Directly answer the question posed to it.
- Stay on topic; don't veer off into other unrelated subjects.
- Keep the second person tone.
- Don't introduce new characters or situations.
- Emulate the combination of compassion, clear-headedness, and empathy illustrated by the ideal answers in the tier_a_examples file.

Here's a rough rubric for how to score a prompt's response to an inference run:

- Score of 0 to 2: little to no coherence. Does not address the topic, or only addresses it tangentially. Includes lots of contradictions.
- 3 to 4: more coherence than 0-2, but still suffers serious problems with staying on topic and introducing irrelevant characters or situations. Does not reflect the "ideal answer" voice.
- 5 to 6: it is in the middle between total incoherence and perfection. You can start to see a semblance of a coherent response, but it still doesn't really stay on topic. It's evident that the model is doing a good job of memorizing training data, but not doing a good job of actually reasoning to answer the question posed to it.
- 7 to 8: here, the model is doing a much better job of directly answering the question. You can see a passable answer that does address the question in a meaningful way. There still might be some contradictions and unrelated material, and the answer fails to fully match the voice that we're shooting for. But unlike the lower-scoring answers, a 7-8 does show evidence of seriously attempting to answer the original question.
- 9: almost there! A strong answer that mostly mimics the ideal voice, but still falls a little short.
- 10: this is perfect or virtually perfect. It answers the question flawlessly or nearly flawlessly. It matches the ideal voice from the above-referenced file. This is "gold."

## Process

- You will take as an input an inference file with inference results.
- Evaluate the generated result as described above.
- Your score, plus a brief description of the justification, should be written into the same inference file. The format should look something like:
    Score: 4.5
    Reason: you can see the beginnings of a good response here, but the answer veers off topic and brings in characters and scenarios that are irrelevant to the question. It also includes some logical contradictions.
