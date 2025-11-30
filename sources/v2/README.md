"""
Version 2 training data: curated and synthetic Q&A set for advice-model polishing.

This directory is meant to hold:
- A small, hand-written "gold" set of QUESTION/ANSWER pairs (20–50 examples)
- Additional high-quality synthetic Q&A pairs generated with a stronger LLM
- A single merged file you can point TRAINING_DATA_SOURCE at (for v2 runs)

Core goals for v2:
- Remove Reddit/platform meta, URLs, updates, and comments
- Enforce a clean, consistent schema:
  - QUESTION: one paragraph (or a few) describing the situation
  - ANSWER: one coherent, self-contained answer in the tone you want
- Emphasize clarity, empathy, and concrete, ethical advice
"""

# Suggested file layout

- `training_data_v2_seed.md` — hand-written gold examples (you extend this)
- `training_data_v2_full.md` — final merged dataset (gold + edited synthetic)

You can keep everything in a single file if you prefer, but this structure
makes it easy to see your original gold set vs the expanded dataset.

## Format specification

Each example should follow this exact pattern, separated by at least one blank line:

```text
QUESTION: <one or more paragraphs, in plain English. No meta, no URLs.>

ANSWER: <one or more paragraphs of advice. No meta, no URLs.>
```

Guidelines:
- Do NOT include:
  - Reddit usernames, upvote/score counts, or "EDIT/UPDATE" sections
  - URLs, tracking parameters, or any platform-specific boilerplate
  - Comments from other users — only the main answer
- Do:
  - Keep the answer grounded, kind, and practical
  - Resolve obvious contradictions in the question instead of mirroring them
  - Use complete sentences and paragraphs (no bullet lists unless you want them)

## Using ChatGPT (or another LLM) to expand the dataset

1. First, write 20–50 **gold** examples in `training_data_v2_seed.md` by hand.
2. Then, use a prompt like this with ChatGPT, providing several of your gold examples:

```text
You are helping me build a small dataset of relationship-advice Q&A pairs.

Follow this exact schema, repeated for multiple examples:

QUESTION: <question text>

ANSWER: <answer text>

Requirements:
- Stay in the domain of interpersonal relationships: romantic partners, family, friends, roommates, coworkers.
- In generating your responses, vary the topics from question to question. We want to avoid overfitting the model on one particular topic or type of question.
- Do NOT mention Reddit, posts, threads, upvotes, or any platform.
- Do NOT include URLs, usernames, or emojis. (The exception to the URLs rule is if the URL is directly part of a question or an answer. An example of this might be a URL to a help website if the question concerns how to get help for a mental illness.)
- The answer should be:
  - Empathetic but direct
  - Ethically sound (no revenge, harassment, or bigotry)
  - 1–3 short paragraphs, no more than ~250–300 words.
- Do NOT include any explanations to me or any other metadata or meta-analysis; only output raw QUESTION/ANSWER pairs.
- Use the following custom token between Q&A pairs: `<END_OF_SET>`. Always use this token after every Q&A set. Never use this token inside any question or answer. Just use it after a Q&A set concludes. This should help the model that trains on this data to know exactly how to separate Q and A pairs.

Here are some examples of the style and format I want:

<paste a few of your best gold QUESTION/ANSWER pairs here>

Now generate 10 new QUESTION/ANSWER pairs in exactly this format.
```

3. Paste the model’s output into a scratch file, then **edit ruthlessly**:
   - Remove any meta language the model sneaks in (e.g. "As an AI", "On Reddit…")
   - Fix tone, ethics, and clarity
   - Make sure the formatting still matches the schema exactly
4. Append the cleaned pairs into `training_data_v2_full.md`.

When you’re ready to train v2, you can point `TRAINING_DATA_SOURCE` at:

```bash
TRAINING_DATA_SOURCE=sources/v2/training_data_v2_full.md
```


