# Normalize The Dataset Source Document

## Working directory

`~/Desktop/build_llm_karpathy`

## Contents

The document `sources/v3/dataset_source.md` contains data that I'm going to use for my next round of fine-tuning an LLM. My LLM will score advice column responses and then give the strengths and weaknesses and then rewrite the response to be better according to criteria that I specify. This file needs to contain highly standardized data in order for the fine-tuning to be optimal.To this end, I want all the examples in the file to mimic the following structure:

```markdown
QUESTION: 
[the advice column question]

DRAFT_RESPONSE: 
[the draft response for the LLM being fine-tuned to evaluate]

SCORE: 8.0 [a score on the scale of 1 to 10, only in increments of 0.5. So 1.0, 1.5, 2.0...9.5, 10.0]

STRENGTHS:
- [First strength in a bullet point.]
- [Second strength in a bullet point.]
- [etc...]

WEAKNESSES:
- [First weakness in a bullet point.]
- [Second weakness in a bullet point.]
- [etc...]

REVISED_RESPONSE: 
[The fine-tuned LLM's response.]
<END_OF_SET>

QUESTION:
[Next question...]
```

This is the canonical template. Can you revise this document to exactly match this format? It will involve things such as:
- Changing header names. (Example - In most of my examples, the header `REVISED_RESPONSE:` its currently `OUTPUT:`. This needs to be changed.)
- Changing spacing if needed, such as removing whitespace.
- Making sure that all strengths and weaknesses lists are in bullet points as in the example. Each list item should be in a single bullet point.
- Remove all other extraneous text that does not fall into the structure above. (For example, headers not mentioned above, such as `INPUT` or `EVALUATION.`)
- Making sure that any rubric has been removed.

## Acceptance criteria
- The document listed above has been transformed to exactly match the format that I've described.
- No text has been touched except for formatting reasons, according to this document. This is not an editorial revision. You won't change any of the actual wording, arguments, et cetera. This is strictly a mechanical revision.

## Other notes.
The first three examples in the document @dataset_source.md  have already been formatted correctly to give you more examples of how you need to format the rest of the document.

You can just do the reformatting yourself, or you can write a script to do that, whichever is easiest for you. All I care about is that the document be reformatted to exactly match this format.

Please let me know if you have any questions before we proceed.

<!-- DONE -->
