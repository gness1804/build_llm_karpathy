# Create Script To Convert New Document Structure Into Json L

## Working directory

`~/Desktop/build_llm_karpathy`

## Contents
A script that takes a document that's structured like the following and converts it to JSON-L: `sources/v3/dataset_source.md`. The output JSONL will be used to fine-tune an LLM that will evaluate advice column drafts and rewrite them in a distinctive voice. In the JSON-L, each training example should look like:

```text
system: your stable instruction prompt (short, fixed). This will be pulled from the document `models/prompts.py`. 

user: includes QUESTION + DRAFT_RESPONSE

assistant: includes SCORE + STRENGTHS + WEAKNESSES + REVISED_RESPONSE
```

You should create a small script that parses the Markdown blocks from the input file into structured fields, then emits JSONL rows based on the structure breakdown above. 

If possible, the following things should be enforced during conversion:

- Strip trailing spaces and normalize newlines consistently.
- Ensure the score is always one of: 1.0, 1.5, 2.0, ... 10.0 (No 7.2, 6.25, etc.)
- Keep bullet formatting identical across all examples.
- Do not include links.
- Remove em dashes FROM THE `REVISED_RESPONSE` if they sneak in. Insert the most appropriate alternate punctuation. But please leave em dashes in the `DRAFT_RESPONSE`. The purpose of this is that I'm training the fine-tuned model to not use em dashes. So if I have em dashes in the final response, then that's going to train the model to keep using them.
- Aside from the above, all content in the question, `REVISED_RESPONSE`, and `DRAFT_RESPONSE` should remain untouched. The script you're going to create is strictly a mechanical one that's not to touch content except to make the minor edits noted above.

## Acceptance criteria
- A script that correctly transforms data structured as in the document @dataset_source.md into JSONL that matches the model described above.
- The script should optionally take in an input document. But the default should be @dataset_source.md.
- The script should check if the data in the input document is structured correctly--as in, matches the same structure as in @dataset_source.md. If not, it should throw an error rather than trying to reformat the incorrectly structured data.
- There should be an optional argument for a document to output the results to. The default output document should be `sources/v3/<original_document_name>_{timestamp}.jsonl`.
- The final JSONL output should be free of errors and should be ready to fine-tune an LLM.

<!-- DONE -->
