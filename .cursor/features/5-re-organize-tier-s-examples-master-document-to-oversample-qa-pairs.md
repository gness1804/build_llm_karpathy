# Re Organize Tier S Examples Master Document To Oversample Qa Pairs

## Working directory

`~/Desktop/build_llm_karpathy`

## Contents

I'm fine-tuning a large language model to serve as an advice columnist. The following document contains many question and answer pairs that I created to fine-tune the model: `sources/v2/v_2_1/tier_s_examples_master_document.md`. These question-and-answer pairs follow a simple formula: the word QUESTION:, followed by the question, and then the word "ANSWER:", followed by the answer. Then an oversampling score which I will discuss below. And finally the token `<END_OF_SET>`. 

I worked hard on this and have curated these examples to be the best possible expressions of my voice so that the model can learn my voice and reasoning style. Part of this is that I'm intentionally over-sampling my best examples. Each Q&A pair as a line `OVERSAMPLE_WEIGHT:` and a score. For example, `OVERSAMPLE_WEIGHT: 4`. The higher the score, the more the pair should be over-sampled. I use a 1-5 scoring system involving proportional representation counts. For example, a score of 2 should be over-sampled twice as much as a score of 1. A score of 3 should be over-sampled 3 times as much as a score of 1, etc. 

Can you write a script that does this? It should take in a base number as a multiplier. For instance, if I give it the number 2, then pairs with oversample weight of 1 would be duplicated once (for two total copies). Pairs with an oversample weight of 2 should have 4 total copies. Pairs with an oversample weight of 3 should have 8 total copies. 

The script should take all of the QA pairs in the input Markdown document that I noted above. It should output a new document with all the QA pairs oversampled as described above. The script should also randomize the order of the QA pairs in the output document. 

Finally, the script should NEVER alter the content of any of the QA pairs. Not even changing a letter or a word, let alone an entire sentence. The sole job of the script is purely mechanical rather than editorial. So each QA pair should be treated as immutable. The script's only job is to duplicate the QA pairs as appropriate and then randomize the order in a new output document.

Please let me know if you have any questions before we begin.
