# Determine And Display The Scoring Spread Of Qa Pairs For The V3 Dataset

## Working directory

`~/Desktop/build_llm_karpathy`

## Contents

The file `sources/v3/dataset_source.md` is a list of question and answer sets for an advice column LLM that I am building. Each set has a score from 0 to 10. The score can be found under the `EVALUATION:` section in each example. The score refers to the rating of the draft response in each example, as you can see in the document. I'm aiming for the following spread of scores across the entire document:

```text
10: ~1-2 percent
(Perfect response or virtually perfect. This should be very rare or non-existent. It's okay if there are no 10s in the dataset.)

8–9: ~40–50 percent
(good but improvable, typical base output)

6–7: ~30–35 percent
(vague, therapy-ish, hedgy, missing clarity)

4–5: ~10–15 percent
(clearly flawed but realistic)

3 or below: ~5 percent
(teaches “never do this”)
```

Can you look over this document and give me the CURRENT spread of scores? And then show me the difference between the current spread and the ideal spread as I just noted above. This will be useful in comparing how close I am to the benchmark as I progress. Also tell me how many examples in the document are currently unscored.

An example response from you could look like:

```text
Here are the results of the current score spread from the file `sources/v3/dataset_source.md`: 

10: 
CURRENT: 0 %
IDEAL: 0-2 percent
You are fine as is, or you can try to create one perfect example.

8–9: 
CURRENT: 32% 
IDEAL: ~40–50 percent
You need approximately N MORE examples to score in this range to bring the total of examples in this document up to the ideal amount.

6–7: 
CURRENT: 50%
IDEAL: ~30–35 percent
You need approximately O FEWER examples to bring the spread down to the ideal percentage.

4–5: 
CURRENT: 12%
IDEAL: ~10–15 percent
Nice! You are within the acceptable percent range for this number of examples in this point range.

3 or below: 
CURRENT: 5%
IDEAL: ~5 percent
Reduce the number of these examples by P to get to the ideal range.

UNSCORED: 3%
IDEAL: 0%
You need to score X more examples to get to the ideal range.
```
