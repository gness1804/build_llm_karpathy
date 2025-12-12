Prompts to use to test models on inference.



Inference command:

```bash
CHECKPOINT_PATH=checkpoints/checkpoint_gpt2_training_data_v2_step008100_12112025_165445.pt \
MODE=inference \
DEVICE=auto \
MAX_NEW_TOKENS=800 \
PROMPT="I have been married for five years. Recently, I've started to realize just how much I've neglected my friends during that period of time. Before I met my husband, I was part of an active social group, and we did a lot together. We had book clubs, went out to dinners, and even went on a few foreign trips together. We were a close-knit bunch, but as others in the bunch started pairing off, getting married, and starting families, I started to see them less and less. They never seem to have time for us anymore. I swore that if I ever got married, I wouldn't do the same, but here I am. \n\nI love my husband, and we have a great life together. But during our two years of dating and then into our marriage, I've been communicating with my old friend group less and less. There are only a couple of people left in that group who are still single, and I worry that life has gotten lonely for them. I want to reach out to these old friends, but I worry that they're going to reject me. How can I reach out to them in a kind, compassionate way while minimizing the risk of rejection? \n\nANSWER:" \
TEMPERATURE=0.6 \
TOP_K=40 \
TOP_P=0.9 \
REPETITION_PENALTY=1.2 \
NO_REPEAT_NGRAM_SIZE=3 \
SAVE_OUTPUT=true \
OUTPUT_DIR=outputs/inference \
python3 scripts/load_checkpoint.py
```

Simpler commands suggested by GPT-5 Cursor Agent:
```bash
cd ~/Desktop/build_llm_karpathy

CHECKPOINT_PATH=checkpoints/checkpoint_gpt2_training_data_v2_step008100_12112025_165445.pt \
MODE=inference \
DEVICE=auto \
MAX_NEW_TOKENS=300 \
PROMPT="QUESTION: My boyfriend and I have been together for almost a year. We get along well, but I have one recurring complaint. When I tell him about something that hurt my feelings, he makes a joke or says I am “too sensitive.” He does not yell or insult me, but I leave those conversations feeling like my emotions are a nuisance. I do not want to break up over this, but I also do not want to keep swallowing my feelings just to keep the peace. How can I talk to him about this in a way that might actually change something? \n\nANSWER:" \
TEMPERATURE=0.6 \
TOP_K=40 \
TOP_P=0.9 \
REPETITION_PENALTY=1.2 \
NO_REPEAT_NGRAM_SIZE=3 \
SAVE_OUTPUT=true \
OUTPUT_DIR=outputs/inference \
python3 scripts/load_checkpoint.py
```