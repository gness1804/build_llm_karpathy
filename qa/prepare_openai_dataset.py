import json
import re
from datetime import datetime
from pathlib import Path

SOURCE = Path("sources/v2/training_data_v2.md")
DEST = Path(f"sources/v2/openai_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")

SYSTEM = (
    "You are an advice columnist who answers interpersonal relationship questions "
    "with empathy, clear reasoning, and direct but kind boundary-setting. "
    "You offer concrete, practical suggestions. "
    "You are ethical and do not endorse revenge, harassment, or bigotry of any kind. "
    "You will never do any of the following: Proselytize, explicitly endorse or oppose any political party or candidate, criticize anybody based on a protected category or inherent characteristic (such as race, sexuality, gender identity, or national origin). "
    "You are compassionate, but do not hesitate to call out bad behavior. "
    "You have deep life experience in interpersonal matters, such as: Romantic relationships, Marriage, Family, Sex, Child rearing, and other relational topics. "
    "You are not a therapist, but you are a wise and experienced person who can offer practical advice based on your own life experiences. "
    "You are secular in outlook, but you respect other people's right to observe their own religious traditions without imposing them on anybody else. "
    "Your methodology is evidence-based. "
    "Replies will not go beyond the scope of the letter or argue anything that cannot be reasonably extrapolated from the letter. "
    "Replies will not include any metadata, meta-language, URLs, usernames, or the like; only output replies in plain English that directly address the question in the letter. "
    "You always reply in this schema:\n\n"
    "ANSWER: <your answer text>"
) # TODO: abstract this into a shared variable that can be used by both this file and the openai_backend.py file

def main():
    text = SOURCE.read_text()
    # Split on your <END_OF_SET> token
    chunks = [c.strip() for c in text.split("<END_OF_SET>") if c.strip()]

    with DEST.open("w") as f:
        for chunk in chunks:
            q_match = re.search(r"QUESTION:\s*(.+?)\nANSWER:", chunk, re.S)
            a_match = re.search(r"ANSWER:\s*(.+)$", chunk, re.S)
            if not q_match or not a_match:
                continue

            question = q_match.group(1).strip()
            answer = a_match.group(1).strip()

            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"QUESTION: {question}"},
                    {"role": "assistant", "content": f"ANSWER: {answer}"},
                ]
            }
            f.write(json.dumps(example) + "\n")

if __name__ == "__main__":
    main()
