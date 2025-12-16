import json
import re
from datetime import datetime
from pathlib import Path

from models.prompts import ADVICE_COLUMNIST_SYSTEM_PROMPT

SOURCE = Path("sources/v2/training_data_v2.md")
DEST = Path(f"sources/v2/openai_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")

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
                    {"role": "system", "content": ADVICE_COLUMNIST_SYSTEM_PROMPT},
                    {"role": "user", "content": f"QUESTION: {question}"},
                    {"role": "assistant", "content": f"ANSWER: {answer}"},
                ]
            }
            f.write(json.dumps(example) + "\n")

if __name__ == "__main__":
    main()
