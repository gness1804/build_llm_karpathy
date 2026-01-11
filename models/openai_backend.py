import os
from openai import OpenAI

from models.prompts import ADVICE_COLUMNIST_SYSTEM_PROMPT, SYSTEM_PROMPT_V3

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "ft:gpt-4.1-mini")

def generate_answer(input: str, version: str = "v1") -> str:
    if version == "v1":
        messages = [
            {"role": "system", "content": ADVICE_COLUMNIST_SYSTEM_PROMPT},
            {"role": "user", "content": f"QUESTION: {input}"},
        ]
    elif version == "v3":
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_V3},
            {"role": "user", "content": f"{input}"},
        ]
    else:
        raise ValueError(f"Invalid version: {version}")

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=float(os.environ.get("TEMPERATURE", 0.3)),
        top_p=float(os.environ.get("TOP_P", 0.9)),    
        max_tokens=int(os.environ.get("MAX_NEW_TOKENS", 700)),
    )
    return resp.choices[0].message.content.strip()
