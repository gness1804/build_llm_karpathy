import os
from openai import OpenAI

from models.prompts import ADVICE_COLUMNIST_SYSTEM_PROMPT

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "ft:gpt-4.1-mini")

def generate_answer(question: str) -> str:
    messages = [
        {"role": "system", "content": ADVICE_COLUMNIST_SYSTEM_PROMPT},
        {"role": "user", "content": f"QUESTION: {question}"},
    ]

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=float(os.environ.get("TEMPERATURE", 0.3)),
        top_p=float(os.environ.get("TOP_P", 0.9)),    
        max_tokens=int(os.environ.get("MAX_NEW_TOKENS", 700)),
    )
    return resp.choices[0].message.content.strip()
