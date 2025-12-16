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
        temperature=0.3, # TODO: Make this a parameter
        top_p=0.9, # TODO: Make this a parameter    
        max_tokens=700, # TODO: Make this a parameter
    )
    return resp.choices[0].message.content.strip()
