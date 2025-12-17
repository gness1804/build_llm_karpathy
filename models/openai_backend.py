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

    # Safely parse environment variables with proper defaults
    # Handle empty strings, whitespace, and invalid values by falling back to defaults
    def safe_float(env_var: str, default: float) -> float:
        value = os.environ.get(env_var, "").strip()
        if not value:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def safe_int(env_var: str, default: int) -> int:
        value = os.environ.get(env_var, "").strip()
        if not value:
            return default
        try:
            parsed = int(value)
            # Ensure max_tokens is reasonable (at least 1)
            if env_var == "MAX_NEW_TOKENS" and parsed < 1:
                return default
            return parsed
        except (ValueError, TypeError):
            return default
    
    temperature = safe_float("TEMPERATURE", 0.3)
    top_p = safe_float("TOP_P", 0.9)
    max_new_tokens = safe_int("MAX_NEW_TOKENS", 700)

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )
    return resp.choices[0].message.content.strip()
