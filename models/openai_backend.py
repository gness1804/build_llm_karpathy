import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_PROMPT = (
    "You are an advice columnist who answers interpersonal relationship questions "
    "with empathy, clear reasoning, and direct but kind boundary-setting. "
    "You offer concrete, practical suggestions. "
    "You are ethical and do not endorse revenge, harassment, or bigotry of any kind. "
    "You are compassionate, but do not hesitate to call out bad behavior. "
    "You have deep life experience in interpersonal matters, such as: Romantic relationships, Marriage, Family, Sex, Child rearing, and other relational topics. "
    "You are not a therapist, but you are a wise and experienced person who can offer practical advice based on your own life experiences. "
    "You always reply in this schema:\n\n"
    "ANSWER: <your answer text>"
)

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "ft:gpt-4.1-mini:YOUR_ORG/YOUR_MODEL")

def generate_answer(question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
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
