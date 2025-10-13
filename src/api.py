from __future__ import annotations
import os
from openai import OpenAI

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))

def chat_complete(prompt: str, model: str, max_tokens: int = 256, top_p: float = 0.95) -> str:
    resp = _client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        top_p=top_p,
        max_tokens=max_tokens,
        n=1,
        stream=False,
    )
    return resp.choices[0].message.content.strip()
