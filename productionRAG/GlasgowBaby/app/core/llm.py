# app/core/llm.py

import ollama
import time
from app.config import settings

def generate_answer(prompt, system_message=None, history=None, retries=3):
        for attempt in range(retries):
            try:
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                if history:
                    messages.extend(history)
                messages.append({"role": "user", "content": prompt})
                response = ollama.chat(
                    model=settings.LLM_MODEL,
                    messages=messages,
                    options={"temperature": 0.2, "num_predict": 500}
                )
                return response['message']['content'].strip()
            except Exception as e:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)