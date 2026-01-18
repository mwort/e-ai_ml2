"""
📄 dawid_openai.py – OpenAI Integration for DAWID Assistant
-----------------------------------------------------------

This module handles all communication with the OpenAI API (>=1.0.0)
for the DAWID assistant backend. It supports:

- Loading the OpenAI API key from environment variables or `.env`
- Defining a default model (`gpt-4o-mini`)
- Sending streamed responses using OpenAI's modern chat API
- Sending simple non-streamed responses
- Handling exceptions gracefully

Dependencies:
- openai >= 1.0.0
- python-dotenv
"""

import os
import json
from dotenv import load_dotenv
from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI  # added for non-streaming call

# === Load .env and API key ===
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print(f"✅ OpenAI API key loaded (length {len(api_key)} chars)")
else:
    print("❌ OPENAI_API_KEY not found in environment or .env")

client = AsyncOpenAI(api_key=api_key)
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# === Streaming Call (async) ===
async def call_openai_streaming(session_history, user_prompt, session_id, system_identity, stream_callback, model=DEFAULT_OPENAI_MODEL):
    """
    Asynchronous OpenAI streaming for FastAPI.
    """
    try:
        messages = [{"role": "system", "content": system_identity}]
        for entry in session_history:
            if entry["timestamp"] != "SUMMARY":
                messages.append({"role": "user", "content": entry["query"]})
                messages.append({"role": "assistant", "content": entry["response"]})
        messages.append({"role": "user", "content": user_prompt})

        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield stream_callback(delta.content)

    except Exception as e:
        error_message = f"❌ OpenAI streaming error: {str(e)}"
        print(error_message)
        yield json.dumps({"error": error_message}) + "\n"

# === Non-Streaming Simple Call (sync) ===
def call_openai(prompt: str, model=DEFAULT_OPENAI_MODEL):
    """
    Simple synchronous call to OpenAI without streaming.
    Returns the response content as a string.
    """
    llm = ChatOpenAI(
        model=model,
        temperature=0,
        api_key=os.environ["OPENAI_API_KEY"]
    )
    response = llm.invoke(prompt)
    return response.content.strip()
