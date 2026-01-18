"""
📄 dawid_ollama.py – Ollama Integration for DAWID Server (Async Version)
"""

import json
import httpx
import os
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("https_proxy", None)

OLLAMA_URL = "http://localhost:11434/api/generate"

async def call_ollama(prompt, model="mistral:latest"):
    """
    Non-streaming Ollama call to get a full text response.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("response", "").strip()


async def call_ollama_streaming(session_history, user_prompt, session_id, system_identity, stream_callback, model="mistral:latest"):
    print(f"🌍 Using Ollama model: {model}")

    context = ""
    for entry in session_history:
        if entry["timestamp"] == "SUMMARY":
            context += f"Summary:\n{entry['response']}\n\n"
        else:
            context += f"User: {entry['query']}\nAssistant: {entry['response']}\n"

    full_prompt = f"System:\n{system_identity}\n\n{context}\nUser: {user_prompt}\nAssistant:"

    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": True
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream("POST", OLLAMA_URL, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    text = await response.aread()
                    yield json.dumps({"response": f"❌ Ollama error {response.status_code}: {text.decode()}"}) + "\n"
                    return

                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            chunk = data.get("response", "")
                            yield stream_callback(chunk)
                        except json.JSONDecodeError:
                            continue
    except httpx.RequestError as e:
        yield json.dumps({"response": f"❌ Error contacting Ollama: {str(e)}"}) + "\n"
