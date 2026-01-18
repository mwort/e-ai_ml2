import httpx
import json
import random
import os

LLAMA_SERVER_PORTS = [8080, 8081, 8082, 8083, 8084, 8085]

def get_llama_url():
    port = random.choice(LLAMA_SERVER_PORTS)
    print(f"🔀 PID {os.getpid()} -> llama.cpp on port {port}")
    return f"http://localhost:{port}/completion"
# LLAMA_SERVER_URL = "http://localhost:8080/completion"

async def call_llama(prompt, model="llama3:8b"):
    """
    Non-streaming LLaMA call to get a full text response.
    Wraps prompt using ChatML format expected by llama.cpp.
    """
    llama_url = get_llama_url()
    print(f"Query to {llama_url}")

    headers = {
        "Content-Type": "application/json"
    }

    prompt_chat = "<|begin_of_text|>"
    prompt_chat += (
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful assistant.\n"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        + prompt.strip() +
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    payload = {
        "prompt": prompt_chat,
        "n_predict": 512,
        "stream": False,
        "stop": ["<|eot_id|>", "User:", "Next Question."]
    }

    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(llama_url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("content", "").strip()

async def call_llama_streaming(session_history, user_prompt, session_id, system_prompt, stream_callback, model):
    prompt_chat = "<|begin_of_text|>"

    # System prompt
    prompt_chat += (
        "<|start_header_id|>system<|end_header_id|>\n\n" +
        system_prompt.strip() +
        "<|eot_id|>"
    )

    # History
    for entry in session_history:
        if entry["timestamp"] != "SUMMARY":
            prompt_chat += (
                "<|start_header_id|>user<|end_header_id|>\n\n" +
                entry["query"].strip() +
                "<|eot_id|>" +
                "<|start_header_id|>assistant<|end_header_id|>\n\n" +
                entry["response"].strip() +
                "<|eot_id|>"
            )

    # Final user prompt
    prompt_chat += (
        "<|start_header_id|>user<|end_header_id|>\n\n" +
        user_prompt.strip() +
        "<|eot_id|>" +
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    payload = {
        "prompt": prompt_chat,
        "n_predict": 256,
        "stream": True,
        "stop": ["<|eot_id|>", "User:", "Next Question."]
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            LLAMA_SERVER_URL = get_llama_url()
            print(f"Query to {LLAMA_SERVER_URL}")
            async with client.stream("POST", LLAMA_SERVER_URL, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    text = await response.aread()
                    yield json.dumps({"response": f"❌ LLaMA error {response.status_code}: {text.decode()}"}) + "\n"
                    return

                buffer = ""
                async for chunk in response.aiter_bytes():
                    buffer += chunk.decode("utf-8", errors="ignore")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if not line.strip().startswith("data:"):
                            continue
                        try:
                            json_str = line[len("data:"):].strip()
                            data = json.loads(json_str)
                            yield stream_callback(data.get("content", ""))
                        except Exception:
                            continue
    except httpx.RequestError as e:
        yield json.dumps({"response": f"❌ Error contacting LLaMA: {str(e)}"}) + "\n"
