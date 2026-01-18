#!/usr/bin/env python
# coding: utf-8

# # Function Calling Basics (Local LLMs)
# 
# This notebook demonstrates the **core idea of function calling** using a **local large language model** (LLaMA 3 via Ollama), *without* any agent framework or API-level support.
# 
# The goal is to understand function calling in its most basic form:
# 
# - The **LLM does not execute code**
# - The **LLM decides whether an action is needed**
# - The **LLM emits a structured request** (JSON)
# - The **system executes the function**
# 
# This approach reflects how early agent systems (including DAWID, 2024) worked:
# by **constraining model output** and **parsing structured JSON**.
# 
# Later notebooks will show how modern APIs and frameworks make this more robust.
# 
# ---
# 
# ## Demo setup
# 
# We define a single conceptual function:
# 
# **`get_time(timezone: string)`**
# 
# If the user asks for the current time, the model should return a JSON object
# requesting a call to this function — and *nothing else*.
# 
# This is *manual* function calling, implemented entirely through prompting.
# 

# In[1]:


# This cell demonstrates function calling with a local LLaMA 3 model via Ollama.
# The model is instructed to return a JSON "tool call" instead of free text.

prompt = """
You are an AI assistant.

You may respond in ONE of two ways only:

1. Normal text
2. A JSON object of the following exact form:

{
  "tool_call": {
    "name": "<function name>",
    "arguments": { ... }
  }
}

Available function:
- get_time(timezone: string)

Rules:
- If the user asks for the current time, you MUST return ONLY the JSON tool call.
- Do NOT include explanations.
- Do NOT include markdown.
- Do NOT include any text outside the JSON object.
- The JSON must be valid and parseable.

User request:
What time is it in UTC?
"""

print("Run the following command in your shell:\n")
print("ollama run llama3:latest << 'EOF'")
print(prompt.strip())
print("EOF")


# In[ ]:





# In[2]:


"""
Call a local LLaMA 3 model via Ollama from inside Jupyter,
and capture a JSON-style function call.

This demonstrates:
- local LLM execution
- constrained output
- machine-readable tool requests
"""

import subprocess
import json
import textwrap

prompt = textwrap.dedent("""
You are an AI assistant.

You may respond in ONE of two ways only:

1. Normal text
2. A JSON object of the following exact form:

{
  "tool_call": {
    "name": "<function name>",
    "arguments": { ... }
  }
}

Available function:
- get_time(timezone: string)

Rules:
- If the user asks for the current time, you MUST return ONLY the JSON tool call.
- Do NOT include explanations.
- Do NOT include markdown.
- Do NOT include any text outside the JSON object.
- The JSON must be valid and parseable.

User request:
What time is it in UTC?
""")

# Call Ollama
result = subprocess.run(
    ["ollama", "run", "llama3:latest"],
    input=prompt,
    text=True,
    capture_output=True,
    check=True,
)

raw_output = result.stdout.strip()

print("Raw model output:\n")
print(raw_output)


# In[3]:


"""
Parse the JSON function call emitted by the model.
"""

try:
    tool_call = json.loads(raw_output)["tool_call"]
    print("Parsed tool call:")
    print("Tool name:", tool_call["name"])
    print("Arguments:", tool_call["arguments"])
except Exception as e:
    print("❌ Failed to parse JSON output")
    print("Error:", e)


# In[4]:


"""
Execute the requested function.
Fixes name shadowing between string arguments and datetime.timezone.
"""

from datetime import datetime
import pytz

def get_time(timezone: str) -> str:
    """
    Return the current time in the given timezone.
    Supported values: UTC, CET
    """
    tz = timezone.upper()

    if tz == "UTC":
        return datetime.now(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    if tz == "CET":
        return datetime.now(pytz.timezone("Europe/Berlin")).strftime(
            "%Y-%m-%d %H:%M:%S CET"
        )

    return f"Unsupported timezone: {timezone}"


# In[5]:


import os

# DWD proxy
os.environ["HTTP_PROXY"]  = "http://ofsquid.dwd.de:8080"
os.environ["HTTPS_PROXY"] = "http://ofsquid.dwd.de:8080"

# Optional but recommended
os.environ["http_proxy"]  = os.environ["HTTP_PROXY"]
os.environ["https_proxy"] = os.environ["HTTPS_PROXY"]

import os

# Explicitly bypass proxy for local Ollama
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

print("✅ Proxy bypass set for localhost")


# In[6]:


"""
Function calling via Ollama REST API.

This notebook demonstrates:
- REST-based LLM calls
- JSON tool-call extraction
- system-side function execution
- feeding results back to the model
"""

import requests
import json
from datetime import datetime, timezone as dt_timezone
from IPython.display import Markdown, display


# In[7]:


from datetime import datetime
from datetime import timezone as dt_timezone

"""
System tool implementation.
The LLM never executes this.
"""

def get_time(timezone: str) -> str:
    tz = timezone.upper()

    if tz == "UTC":
        return datetime.now(dt_timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    elif tz == "CET":
        # Local time; assumes system is CET/CEST aware
        return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S CET")

    else:
        return f"Unsupported timezone: {timezone}"


# In[8]:


"""
Prompt that forces JSON-only tool calls.
"""

TOOL_PROMPT = """
You are an AI assistant.

You may respond in ONE of two ways only:

1. Normal text
2. A JSON object of the following exact form:

{
  "name": "get_time",
  "arguments": {
    "timezone": "UTC"
  }
}

Available function:
- get_time(timezone: string)

Rules:
- If the user asks for the current time, return ONLY the JSON tool call.
- No explanations.
- No markdown.
- No additional text.
- Output must be valid JSON.

User request:
What time is it in UTC?
"""


# In[9]:


"""
Call Ollama via REST API.
"""

def ollama_call(prompt: str, model="llama3:latest") -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


raw_output = ollama_call(TOOL_PROMPT)

print("Raw model output:\n")
print(raw_output)


# In[10]:


"""
Parse the JSON tool call emitted by the model.
"""

tool_call = json.loads(raw_output)

tool_name = tool_call["name"]
tool_args = tool_call["arguments"]

print("Parsed tool call:")
print("Tool:", tool_name)
print("Arguments:", tool_args)


# In[11]:


"""
Execute the tool and display the result in grey.
"""

if tool_name == "get_time":
    # tool_args is expected to be {"timezone": "..."}
    tool_result = get_time(**tool_args)
else:
    raise ValueError(f"Unknown tool: {tool_name}")

display(
    Markdown(
        "<div style='color:gray; font-style:italic'>"
        f"Tool result: {tool_result}"
        "</div>"
    )
)


# In[12]:


"""
Feed the tool result back to the LLM
and ask for a final user-facing answer.
"""

FOLLOWUP_PROMPT = f"""
The following tool was executed:

Function: get_time
Arguments: {tool_args}
Result: {tool_result}

Now produce a short, clear answer for the user.
"""

final_answer = ollama_call(FOLLOWUP_PROMPT)

display(Markdown(final_answer))


# # Lets define a function now to do all this in one go

# In[13]:


def build_tool_prompt(query: str) -> str:
    return f"""
You can call the following function:

Function name: get_time
Arguments (JSON):
  - timezone_name (string)

If the user request requires calling the function,
respond ONLY with valid JSON like this:

{{
  "name": "get_time",
  "arguments": {{
    "timezone_name": "UTC"
  }}
}}

If no function is needed, answer normally.

User request:
{query}
"""


# In[14]:


import re
import json

def ai(query: str) -> str:
    # ------------------------------------------------------------
    # Step 1: Ask model for a tool call
    # ------------------------------------------------------------
    prompt = build_tool_prompt(query)
    raw = ollama_call(prompt)

    print("\n--- Raw model output ---")
    print(raw)

    # ------------------------------------------------------------
    # Step 2: Extract JSON tool call (if present)
    # ------------------------------------------------------------
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        # No tool call → model answered directly
        return raw.strip()

    tool_call = json.loads(match.group(0))

    # ------------------------------------------------------------
    # Step 3: Execute tool locally
    # ------------------------------------------------------------
    if tool_call.get("name") == "get_time":
        args = tool_call.get("arguments", {})

        # Normalize argument names
        if "timezone_name" in args:
            args["timezone"] = args.pop("timezone_name")

        result = get_time(**args)
    else:
        result = f"Unknown tool: {tool_call.get('name')}"

    print("\n--- Tool result (executed) ---")
    print(result)

    # ------------------------------------------------------------
    # Step 4: Feed tool result back to the model
    # ------------------------------------------------------------
    followup_prompt = f"""
The user asked:
{query}

You requested a tool call and received this result:
{result}

Provide a clear and concise answer to the user.
"""

    final = ollama_call(followup_prompt)

    return final.strip()


# In[15]:


ai("What is the current time in CET?")


# In[ ]:




