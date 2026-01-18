import os
import json
import asyncio
import datetime

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from dawid_identity import load_full_system_prompt
from dawid_openai import call_openai_streaming
from dawid_ollama import call_ollama_streaming
from dawid_ollama import call_ollama
from dawid_llama import call_llama_streaming
from dawid_user_history import load_session, update_session_history, background_tasks, background_summarize, KEEP_INTERACTIONS
from dawid_functions import extract_function_calls, get_functions
from dawid_graphs import run_dawid_graph

app = FastAPI()
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("https_proxy", None)

# Enable CORS (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import APIRouter
from starlette.responses import StreamingResponse
from fastapi.responses import PlainTextResponse

@app.get("/ollama_test")
async def ollama_test():
    response = await call_ollama("say pong!", model="llama3")
    return PlainTextResponse(response)

@app.post("/")
async def stream_response(request: Request):
    try:
        body = await request.json()
        user_prompt = body.get("q", "Hello?")
        session_id = body.get("session_id", "default")
        model = body.get("model", "mistral:latest")

        system_prompt = load_full_system_prompt()

        # Final answer
        session_history = load_session(session_id)
        print(f"📨 Prompt received: {user_prompt} [session={session_id}]")
        print(f"📥 Handling session {session_id} in PID {os.getpid()}")

        response_accumulator = ""

        def stream_callback(chunk):
            nonlocal response_accumulator
            response_accumulator += chunk
            return chunk

        async def run_streaming():
            nonlocal response_accumulator

            if model.startswith("gpt"):
                stream = call_openai_streaming(
                    session_history, user_prompt, session_id,
                    system_prompt, stream_callback, model
                )
            else:
                stream = call_llama_streaming(
                    session_history, user_prompt, session_id,
                    system_prompt, stream_callback, model
                )

            buffering = True
            nbuffer = 1
            buffer = ""
            word_count = 0
            function_call_json = False
            last_parts = []

            start_str="One moment ..."
            yield (json.dumps({"response": start_str}) + "\n").encode("utf-8")

            async for part in stream:
                if not part:
                    continue

                last_parts.append(part)
                if len(last_parts) > 4:
                    last_parts.pop(0)  # keep only the last 4 parts
                #print(last_parts)

                if buffering:
                    buffer += part

                    word_count = len(buffer.split())

                    if word_count >= nbuffer:
                        print(f"🟰 Flushing buffer after {word_count} words!")
                        print(f"{buffer}")
                        #-------------------------------------------------
                        # Function Calls here
                        #-------------------------------------------------
                        #function_call_json, buffer = get_functions(buffer)
                        if function_call_json:
                            print(f"📦 1 Function call detected and extracted: {function_call_json}")
                            functions_requested = function_call_json.get("function_calls", [])
                            result = run_dawid_graph(
                                user_prompt=user_prompt,
                                session_id=session_id,
                                functions_requested=functions_requested
                            )
                            yield json.dumps({"response": result}).encode("utf-8")
                            return  # Important! Skip normal session update if function call handled
                        else:
                            print("1 no function fund")
                        yield (json.dumps({"response": buffer}) + "\n").encode("utf-8")
                        buffer = ""
                        buffering = False
                        #-------------------------------------------------
                else:
                    yield (json.dumps({"response": part}) + "\n").encode("utf-8")

            if buffer.strip():
                #-------------------------------------------------
                # Function calls here again
                #-------------------------------------------------
                #function_call_json, buffer = get_functions(buffer)
                if function_call_json:
                    print(f"📦 2 Function call detected and extracted: {function_call_json}")
                if function_call_json:
                    functions_requested = function_call_json.get("function_calls", [])
                    result = run_dawid_graph(
                        user_prompt=user_prompt,
                        session_id=session_id,
                        functions_requested=functions_requested
                    )
                    yield json.dumps({"response": result}).encode("utf-8")
                    return  # Important! Skip normal session update if function call handled
                else:
                    print("2 no function fund")
                yield (json.dumps({"response": buffer})+ "\n").encode("utf-8")
                #-------------------------------------------------
                       
            n_entries = await update_session_history(session_id, user_prompt, response_accumulator)
            print(f"Session history n_entries={n_entries}.")

            if (n_entries > KEEP_INTERACTIONS + 5) and (session_id != "keepalive"):
                if session_id not in background_tasks:
                    background_tasks[session_id] = asyncio.create_task(background_summarize(session_id))
                    
        return StreamingResponse(run_streaming(), media_type="text/event-stream")

    except Exception as e:
        print(f"❌ Error in stream_response: {e}")
        return JSONResponse(status_code=500, content={
            "error": f"Exception in streaming endpoint: {str(e)}"
        })


@app.post("/upload")
async def upload_file(uploaded_file: UploadFile = File(...)):
    print(f"Trying to upload file {uploaded_file.filename}")
    try:
        contents = await uploaded_file.read()
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, uploaded_file.filename)

        with open(filepath, "wb") as f:
            f.write(contents)

        print(f"📁 Uploaded: {filepath}")
        return JSONResponse(content={
            "status": "success",
            "message": f"✅ File '{uploaded_file.filename}' uploaded to backend successfully."
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": f"❌ Upload failed: {str(e)}"
        })


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    print(f"🚀 Starting FastAPI server on port {port} ...")
    #uvicorn.run("dawid_server:app", host="0.0.0.0", port=port, reload=True)
    uvicorn.run("dawid_server:app", host="0.0.0.0", port=port, workers=4)
