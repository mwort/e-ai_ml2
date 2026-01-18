import requests
import sys
import json

# Your IONOS server address
IONOS_URL = "http://localhost:8001"

# Read input arguments
if len(sys.argv) > 3:
    query = sys.argv[1]
    session_id = sys.argv[2]
    model = sys.argv[3]
else:
    query = "Hello, how are you?"
    session_id = "unknown"
    model = "mistral:latest"

# Request payload
payload = {
    "q": query,
    "session_id": session_id,
    "model": model,
    "stream": True  # Ensure streaming response
}

try:
    # Send POST request with streaming enabled
    with requests.post(IONOS_URL, json=payload, timeout=300, stream=True) as response:
        response.raise_for_status()  # Raise error for 4xx/5xx responses

        # Read and process streaming response
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    print(json.dumps({"response": data.get("response", "")}))
                    sys.stdout.flush()  # Ensure immediate output
                except json.JSONDecodeError:
                    continue  # Skip malformed lines

except requests.exceptions.RequestException as e:
    print(json.dumps({"error": str(e)}))
    sys.stdout.flush()
