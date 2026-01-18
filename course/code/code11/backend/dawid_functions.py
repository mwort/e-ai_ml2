import re
import json

# === Available function names for extraction ===
AVAILABLE_FUNCTIONS = [
    "get_current_datetime",
    "get_current_weather",
]

def extract_function_calls(text):
    """
    Extract function calls based on AVAILABLE_FUNCTIONS list.
    Handles JSON blocks and direct mentions.
    """
    print(f"🔍 Extracting function calls from text: {text}")

    try:
        # Suche nach JSON-ähnlichen Blöcken
        matches = re.findall(r'\{[\s\S]*?\}', text)
        for match in matches:
            try:
                parsed = json.loads(match)

                # Einzelner Funktionsaufruf
                if "function_call" in parsed:
                    func = parsed["function_call"]
                    name = func.get("name")
                    arguments = func.get("arguments", {})

                    if name in AVAILABLE_FUNCTIONS:
                        return {
                            "function_calls": [
                                {"name": name, "arguments": arguments}
                            ]
                        }
                # Mehrere Funktionsaufrufe
                if "function_calls" in parsed:
                    calls = parsed["function_calls"]
                    if isinstance(calls, dict):
                        calls = [calls]
                    result = []
                    for call in calls:
                        if call.get("name") in AVAILABLE_FUNCTIONS:
                            result.append({
                                "name": call.get("name"),
                                "arguments": call.get("arguments", {})
                            })
                    if result:
                        return {"function_calls": result}
            except json.JSONDecodeError:
                continue

        # Alternative: Falls kein JSON-Block, reine Textsuche
        for func_name in AVAILABLE_FUNCTIONS:
            if func_name in text:
                return {
                    "function_calls": [
                        {"name": func_name, "arguments": {}}
                    ]
                }
    except Exception as e:
        print(f"⚠️ Error while extracting function calls: {e}")

    return None

FUNCTION_CALL_REGEX = re.compile(r'(\{.*?"function_calls"\s*:\s*\[.*?\]\s*\})', re.DOTALL)

def get_functions(text):
    """
    Find and extract the first function_calls JSON block, remove it from text.
    """
    print(f"extracting from ... {text}")
    match = FUNCTION_CALL_REGEX.search(text)
    if match:
        function_call_str = match.group(1)
        try:
            function_call = json.loads(function_call_str)

            start, end = match.span()
            text_without_function = (text[:start] + text[end:]).strip()
            print(f"✅ Extracted function call. Remaining text: {text_without_function}")
            return function_call, text_without_function

        except json.JSONDecodeError as e:
            print(f"⚠️ JSON decode error: {e}")

    return False, text