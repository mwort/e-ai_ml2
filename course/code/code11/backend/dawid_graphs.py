from datetime import datetime
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
import pytz
import requests

# === Define the LangGraph State ===

class DawidState(TypedDict):
    functions_requested: List[dict]
    user_prompt: str
    session_id: str
    datetime: str
    final_result: Optional[str]

# === Define Functions ===

def should_call_weather(state: DawidState) -> bool:
    return any(f["name"] == "get_current_weather" for f in state["functions_requested"])


def branch_logic(state: DawidState) -> str:
    if should_call_weather(state):
        return "get_current_weather"
    else:
        return "end"

# === Define Node Functions (modifying state) ===

def get_current_datetime_node(state: DawidState) -> DawidState:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state["datetime"] = now
    state["final_result"] = f"The current datetime is: {now}"
    return state

def get_current_weather_node(state: DawidState) -> DawidState:
    state["final_result"] = "The current weather is sunny, 20°C."
    return state

def branch_after_datetime(state: DawidState) -> str:
    if should_call_weather(state):
        return "get_current_weather"
    else:
        return "__end__"

# === Build the Graph ===

def create_dawid_graph():
    graph = StateGraph(DawidState)

    graph.add_node("get_current_datetime", get_current_datetime_node)
    graph.add_node("get_current_weather", get_current_weather_node)

    graph.set_entry_point("get_current_datetime")

    # Correct: no "condition=" keyword!
    graph.add_conditional_edges(
        "get_current_datetime",
        branch_logic,
        {
            "get_current_weather": "get_current_weather",
            "end": END
        }
    )

    graph.add_edge("get_current_weather", END)

    return graph.compile()


# === Entry Point ===

def run_dawid_graph(user_prompt, session_id, functions_requested):
    print(f"\n🔵 [run_dawid_graph] Entered function for session '{session_id}'")
    print(f"📝 Prompt: {user_prompt}")
    print(f"🛠️ Functions requested: {functions_requested}")

    compiled_graph = create_dawid_graph()

    # Initialize state
    state = {
        "functions_requested": functions_requested,
        "user_prompt": user_prompt,
        "session_id": session_id,
        "datetime": None,
        "final_result": None
    }

    print("⚙️ Starting dawid graph ...")
    state = compiled_graph.invoke(state)
    print("✅ Dawid graph completed.")

    print(f"🎯 Final state: {state}\n")

    return state.get("final_result", "No result.")
