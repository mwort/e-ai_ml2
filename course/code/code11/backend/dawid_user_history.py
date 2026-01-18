"""
📄 dawid_user_history.py – Session management for DAWID Assistant

Handles:
- Loading and saving user sessions and summaries
- Keeping session files short by summarizing old interactions
- Managing asynchronous background summarization tasks

Configuration:
- KEEP_INTERACTIONS: number of full interactions before summarizing
- background_tasks: tracks running summarization tasks per session_id
"""

import os
import json
from datetime import datetime
from dawid_llama import call_llama  # oder OpenAI
import asyncio

# === Session and Summarization Parameters ===
KEEP_INTERACTIONS = 5  # Max full interactions to keep before summarizing
background_tasks = {}  # Tracks running background summarizations per session_id
SESSION_DIR = "./sessions"
os.makedirs(SESSION_DIR, exist_ok=True)
SUMMARY_DIR = "./summaries"
os.makedirs(SUMMARY_DIR, exist_ok=True)

def load_session(session_id):
    """Loads a session by combining summaries and interaction history."""
    session_file = os.path.join(SESSION_DIR, f"{session_id}.json")
    summary_file = os.path.join(SUMMARY_DIR, f"{session_id}_summaries.json")

    # Load interactions
    interactions = []
    if os.path.exists(session_file):
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                interactions = json.load(f)
        except json.JSONDecodeError:
            interactions = []

    # Load summaries
    summaries = []
    if os.path.exists(summary_file):
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                summaries = json.load(f)
        except json.JSONDecodeError:
            summaries = []

    # Convert summaries into the same format as interactions
    converted_summaries = []
    for summary in summaries:
        converted_summaries.append({
            "timestamp": summary.get("timestamp", "SUMMARY"),
            "query": "Summary of earlier interaction history",
            "response": summary.get("summary", "")
        })

    # Combine: first all summaries, then all current interactions
    full_history = converted_summaries + interactions

    return full_history

def save_session(session_id, session_history):
    """Speichert die Session auf Disk."""
    session_file = os.path.join(SESSION_DIR, f"{session_id}.json")
    with open(session_file, "w", encoding="utf-8") as file:
        json.dump(session_history, file, indent=4)


async def update_session_history(session_id, user_query, llm_response):
    """
    Fügt nur den neuesten Eintrag hinzu und speichert nur die echten Interaktionen.
    """
    if asyncio.iscoroutine(llm_response):
        llm_response = await llm_response

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session_history = load_session(session_id)

    # Only keep real interactions, exclude summaries
    interactions = [e for e in session_history if e.get("timestamp") != "SUMMARY"]

    interactions.append({
        "timestamp": current_time,
        "query": user_query,
        "response": llm_response
    })

    save_session(session_id, interactions)
    return len(interactions)


async def summarize_entries(entries, model="llama3:8b"):
    """
    Erstellt eine Zusammenfassung über eine Liste von Interaktionen.
    
    Args:
        entries (list): Liste von Einträgen (jeweils mit 'query' und 'response').
        model (str): Modellname (z.B. 'mistral:latest').

    Returns:
        summary_text (str): Die erstellte Zusammenfassung.
        n_entries (int): Anzahl der zusammengefassten Einträge.
    """
    if not entries:
        return "No content to summarize.", 0

    # Build the summarization prompt
    summary_prompt = (
        "Create a concise summary of the following user-assistant interactions. "
        "Preserve important information, reasoning, and references. "
        "Stay within one page if possible.\n\n"
    )
    
    for entry in entries:
        query = entry.get("query", "")
        response = entry.get("response", "")
        summary_prompt += f"User: {query}\nAssistant: {response}\n\n"

    print(f"📄 Summarizing {len(entries)} entries...")

    try:
        summary_text = await call_llama(summary_prompt, model=model)
    except Exception as e:
        summary_text = f"Summary failed: {e}"

    return summary_text.strip(), len(entries)


async def background_summarize(session_id, model="llama3:8b"):
    try:
        session_history = load_session(session_id)

        # Only actual interactions (not summaries)
        interactions = [e for e in session_history if e.get("timestamp") != "SUMMARY"]
        if len(interactions) <= KEEP_INTERACTIONS + 5:
            return

        # Load existing summaries (raw)
        summary_file = os.path.join(SUMMARY_DIR, f"{session_id}_summaries.json")
        previous_summary_text = ""
        if os.path.exists(summary_file):
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    summaries = json.load(f)
                    previous_summary_text = "\n".join(s["summary"] for s in summaries)
            except json.JSONDecodeError:
                previous_summary_text = ""

        # Pick the 5 oldest real interactions
        entries_to_summarize = interactions[0:5]

        # Prepend previous summary into a fake interaction to pass into summarize_entries
        combined_entries = []
        if previous_summary_text.strip():
            combined_entries.append({
                "query": "Summary of earlier history",
                "response": previous_summary_text
            })
        combined_entries.extend(entries_to_summarize)

        # Create merged summary
        summary_text, n_entries = await summarize_entries(combined_entries, model=model)

        # Overwrite summary file with the new one
        await overwrite_summary(session_id, summary_text)

        # Remove summarized entries from interaction history
        new_interactions = interactions[5:]
        save_session(session_id, new_interactions)

        print(f"✅ Background summarization for session {session_id} finished. {n_entries} entries summarized.")

    finally:
        background_tasks.pop(session_id, None)

async def overwrite_summary(session_id, summary_text):
    """
    Overwrites the current summary and saves a timestamped backup copy.
    """
    from datetime import datetime
    summary_file = os.path.join(SUMMARY_DIR, f"{session_id}_summaries.json")
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    backup_file = os.path.join(SUMMARY_DIR, f"{session_id}_summaries_{timestamp}.json")

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summaries = [{
        "timestamp": current_time,
        "summary": summary_text
    }]

    # Overwrite main summary
    with open(summary_file, "w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=4)

    # Write timestamped backup
    with open(backup_file, "w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=4)

    print(f"💾 Saved summary and archived as {backup_file}")


async def save_summary(session_id, summary_text):
    """
    Speichert eine neue Zusammenfassung in summaries/{session_id}_summaries.json.
    """
    summary_file = os.path.join(SUMMARY_DIR, f"{session_id}_summaries.json")

    try:
        if os.path.exists(summary_file):
            with open(summary_file, "r", encoding="utf-8") as file:
                summaries = json.load(file)
        else:
            summaries = []
    except json.JSONDecodeError:
        summaries = []

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summaries.append({
        "timestamp": current_time,
        "summary": summary_text
    })

    with open(summary_file, "w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=4)
