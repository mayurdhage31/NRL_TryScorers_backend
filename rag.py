"""
RAG pipeline for NRL tryscorers using Gemini File Search Tool (fully managed RAG).
Requires GEMINI_API_KEY in environment.
"""
import json
import os
import tempfile
import time
from typing import Any

import pandas as pd

_gemini_client: Any = None

STORE_STATE_PATH = os.path.join(os.path.dirname(__file__), "data", "file_search_store.json")
GEMINI_MODEL = "gemini-2.5-flash"
STORE_DISPLAY_NAME = "nrl_tryscorers"


def _get_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not set")
        _gemini_client = genai.Client(api_key=key)
    return _gemini_client


def build_chunks():
    """Yield text chunks and metadata from tryscorers CSV for indexing."""
    import tryscorers_chat as chat
    chat.load_data()
    df_full = chat._df_full
    if df_full is None or df_full.empty:
        return
    stat_cols = [c for c in chat.STAT_COLUMNS if c in df_full.columns]
    for _, row in df_full.iterrows():
        player = row.get("Player", "")
        team = row.get("Team", "")
        season = row.get("season", "")
        games = row.get("Games played", 0)
        if not player or not season:
            continue
        parts = [f"{player} ({team}): season {int(season)}, Games played: {games}."]
        for col in stat_cols:
            val = row.get(col, 0)
            if pd.isna(val):
                val = 0
            try:
                val = int(val)
            except (ValueError, TypeError):
                pass
            parts.append(f" {col}: {val}")
        yield "".join(parts)


def _load_store_name() -> str | None:
    """Load the persisted File Search store name from disk."""
    if not os.path.exists(STORE_STATE_PATH):
        return None
    try:
        with open(STORE_STATE_PATH) as f:
            return json.load(f).get("store_name")
    except Exception:
        return None


def _save_store_name(store_name: str) -> None:
    os.makedirs(os.path.dirname(STORE_STATE_PATH), exist_ok=True)
    with open(STORE_STATE_PATH, "w") as f:
        json.dump({"store_name": store_name}, f)


def _store_exists(client, store_name: str) -> bool:
    """Return True if the store still exists in the Gemini API."""
    try:
        client.file_search_stores.get(name=store_name)
        return True
    except Exception:
        return False


def ensure_index() -> int:
    """Build or refresh the Gemini File Search store from tryscorers data. Idempotent."""
    client = _get_client()

    # Reuse existing store if it is still alive
    store_name = _load_store_name()
    if store_name and _store_exists(client, store_name):
        return 0  # Already indexed

    # Create a new store
    store = client.file_search_stores.create(
        config={"display_name": STORE_DISPLAY_NAME}
    )
    store_name = store.name

    # Build a single TXT file from all chunks
    chunks = list(build_chunks())
    if not chunks:
        return 0
    content = "\n\n".join(chunks)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        operation = client.file_search_stores.upload_to_file_search_store(
            file=tmp_path,
            file_search_store_name=store_name,
            config={"display_name": "nrl_tryscorers_stats"},
        )
        # Poll until upload/indexing is complete
        while not operation.done:
            time.sleep(2)
            operation = client.operations.get(operation)
    finally:
        os.unlink(tmp_path)

    _save_store_name(store_name)
    return len(chunks)


def _get_store_name() -> str:
    """Return a valid store name, building the index if necessary."""
    store_name = _load_store_name()
    client = _get_client()
    if store_name and _store_exists(client, store_name):
        return store_name
    ensure_index()
    store_name = _load_store_name()
    if not store_name:
        raise RuntimeError("Failed to create Gemini File Search store.")
    return store_name


def get_rag_response(message: str, history: list[dict]) -> str:
    """Retrieve relevant chunks via Gemini File Search, then generate an answer."""
    from google.genai import types

    client = _get_client()
    try:
        store_name = _get_store_name()
    except Exception as e:
        return f"RAG index unavailable: {e}"

    system_instruction = (
        "You are an NRL tryscoring stats assistant. Answer only from the provided context "
        "(NRL player tryscorer statistics). Be concise. If the context doesn't contain enough "
        "information, say so and suggest a more specific question."
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=message,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[
                    types.Tool(
                        file_search=types.FileSearch(
                            file_search_store_names=[store_name]
                        )
                    )
                ],
            ),
        )
        text = (response.text or "").strip()
        return text if text else "I couldn't generate an answer. Try rephrasing your question."
    except Exception as e:
        return f"Error generating response: {e}"


def is_rag_available() -> bool:
    """True if GEMINI_API_KEY is set and RAG can be used."""
    return bool(os.environ.get("GEMINI_API_KEY"))
