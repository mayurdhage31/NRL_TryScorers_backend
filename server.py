"""
FastAPI server for NRL tryscorers chatbot.
Endpoints: GET /api/health, POST /api/chat, POST /api/chat/stream,
           GET /api/players, GET /api/players/{player_id}/seasons.
"""
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import tryscorers_chat as chat

load_dotenv()

app = FastAPI(
    title="NRL Tryscorers Chat API",
    description="Chat API for NRL try scoring stats (FTS, ATS, LTS, FTS2H, 2+).",
)

_DEFAULT_CORS = "http://localhost:3000,http://127.0.0.1:3000"
_VERCEL_ORIGIN = "https://nrl-tryscorers-frontend.vercel.app"
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", _DEFAULT_CORS)
if CORS_ORIGINS.strip() == _DEFAULT_CORS:
    CORS_ORIGINS = f"{CORS_ORIGINS},{_VERCEL_ORIGIN}"
origins_list = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_chat_executor = ThreadPoolExecutor(max_workers=4)
CHAT_TIMEOUT = 180


class ChatMessage(BaseModel):
    role: str  # "user" | "model"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    response: str
    history: list[ChatMessage]


@app.on_event("startup")
def startup():
    """Preload data on startup, print dataset sanity check, and start RAG index build."""
    try:
        chat.load_data()
    except FileNotFoundError:
        pass

    # Dataset sanity check for the minutes-bands CSV
    import pandas as pd
    mb_path = chat._MINUTESBANDS_CSV
    if os.path.isfile(mb_path):
        try:
            df = pd.read_csv(mb_path)
            print(
                f"[STARTUP] minutes-bands CSV loaded: {len(df)} rows, "
                f"{df['Player'].nunique()} players, "
                f"columns={list(df.columns)}, "
                f"minutes bands={sorted(df['Minutes_Band'].unique())}"
            )
        except Exception as exc:
            print(f"[STARTUP] Failed to inspect minutes-bands CSV: {exc}")
    else:
        print(f"[STARTUP] WARNING: minutes-bands CSV not found at {mb_path}")

    # Build the RAG index in the background so the first chat request is not blocked.
    # The chatbot falls back to rule-based answers while the index is being built.
    if os.environ.get("GEMINI_API_KEY"):
        try:
            import rag as _rag
            # If the store state file already exists and is valid, mark ready immediately.
            _store_name = _rag._load_store_name(_rag.MINUTESBANDS_STORE_STATE_PATH)
            if _store_name:
                # Quick existence check (non-blocking: uses cached client if already made)
                try:
                    _client = _rag._get_client()
                    if _rag._store_exists(_client, _store_name):
                        _rag._minutesbands_index_ready = True
                        print("[STARTUP] Minutes-bands RAG store already exists — ready immediately.")
                        return
                except Exception:
                    pass
            # Store not ready — build it in the background
            print("[STARTUP] Starting minutes-bands RAG index build in background...")
            threading.Thread(
                target=_rag.build_minutesbands_index_background,
                daemon=True,
            ).start()
        except Exception as exc:
            print(f"[STARTUP] Could not start RAG index build: {exc}")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/players")
def list_players():
    """Return unique players (player_id, name) for dropdown."""
    try:
        chat.load_data()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return chat.get_unique_players()


@app.get("/api/players/{player_id}/positions")
def player_positions(player_id: int):
    """Return the distinct positions available for this player in the minutes-bands dataset."""
    try:
        chat.load_data()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    positions = chat.get_player_positions(player_id)
    if positions is None:
        raise HTTPException(status_code=404, detail="Player not found")
    return {"positions": positions}


@app.get("/api/players/{player_id}/seasons")
def player_seasons(
    player_id: int,
    minutes_band: Optional[list[str]] = Query(default=None, description="e.g. minutes_band=Over+20+mins&minutes_band=Over+30+mins (multiple allowed; default: Over 20 mins)"),
    positions: Optional[list[str]] = Query(default=None, description="Position filters, e.g. positions=Prop&positions=Lock. Omit for all."),
    seasons: Optional[list[int]] = Query(default=None, description="Season years to include, e.g. seasons=2023&seasons=2024"),
):
    """
    Return per-season stats for the given player from the minutes-bands dataset.
    Includes a Total row aggregated across selected seasons and positions.
    """
    try:
        chat.load_data()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    rows = chat.get_player_season_stats_minutesbands(
        player_id=player_id,
        minutes_bands=list(minutes_band) if minutes_band else None,
        positions=list(positions) if positions else None,
        seasons=list(seasons) if seasons else None,
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Player not found or no data for given filters")
    return rows


@app.post("/api/chat", response_model=ChatResponse)
def post_chat(req: ChatRequest):
    """Non-streaming chat."""
    try:
        chat.load_data()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    history = [{"role": m.role, "content": m.content} for m in req.history]
    try:
        response_text = chat.get_chat_response(req.message, history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    new_history = history + [
        {"role": "user", "content": req.message},
        {"role": "model", "content": response_text},
    ]
    return ChatResponse(
        response=response_text,
        history=[ChatMessage(role=m["role"], content=m["content"]) for m in new_history],
    )


def _stream_generator(message: str, history: list[dict]):
    try:
        chat.load_data()
    except FileNotFoundError as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        return
    try:
        full_parts = []
        for chunk in chat.stream_chat_response(message, history):
            full_parts.append(chunk)
            yield f"data: {json.dumps({'token': chunk})}\n\n"
        full = "".join(full_parts)
        new_history = history + [
            {"role": "user", "content": message},
            {"role": "model", "content": full},
        ]
        yield f"data: {json.dumps({'done': True, 'history': new_history})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@app.post("/api/chat/stream")
def post_chat_stream(req: ChatRequest):
    """Streaming chat via SSE."""
    history = [{"role": m.role, "content": m.content} for m in req.history]

    def gen():
        for s in _stream_generator(req.message, history):
            yield s

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
