"""
FastAPI server for NRL tryscorers chatbot.
Endpoints: GET /api/health, POST /api/chat, POST /api/chat/stream.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import tryscorers_chat as chat

load_dotenv()

app = FastAPI(
    title="NRL Tryscorers Chat API",
    description="Chat API for NRL try scoring stats (FTS, ATS, LTS, FTS2H, 2+).",
)

CORS_ORIGINS = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS.split(",") if o.strip()],
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
    """Preload data on startup."""
    try:
        chat.load_data()
    except FileNotFoundError as e:
        # allow server to start; first request will fail with clear error
        pass


@app.get("/api/health")
def health():
    return {"status": "ok"}


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
