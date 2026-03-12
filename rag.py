"""
RAG pipeline for NRL tryscorers using Gemini File Search Tool (fully managed RAG).
Requires GEMINI_API_KEY in environment.

Primary store: nrl_tryscorers_minutesbands (Position + Minutes_Band dataset).
Legacy store: nrl_tryscorers (full CSV, kept for backward compatibility).
"""
import json
import os
import tempfile
import time
from typing import Any

import pandas as pd

_gemini_client: Any = None

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Tracks whether the primary (minutes-bands) index has been built this process.
# Set to True by the background indexing thread once it confirms the store exists.
_minutesbands_index_ready: bool = False

# Legacy store (full CSV, no position/minutes)
STORE_STATE_PATH = os.path.join(_DATA_DIR, "file_search_store.json")
STORE_DISPLAY_NAME = "nrl_tryscorers"

# Primary store (minutes-bands dataset + bookmaker prices)
MINUTESBANDS_CSV = os.path.join(_DATA_DIR, "NRL_tryscorers_2020_2026_by_position_minutesbands.csv")
MINUTESBANDS_STORE_STATE_PATH = os.path.join(_DATA_DIR, "file_search_store_minutesbands_v2.json")
MINUTESBANDS_STORE_DISPLAY_NAME = "nrl_tryscorers_v2"

GEMINI_MODEL = "gemini-2.5-flash"

SYSTEM_INSTRUCTION = """\
You are an expert NRL tryscoring stats assistant. Answer ONLY from the provided context.

DATASET CONTEXT:
- Stats are broken down by Player, Season (2020–2025), Position, and Minutes Band.
- Minutes bands: Over 20/30/40/50/60/70 mins (how long the player was on the field).
- Default band when not specified: Over 20 mins (broadest — includes all players who played 20+ mins).
- Stats: ATS (Anytime Try Scorer), FTS (First Try Scorer), LTS (Last Try Scorer),
  FTS2H (First Try 2nd Half), 2+ (two or more tries).
- Historical odds = Games played / hits (e.g. 22 GP, 7 ATS → $3.14).

BOOKMAKER PRICES:
- Current-round bookmaker prices are available for FTS, ATS, LTS, FTS2H, and 2+ markets.
- Bookmakers: Tab, Neds, Betright, Bet365, Sportsbet, Pointsbet, Dabble, Playup, Boombet, Bluebet, Unibet.
- The "Best price" field shows the highest (best) price across all bookmakers for that player/market.
- For questions about bookmaker prices, best prices, or which bookmaker to use, look for
  "Bookmaker prices this round" entries in the context.

RESPONSE FORMATTING RULES (always follow — never use "***" or "**"):
Single player stats:
  {Player} — {stat} stats ({filters})
  • {season} — GP {n} | {stat} {hits}/{gp} ({pct}%, ${odds})
  • Total — GP {n} | {stat} {hits}/{gp} ({pct}%, ${odds})

Bookmaker price questions:
  {Player} — {market} bookmaker prices this round:
  Best price: ${best_price} on {best_bookmaker}
  All bookmakers:
  • {Bookmaker}: ${price}

Value questions (3 bullets):
  {Player} — {stat} value check ({filters})
  • Stat: {hits}/{gp} games ({pct}% hist), hist odds ${hist_odds}
  • Market: ${market_odds} (implied {implied_pct}%) vs historical {hist_pct}% — edge {edge:+.1f} pp
  • Verdict: positive/negative historical value

Ranking lists:
  {stat} rankings ({filters})
  1) Player (Team, Pos, MinBand) — {stat} {hits}/{gp} ({pct}%), hist ${odds}
  ...
  • Filters: stat=..., seasons=..., position=..., minutes band=..., inactive since 2024 excluded

RULES:
- Never use "***", "__", or bold/italic markdown. Use plain text, bullets (•), and numbers.
- Format odds as "—" if 0 hits, else "$XX.XX".
- Always exclude players with 0 games since 2024 from ranked lists.
- If context is insufficient, say so briefly and suggest rephrasing.
"""


def _get_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not set")
        _gemini_client = genai.Client(api_key=key)
    return _gemini_client


# ---------------------------------------------------------------------------
# Chunk builders
# ---------------------------------------------------------------------------

def build_chunks():
    """Yield text chunks from the main full CSV (legacy store)."""
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


def build_chunks_summary_prices():
    """Yield text chunks of current-round bookmaker prices from all summary CSVs."""
    import tryscorers_chat as chat
    price_cols = sorted(chat._PRICE_COLUMNS)
    for market, filenames in chat._SUMMARY_CSV.items():
        for base in (chat._BASE_DIR, os.path.join(os.path.dirname(__file__), "..", "data")):
            path = os.path.join(base, filenames[0])
            if not os.path.isfile(path):
                continue
            try:
                df = pd.read_csv(path)
                df.columns = [str(c).strip() for c in df.columns]
                if "Player" not in df.columns:
                    break
                for _, row in df.iterrows():
                    player = str(row.get("Player", "")).strip()
                    if not player:
                        continue
                    parts = [f"Player: {player} | Market: {market} | Bookmaker prices this round:"]
                    for col in price_cols:
                        if col in df.columns:
                            v = row.get(col)
                            if pd.notna(v) and str(v).strip() not in ("", "NA"):
                                try:
                                    parts.append(f" | {col}: ${float(v):.2f}")
                                except (TypeError, ValueError):
                                    pass
                    highest = row.get("Highest")
                    if pd.notna(highest) and str(highest).strip() not in ("", "NA"):
                        try:
                            parts.append(f" | Best price: ${float(highest):.2f}")
                        except (TypeError, ValueError):
                            pass
                    if len(parts) > 1:
                        yield "".join(parts)
                break
            except Exception as exc:
                print(f"[RAG] Failed to build price chunks for {market}: {exc}")


def build_chunks_minutesbands():
    """Yield text chunks from the minutes-bands CSV (primary store)."""
    if not os.path.isfile(MINUTESBANDS_CSV):
        return
    df = pd.read_csv(MINUTESBANDS_CSV)
    df.columns = [c.strip() for c in df.columns]

    stat_pairs = [
        ("ATS", "ATS historical odds"),
        ("FTS", "FTS historical odds"),
        ("LTS", "LTS historical odds"),
        ("FTS2H", "FTS2H historical odds"),
        ("2+", "2+ historical odds"),
    ]

    for _, row in df.iterrows():
        player = str(row.get("Player", "")).strip()
        season = row.get("Season", "")
        position = str(row.get("Position", "")).strip()
        minutes_band = str(row.get("Minutes_Band", "")).strip()
        games = row.get("Games played", 0)

        if not player or not season:
            continue
        try:
            season = int(season)
        except (ValueError, TypeError):
            continue

        try:
            games = int(pd.to_numeric(games, errors="coerce") or 0)
        except (ValueError, TypeError):
            games = 0

        parts = [
            f"Player: {player} | Season: {season} | Position: {position} | "
            f"Minutes: {minutes_band} | GP: {games}"
        ]

        for stat_col, odds_col in stat_pairs:
            stat_val = row.get(stat_col, 0)
            odds_val = row.get(odds_col, "NA")
            try:
                stat_val = int(pd.to_numeric(stat_val, errors="coerce") or 0)
            except (ValueError, TypeError):
                stat_val = 0
            if str(odds_val) == "NA" or pd.isna(odds_val):
                odds_str = "NA"
            else:
                try:
                    odds_str = f"${float(odds_val):.2f}"
                except (ValueError, TypeError):
                    odds_str = "NA"
            parts.append(f" | {stat_col}: {stat_val} (hist odds: {odds_str})")

        yield "".join(parts)


# ---------------------------------------------------------------------------
# Store state persistence
# ---------------------------------------------------------------------------

def _load_store_name(state_path: str) -> str | None:
    if not os.path.exists(state_path):
        return None
    try:
        with open(state_path) as f:
            return json.load(f).get("store_name")
    except Exception:
        return None


def _save_store_name(state_path: str, store_name: str) -> None:
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    with open(state_path, "w") as f:
        json.dump({"store_name": store_name}, f)


def _store_exists(client, store_name: str) -> bool:
    try:
        client.file_search_stores.get(name=store_name)
        return True
    except Exception:
        return False


def _build_and_upload_store(client, display_name: str, chunks_iter, file_display_name: str, state_path: str) -> int:
    """Create a new file search store, upload chunks, save state. Returns chunk count."""
    store = client.file_search_stores.create(
        config={"display_name": display_name}
    )
    store_name = store.name

    chunks = list(chunks_iter)
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
            config={"display_name": file_display_name},
        )
        while not operation.done:
            time.sleep(2)
            operation = client.operations.get(operation)
    finally:
        os.unlink(tmp_path)

    _save_store_name(state_path, store_name)
    return len(chunks)


# ---------------------------------------------------------------------------
# Legacy store (full CSV)
# ---------------------------------------------------------------------------

def ensure_index() -> int:
    """Build or refresh the legacy Gemini File Search store from the full CSV. Idempotent."""
    client = _get_client()
    store_name = _load_store_name(STORE_STATE_PATH)
    if store_name and _store_exists(client, store_name):
        return 0
    return _build_and_upload_store(
        client, STORE_DISPLAY_NAME, build_chunks(), "nrl_tryscorers_stats", STORE_STATE_PATH
    )


def _get_legacy_store_name() -> str:
    store_name = _load_store_name(STORE_STATE_PATH)
    client = _get_client()
    if store_name and _store_exists(client, store_name):
        return store_name
    ensure_index()
    store_name = _load_store_name(STORE_STATE_PATH)
    if not store_name:
        raise RuntimeError("Failed to create legacy Gemini File Search store.")
    return store_name


# ---------------------------------------------------------------------------
# Primary store (minutes-bands dataset)
# ---------------------------------------------------------------------------

def _build_chunks_combined():
    """Yield all chunks: historical stats (minutes-bands) + bookmaker prices (summary CSVs)."""
    yield from build_chunks_minutesbands()
    yield from build_chunks_summary_prices()


def ensure_minutesbands_index() -> int:
    """Build or refresh the combined Gemini File Search store (stats + prices). Idempotent."""
    client = _get_client()
    store_name = _load_store_name(MINUTESBANDS_STORE_STATE_PATH)
    if store_name and _store_exists(client, store_name):
        return 0
    return _build_and_upload_store(
        client,
        MINUTESBANDS_STORE_DISPLAY_NAME,
        _build_chunks_combined(),
        "nrl_tryscorers_combined_stats",
        MINUTESBANDS_STORE_STATE_PATH,
    )


def _get_minutesbands_store_name() -> str:
    """
    Return a valid store name, checking the ready flag first to avoid blocking
    during the initial index build. Raises if the store is not ready yet so
    the caller can fall back to rule-based logic.
    """
    global _minutesbands_index_ready
    store_name = _load_store_name(MINUTESBANDS_STORE_STATE_PATH)
    client = _get_client()
    if store_name and _store_exists(client, store_name):
        _minutesbands_index_ready = True
        return store_name
    if not _minutesbands_index_ready:
        raise RuntimeError("Minutes-bands RAG index is still being built. Using rule-based fallback.")
    # Index was marked ready but state file is missing — rebuild (blocking, rare case)
    ensure_minutesbands_index()
    store_name = _load_store_name(MINUTESBANDS_STORE_STATE_PATH)
    if not store_name:
        raise RuntimeError("Failed to create minutes-bands Gemini File Search store.")
    return store_name


def build_minutesbands_index_background() -> None:
    """
    Build the minutes-bands index in the background (call from a thread).
    Sets _minutesbands_index_ready = True once the store is confirmed/built.
    Safe to call multiple times — idempotent via ensure_minutesbands_index().
    """
    global _minutesbands_index_ready
    try:
        ensure_minutesbands_index()
        _minutesbands_index_ready = True
        print("[RAG] Minutes-bands index is ready.")
    except Exception as exc:
        print(f"[RAG] Background index build failed: {exc}")


# ---------------------------------------------------------------------------
# Chat response — uses primary (minutes-bands) store
# ---------------------------------------------------------------------------

def get_rag_response(message: str, history: list[dict]) -> str:
    """
    Retrieve relevant chunks via Gemini File Search (minutes-bands store), then generate an answer.
    Raises on any error (store not ready, API failure) so callers can fall back to rule-based logic.
    """
    from google.genai import types

    client = _get_client()
    # Raises RuntimeError if index is not ready yet — caller should catch and fall back.
    store_name = _get_minutesbands_store_name()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=message,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
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


def is_rag_available() -> bool:
    """True if GEMINI_API_KEY is set and RAG can be used."""
    return bool(os.environ.get("GEMINI_API_KEY"))
