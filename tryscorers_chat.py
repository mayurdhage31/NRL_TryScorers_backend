"""
NRL tryscorers chatbot: deterministic stats, rankings, value analysis.
All calculations in code; LLM optional for phrasing only.
Primary data source: NRL_tryscorers_2020_2026_by_position_minutesbands.csv
"""
import os
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Data paths and cache
# ---------------------------------------------------------------------------
_data_dir = os.path.join(os.path.dirname(__file__), "data")
if os.environ.get("DATA_DIR"):
    _BASE_DIR = os.environ.get("DATA_DIR", _data_dir)
else:
    _full_path = os.path.join(_data_dir, "Nrl_tryscorers_2020_2025_full.csv")
    _repo_data = os.path.join(os.path.dirname(__file__), "..", "data", "Nrl_tryscorers_2020_2025_full.csv")
    _BASE_DIR = _data_dir if os.path.isfile(_full_path) else (os.path.dirname(_repo_data) if os.path.isfile(_repo_data) else _data_dir)

_FULL_CSV = os.path.join(_BASE_DIR, "Nrl_tryscorers_2020_2025_full.csv")
_PLAYERS_CSV = os.path.join(_BASE_DIR, "NRL Players and Teams.csv")
_LIVE_PRICES_CSV = os.path.join(_BASE_DIR, "live_prices.csv")
_MINUTESBANDS_CSV = os.path.join(_BASE_DIR, "NRL_tryscorers_2020_2026_by_position_minutesbands.csv")

# Summary CSVs (market-specific): best prices only from these, never fabricate
_REPO_DATA = os.path.join(os.path.dirname(__file__), "..", "data")
_SUMMARY_CSV = {
    "FTS": ("fts_summary.csv",),
    "LTS": ("lts_summary.csv",),
    "ATS": ("ats_summary.csv",),
    "FTS2H": ("fts2h_summary.csv",),
    "2+": ("tpt_summary.csv",),
}
# Bookmaker columns in summary CSVs (price columns only)
_PRICE_COLUMNS = {"Tab", "Neds", "Betright", "Bet365", "Sportsbet", "Pointsbet", "Dabble", "Playup", "Boombet", "Bluebet", "Unibet", "Topsport"}

_df_full: pd.DataFrame | None = None
_df_players: pd.DataFrame | None = None
_df_live_prices: pd.DataFrame | None = None
_df_minutesbands: pd.DataFrame | None = None
_available_seasons: list[int] = []

# Stat column name for "2+" in CSV
STAT_2PLUS = "2+"
STAT_COLUMNS = ["FTS", "ATS", "LTS", "FTS2H", STAT_2PLUS]

# Minutes band constants (broadest first, excluding "Less than 20 mins" from default ordered list)
MINUTES_BANDS_ORDERED = ["Less than 20 mins", "Over 20 mins", "Over 30 mins", "Over 40 mins", "Over 50 mins", "Over 60 mins", "Over 70 mins"]
DEFAULT_MINUTES_BAND = "Over 20 mins"

MINUTES_BAND_ALIASES: dict[str, str] = {
    "less than 20": "Less than 20 mins",
    "less than 20 mins": "Less than 20 mins",
    "under 20": "Less than 20 mins",
    "under 20 mins": "Less than 20 mins",
    "sub 20": "Less than 20 mins",
    "over 20": "Over 20 mins",
    "over 20 mins": "Over 20 mins",
    "20 mins": "Over 20 mins",
    "20+": "Over 20 mins",
    "over 30": "Over 30 mins",
    "over 30 mins": "Over 30 mins",
    "30 mins": "Over 30 mins",
    "30+": "Over 30 mins",
    "over 40": "Over 40 mins",
    "over 40 mins": "Over 40 mins",
    "40 mins": "Over 40 mins",
    "40+": "Over 40 mins",
    "over 50": "Over 50 mins",
    "over 50 mins": "Over 50 mins",
    "50 mins": "Over 50 mins",
    "50+": "Over 50 mins",
    "over 60": "Over 60 mins",
    "over 60 mins": "Over 60 mins",
    "60 mins": "Over 60 mins",
    "60+": "Over 60 mins",
    "over 70": "Over 70 mins",
    "over 70 mins": "Over 70 mins",
    "70 mins": "Over 70 mins",
    "70+": "Over 70 mins",
}

# Position alias -> dataset value
POSITION_ALIASES = {
    "fullback": "Fullback",
    "fullbacks": "Fullback",
    "winger": "Winger",
    "wingers": "Winger",
    "wing": "Winger",
    "wings": "Winger",
    "centre": "Centre",
    "centres": "Centre",
    "center": "Centre",
    "centers": "Centre",
    "halfback": "Halfback",
    "halfbacks": "Halfback",
    "hb": "Halfback",
    "five-eighth": "Five-Eighth",
    "five eighth": "Five-Eighth",
    "five-eighths": "Five-Eighth",
    "five eighths": "Five-Eighth",
    "6": "Five-Eighth",
    "hooker": "Hooker",
    "hookers": "Hooker",
    "prop": "Prop",
    "props": "Prop",
    "front row": "Prop",
    "front-row": "Prop",
    "front rower": "Prop",
    "front rowers": "Prop",
    "second row": "2nd Row",
    "second-row": "2nd Row",
    "2nd row": "2nd Row",
    "edge forward": "2nd Row",
    "edge forwards": "2nd Row",
    "back row": "2nd Row",
    "back-row": "2nd Row",
    "lock": "Lock",
    "locks": "Lock",
    "loose forward": "Lock",
    "interchange": "Interchange",
    "bench": "Interchange",
}
# Combined groups
POSITION_GROUPS = {
    "outside backs": ["Fullback", "Winger", "Centre"],
    "halves": ["Halfback", "Five-Eighth"],
    "spine": ["Fullback", "Halfback", "Five-Eighth", "Hooker"],
    "middles": ["Prop", "Hooker", "Lock"],
    "edge forwards": ["2nd Row"],
}

# Stat type aliases
STAT_ALIASES = {
    "fts": "FTS",
    "first try scorer": "FTS",
    "first tryscorer": "FTS",
    "score first": "FTS",
    "first scorer": "FTS",
    "to score first": "FTS",
    "score the first": "FTS",
    "first try": "FTS",
    "ats": "ATS",
    "anytime try scorer": "ATS",
    "anytime tryscorer": "ATS",
    "anytime scorer": "ATS",
    "score a try": "ATS",
    "lts": "LTS",
    "last try scorer": "LTS",
    "last tryscorer": "LTS",
    "score last": "LTS",
    "last scorer": "LTS",
    "to score last": "LTS",
    "fts2h": "FTS2H",
    "first try scorer 2nd half": "FTS2H",
    "first try scorer second half": "FTS2H",
    "first try in second half": "FTS2H",
    "first 2nd half tryscorer": "FTS2H",
    "first try 2nd half": "FTS2H",
    "2+": STAT_2PLUS,
    "2+ tries": STAT_2PLUS,
    "two or more tries": STAT_2PLUS,
    "2 or more tries": STAT_2PLUS,
    "multi try": STAT_2PLUS,
    "multi-try": STAT_2PLUS,
}

# Known player nicknames that fuzzy/partial matching can't reliably resolve.
# Keys are lowercase; values are exact player names as they appear in the dataset.
NICKNAME_MAP: dict[str, str] = {
    "luki": "Luciano Leilua",
    "turbo": "Tom Trbojevic",
    "haasy": "Payne Haas",
    "teddy": "James Tedesco",
    "jimmy": "James Tedesco",
}

# Conversational prefixes to strip before player/stat extraction so that
# phrases like "I want to bet on Haas FTS" resolve the player name cleanly.
_INTENT_PREFIXES = [
    r"i want to bet on\s+",
    r"i wanna bet on\s+",
    r"wheres?\s+(the\s+)?best\s+price\s+(for\s+)?",
    r"where\s+is\s+(the\s+)?best\s+price\s+(for\s+)?",
    r"should\s+i\s+(back|bet\s+on)\s+",
    r"can\s+i\s+get\s+(a\s+)?price\s+(for\s+)?",
    r"what\s+(are\s+the\s+)?odds\s+(for\s+)?",
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """Load full tryscorers CSV and players/teams CSV; return (df_full, df_players, seasons)."""
    global _df_full, _df_players, _df_live_prices, _available_seasons, _df_minutesbands
    if _df_full is not None:
        return _df_full, _df_players if _df_players is not None else pd.DataFrame(), _available_seasons

    if not os.path.isfile(_FULL_CSV):
        raise FileNotFoundError(f"Tryscorers data not found: {_FULL_CSV}")

    _df_full = pd.read_csv(_FULL_CSV)
    for col in ["season", "Games played"] + STAT_COLUMNS:
        if col in _df_full.columns:
            _df_full[col] = pd.to_numeric(_df_full[col], errors="coerce").fillna(0).astype(int)

    _available_seasons = sorted(_df_full["season"].dropna().unique().astype(int).tolist())

    if os.path.isfile(_PLAYERS_CSV):
        _df_players = pd.read_csv(_PLAYERS_CSV)
        if _df_players.columns[0].strip() == "" or _df_players.columns[0].startswith("Unnamed"):
            _df_players = _df_players.iloc[:, 1:]
        _df_players.columns = [c.strip() for c in _df_players.columns]
        if "Player" not in _df_players.columns:
            _df_players = pd.DataFrame()
    else:
        _df_players = pd.DataFrame()

    if os.path.isfile(_LIVE_PRICES_CSV):
        try:
            _df_live_prices = pd.read_csv(_LIVE_PRICES_CSV)
            _df_live_prices.columns = [c.strip() for c in _df_live_prices.columns]
            if "Price" in _df_live_prices.columns:
                _df_live_prices["Price"] = pd.to_numeric(_df_live_prices["Price"], errors="coerce")
            else:
                _df_live_prices = pd.DataFrame()
        except Exception:
            _df_live_prices = pd.DataFrame()
    else:
        _df_live_prices = pd.DataFrame()

    # Load minutes-bands CSV (primary for chatbot + player stats)
    _df_minutesbands = _load_minutesbands_csv()

    return _df_full, _df_players, _available_seasons


def _load_minutesbands_csv() -> pd.DataFrame:
    """Load the minutes-bands CSV into a clean DataFrame."""
    if not os.path.isfile(_MINUTESBANDS_CSV):
        print(f"[WARN] minutes-bands CSV not found: {_MINUTESBANDS_CSV}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(_MINUTESBANDS_CSV)
        df.columns = [c.strip() for c in df.columns]
        # Normalise Season to int
        if "Season" in df.columns:
            df["Season"] = pd.to_numeric(df["Season"], errors="coerce").fillna(0).astype(int)
        if "Games played" in df.columns:
            df["Games played"] = pd.to_numeric(df["Games played"], errors="coerce").fillna(0).astype(int)
        if "Total Games played" in df.columns:
            df["Total Games played"] = pd.to_numeric(df["Total Games played"], errors="coerce").fillna(0).astype(int)
        for stat in STAT_COLUMNS:
            if stat in df.columns:
                df[stat] = pd.to_numeric(df[stat], errors="coerce").fillna(0).astype(int)
        return df
    except Exception as exc:
        print(f"[ERROR] Failed to load minutes-bands CSV: {exc}")
        return pd.DataFrame()


def _get_minutesbands() -> pd.DataFrame:
    """Return cached minutes-bands DataFrame (loads if needed)."""
    global _df_minutesbands
    if _df_minutesbands is None:
        _df_minutesbands = _load_minutesbands_csv()
    return _df_minutesbands


def normalize_text(s: str) -> str:
    """Lowercase, collapse whitespace."""
    if not s or not isinstance(s, str):
        return ""
    return " ".join(s.lower().split())


# ---------------------------------------------------------------------------
# Parsed query intent
# ---------------------------------------------------------------------------
@dataclass
class ParsedQuery:
    player_names: list[str] = field(default_factory=list)
    ambiguous_name_map: dict[str, list[str]] = field(default_factory=dict)
    stat_type: str | None = None
    season_from: int | None = None
    season_to: int | None = None
    top_n: int | None = None
    min_games: int | None = None
    min_games_since_2024: int | None = None
    min_pct: float | None = None
    positions: list[str] = field(default_factory=list)
    minutes_band: str | None = None
    market_odds: float | None = None
    best_price_request: bool = False
    bet_analysis_request: bool = False
    round_value_request: bool = False
    raw_message: str = ""


def _strip_intent_prefixes(text: str) -> str:
    """Remove conversational leading phrases so player/stat parsing is cleaner."""
    t = text
    for pattern in _INTENT_PREFIXES:
        t = re.sub(r"(?i)^" + pattern, "", t).strip()
    return t


def parse_query(message: str) -> ParsedQuery:
    """Extract intent from user message (rule-based + regex)."""
    load_data()
    # norm is used for intent detection (timeframe, stat type, best_price flags, etc.)
    # and must be based on the original message so keywords like "best price" are preserved.
    norm = normalize_text(message)
    # clean_message strips conversational prefixes so player names embedded in
    # phrases like "I want to bet on Haas FTS" can be extracted cleanly.
    clean_message = _strip_intent_prefixes(message)
    pq = ParsedQuery(raw_message=message)

    # Timeframe
    since = re.search(r"since\s+(\d{4})", norm)
    if since:
        y = int(since.group(1))
        pq.season_from = y
        pq.season_to = max(_available_seasons) if _available_seasons else 2025

    in_year = re.search(r"\bin\s+(\d{4})\b", norm)
    if in_year:
        y = int(in_year.group(1))
        pq.season_from = pq.season_to = y

    from_to = re.search(r"from\s+(\d{4})\s+to\s+(\d{4})", norm)
    if from_to:
        pq.season_from = int(from_to.group(1))
        pq.season_to = int(from_to.group(2))

    between = re.search(r"between\s+(\d{4})\s+and\s+(\d{4})", norm)
    if between:
        pq.season_from = int(between.group(1))
        pq.season_to = int(between.group(2))

    if "this year" in norm and _available_seasons:
        pq.season_from = pq.season_to = max(_available_seasons)

    for m in re.finditer(r"last\s+(\d+)\s+years?", norm):
        n = int(m.group(1))
        if _available_seasons:
            years = sorted(_available_seasons)[-n:]
            if years:
                pq.season_from, pq.season_to = min(years), max(years)
        break

    # Stat type
    pq.stat_type = resolve_stat_type(norm)

    # Top N / best / worst
    top = re.search(r"top\s+(\d+)", norm)
    if top:
        pq.top_n = int(top.group(1))
    if re.search(r"\b(best|highest|top)\b", norm) and pq.top_n is None:
        pq.top_n = 5
    if re.search(r"\b(worst|lowest)\b", norm):
        pq.top_n = pq.top_n or 5

    # Games filters
    min_g = re.search(r"minimum\s+(\d+)\s+games?", norm)
    if min_g:
        pq.min_games = int(min_g.group(1))
    min_g2 = re.search(r"at\s+least\s+(\d+)\s+games?", norm)
    if min_g2:
        pq.min_games = int(min_g2.group(1))
    if "minimum 1 game played since 2024" in norm or "min 1 game since 2024" in norm or "minimum 1 game since 2024" in norm:
        pq.min_games_since_2024 = 1
    since_2024 = re.search(r"min(?:imum)?\s+(\d+)\s+game[s]?\s+since\s+2024", norm)
    if since_2024:
        pq.min_games_since_2024 = int(since_2024.group(1))

    better = re.search(r"better\s+than\s+1\s+in\s+(\d+)", norm)
    if better:
        n = int(better.group(1))
        pq.min_pct = 100.0 / n if n else None

    # Position
    pq.positions = resolve_positions(norm)

    # Minutes band
    pq.minutes_band = resolve_minutes_band(norm)

    # Market odds
    dollar = re.search(r"\$\s*(\d+(?:\.\d+)?)", message)
    if dollar:
        pq.market_odds = float(dollar.group(1))
    else:
        price_s = re.search(r"\b(\d+(?:\.\d+)?)\s*s\b", norm)
        if price_s:
            pq.market_odds = float(price_s.group(1))
        else:
            num = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:for|ats?|fts?|odds?)", norm)
            if num:
                pq.market_odds = float(num.group(1))
            else:
                odds = re.search(r"odds?\s+of\s+(\d+(?:\.\d+)?)", norm)
                if odds:
                    pq.market_odds = float(odds.group(1))

    # Player names (also captures ambiguous partial-name matches)
    # Use the prefix-stripped clean_message for better player name extraction
    norm_clean = normalize_text(clean_message)
    pq.player_names, pq.ambiguous_name_map = resolve_player_names(clean_message, norm_clean)

    # Best available price — broad detection
    if re.search(r"best\s+(available\s+)?price", norm):
        pq.best_price_request = True
    if re.search(r"price[s]?\s+for\s+.+\s+(fts|ats|lts|fts2h|2\+)", norm):
        pq.best_price_request = True
    if re.search(r"\b(bookmaker|bookie|bookies|which\s+book)\b", norm):
        pq.best_price_request = True
    if re.search(r"(what|best|available|current)\s+(is\s+)?(the\s+)?(best\s+)?(price|odds)", norm):
        pq.best_price_request = True
    # Any message with a player name + price/odds keyword is a price request
    if pq.player_names and re.search(r"\b(price|odds|paying|payout)\b", norm):
        pq.best_price_request = True

    # Combined bet analysis — player + price intent + value/quality language
    _good_bet_pattern = r"\b(good\s+bet|worth\s+(a\s+)?(bet|punt|backing?)|is\s+it\s+(good|worth|value)|should\s+i\s+bet|worth\s+betting)\b"
    if re.search(_good_bet_pattern, norm):
        pq.bet_analysis_request = True
    # "is it value?" + best_price_request together implies combined analysis
    if pq.best_price_request and re.search(r"\b(value|worth|good)\b", norm):
        pq.bet_analysis_request = True

    # Round value bets intent — no specific player required.
    # Match patterns like "best value ATS bets this round", "value bets", "who's value this round"
    if re.search(r"\b(value\s+\w*\s*bets?|best\s+value\s+\w*\s*bets?|who.?s\s+(the\s+)?(best\s+)?value|value\s+picks?)\b", norm):
        pq.round_value_request = True
    if re.search(r"\bvalue\s+bets?\b", norm):
        pq.round_value_request = True
    if re.search(r"\bwho\s+is\s+(the\s+)?(best\s+)?value\b", norm):
        pq.round_value_request = True

    return pq


def resolve_minutes_band(norm_text: str) -> str | None:
    """Extract minutes band from text, e.g. 'over 40 mins' -> 'Over 40 mins'."""
    for alias, band in MINUTES_BAND_ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", norm_text):
            return band
    return None


def resolve_timeframe(pq: ParsedQuery) -> tuple[int, int]:
    """Return (season_from, season_to) inclusive. Default all available."""
    load_data()
    lo = min(_available_seasons) if _available_seasons else 2020
    hi = max(_available_seasons) if _available_seasons else 2025
    if pq.season_from is not None:
        lo = max(lo, pq.season_from)
    if pq.season_to is not None:
        hi = min(hi, pq.season_to)
    return lo, hi


def resolve_stat_type(norm_text: str) -> str | None:
    """Map natural language to FTS/ATS/LTS/FTS2H/2+."""
    norm = " " + norm_text + " "
    for alias, stat in STAT_ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", norm):
            return stat
    return None


def resolve_positions(norm_text: str) -> list[str]:
    """Return list of Position values."""
    out = set()
    for alias, pos in POSITION_ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", norm_text):
            out.add(pos)
    for group_name, positions in POSITION_GROUPS.items():
        if re.search(r"\b" + re.escape(group_name.replace(" ", r"\s+")) + r"\b", norm_text):
            out.update(positions)
    return list(out)


def resolve_player_names(message: str, norm_text: str) -> tuple[list[str], dict[str, list[str]]]:
    """Match player names from message against dataset.

    Returns (resolved, ambiguous_map):
      resolved      — unambiguous full player names.
      ambiguous_map — {partial_word: [candidate_full_names]} for words that matched
                      multiple players so the caller can ask for disambiguation.

    Resolution passes (in order):
      0. Nickname map               — exact lookup in NICKNAME_MAP; silent, highest priority.
      1. Full-name substring match  — silent, takes priority.
      2. Exact last-name match      — silent if unique, disambiguate if multiple.
      3. Exact first-name match     — silent if unique, disambiguate if multiple.
      4. Fuzzy (typo) match         — difflib against all first/last names; silent if
                                      one clear winner, disambiguate if several.
    """
    import difflib

    df_full, df_players, _ = load_data()
    all_names: set[str] = set()
    if "Player" in df_full.columns:
        all_names.update(df_full["Player"].dropna().astype(str).unique())
    if not df_players.empty and "Player" in df_players.columns:
        all_names.update(df_players["Player"].dropna().astype(str).unique())

    _SKIP_WORDS = {
        "the", "for", "and", "with", "from", "this", "that", "what", "who", "how",
        "best", "last", "first", "next", "over", "under", "round", "game", "games",
        "try", "tries", "score", "scorer", "price", "odds", "rate", "stats", "stat",
        "season", "seasons", "year", "years", "team", "position", "player", "players",
        "bookmaker", "bookmakers", "available", "current", "historical", "total",
        "fts", "ats", "lts", "fts2h", "2plus", "more", "half", "match", "week",
        "give", "show", "tell", "record", "records", "since", "into", "when", "been",
        "have", "does", "data", "info", "check", "look", "find",
        # Bet/value query words that fuzzy-match player names
        "value", "values", "bet", "bets", "betting", "back", "backing",
        "good", "great", "want", "wanna", "should", "worth", "punt",
        "pick", "picks", "play", "plays", "tip", "tips",
        "where", "wheres", "which", "there", "their", "about",
    }

    # ------------------------------------------------------------------
    # Pass 0: nickname map — highest priority, checked before all other passes
    # ------------------------------------------------------------------
    nickname_result: list[str] = []
    words_in_msg = re.findall(r'\b[A-Za-z]{3,}\b', message)
    for word in words_in_msg:
        resolved_nick = NICKNAME_MAP.get(word.lower())
        if resolved_nick and resolved_nick in all_names and resolved_nick not in nickname_result:
            nickname_result.append(resolved_nick)
    if nickname_result:
        return nickname_result[:10], {}

    # ------------------------------------------------------------------
    # Pass 1: full-name substring match (case-insensitive)
    # ------------------------------------------------------------------
    candidates: list[tuple[str, int]] = []
    for name in all_names:
        name_clean = name.strip()
        if len(name_clean) < 3:
            continue
        if re.search(re.escape(name_clean), message, re.I):
            candidates.append((name_clean, len(name_clean)))
    candidates.sort(key=lambda x: -x[1])
    seen: set[str] = set()
    result: list[str] = []
    for name, _ in candidates:
        if name not in seen and not any(name in s and name != s for s in seen):
            seen.add(name)
            result.append(name)

    if result:
        return result[:10], {}

    # ------------------------------------------------------------------
    # Build first/last name maps for partial matching
    # ------------------------------------------------------------------
    last_name_map: dict[str, list[str]] = {}
    first_name_map: dict[str, list[str]] = {}
    for name in all_names:
        parts = name.strip().split()
        if len(parts) >= 2:
            last_name_map.setdefault(parts[-1].lower(), []).append(name.strip())
            first_name_map.setdefault(parts[0].lower(), []).append(name.strip())

    all_last_names = list(last_name_map.keys())
    all_first_names = list(first_name_map.keys())

    ambiguous_map: dict[str, list[str]] = {}
    words = re.findall(r'\b[A-Za-z]{3,}\b', message)

    for word in words:
        wl = word.lower()
        if wl in _SKIP_WORDS:
            continue

        # Pass 2: exact last-name
        last_matches = last_name_map.get(wl, [])
        if len(last_matches) == 1:
            if last_matches[0] not in result:
                result.append(last_matches[0])
            continue
        if len(last_matches) > 1:
            ambiguous_map[word] = sorted(set(last_matches))[:10]
            continue

        # Pass 3: exact first-name
        first_matches = first_name_map.get(wl, [])
        if len(first_matches) == 1:
            if first_matches[0] not in result:
                result.append(first_matches[0])
            continue
        if len(first_matches) > 1:
            ambiguous_map[word] = sorted(set(first_matches))[:10]
            continue

        # Pass 4: fuzzy typo correction (only when no exact match at all)
        fuzzy_candidates: list[str] = []
        for fl in difflib.get_close_matches(wl, all_last_names, n=5, cutoff=0.75):
            fuzzy_candidates.extend(last_name_map[fl])
        for ff in difflib.get_close_matches(wl, all_first_names, n=5, cutoff=0.75):
            fuzzy_candidates.extend(first_name_map[ff])
        # Deduplicate while preserving order
        seen_fuzzy: set[str] = set()
        deduped: list[str] = []
        for fc in fuzzy_candidates:
            if fc not in seen_fuzzy:
                seen_fuzzy.add(fc)
                deduped.append(fc)

        if len(deduped) == 1:
            if deduped[0] not in result:
                result.append(deduped[0])
        elif len(deduped) > 1:
            ambiguous_map[word] = deduped[:10]

    # If we resolved at least one name via partial/fuzzy matching, suppress the
    # ambiguous map — the resolved name(s) are sufficient to answer.
    if result:
        return result[:10], {}

    return [], ambiguous_map


def resolve_top_n(pq: ParsedQuery) -> int:
    return pq.top_n if pq.top_n is not None else 5


def resolve_games_filters(pq: ParsedQuery) -> tuple[int | None, int | None, float | None]:
    return pq.min_games, pq.min_games_since_2024, pq.min_pct


def _get_season_mask(df: pd.DataFrame, season_from: int, season_to: int) -> pd.Series:
    return (df["season"] >= season_from) & (df["season"] <= season_to)


def _stat_col(stat: str) -> str:
    return stat if stat in ["FTS", "ATS", "LTS", "FTS2H", STAT_2PLUS] else "ATS"


def compute_player_stats(
    player_id: int,
    stat_type: str,
    season_from: int,
    season_to: int,
) -> dict[str, Any]:
    """Aggregate by player_id in range from the full CSV."""
    df_full, _, _ = load_data()
    mask = (df_full["player_id"] == player_id) & _get_season_mask(df_full, season_from, season_to)
    sub = df_full.loc[mask]
    total_games = int(sub["Games played"].sum())
    col = _stat_col(stat_type)
    stat_hits = int(sub[col].sum())
    pct = (stat_hits / total_games * 100) if total_games else 0.0
    hist_odds = (total_games / stat_hits) if stat_hits else None
    return {
        "player_id": player_id,
        "total_games": total_games,
        "stat_hits": stat_hits,
        "stat_type": stat_type,
        "pct": round(pct, 2),
        "hist_odds": round(hist_odds, 2) if hist_odds is not None else None,
    }


def _current_meta(player_name: str) -> tuple[str | None, str | None]:
    """(team, position) from NRL Players and Teams."""
    _, df_players, _ = load_data()
    if df_players.empty or "Player" not in df_players.columns:
        return None, None
    row = df_players[df_players["Player"].astype(str).str.strip() == player_name.strip()]
    if row.empty:
        return None, None
    r = row.iloc[0]
    team = r.get("Team")
    pos = r.get("Position")
    return (str(team) if pd.notna(team) else None), (str(pos) if pd.notna(pos) else None)


def _recent_games_since_2024(df_full: pd.DataFrame, player_id: int) -> int:
    return int(df_full[(df_full["player_id"] == player_id) & (df_full["season"] >= 2024)]["Games played"].sum())


def get_unique_players() -> list[dict[str, Any]]:
    """Return unique players (player_id, name) sorted by name, excluding those with 0 games since 2024."""
    df_full, _, _ = load_data()
    if df_full.empty or "Player" not in df_full.columns or "player_id" not in df_full.columns:
        return []
    recent = (
        df_full[df_full["season"] >= 2024]
        .groupby("player_id")["Games played"]
        .sum()
    )
    recent_games = recent.to_dict()

    players = (
        df_full.groupby("player_id", as_index=False)
        .agg({"Player": "first"})
        .sort_values("Player")
    )
    return [
        {"player_id": int(r["player_id"]), "name": str(r["Player"]).strip()}
        for _, r in players.iterrows()
        if pd.notna(r["Player"])
        and str(r["Player"]).strip()
        and int(recent_games.get(int(r["player_id"]), 0)) >= 1
    ]


def get_player_season_stats(player_id: int) -> list[dict[str, Any]]:
    """Return per-season stats for one player from the full CSV (legacy path)."""
    df_full, _, _ = load_data()
    sub = df_full[df_full["player_id"] == player_id].copy()
    if sub.empty:
        return []
    odds_cols = [
        "FTS historical odds", "ATS historical odds", "LTS historical odds",
        "FTS2H historical odds", "2+ historical odds",
    ]
    for c in odds_cols:
        if c in sub.columns:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")

    out = []
    for _, row in sub.sort_values("season").iterrows():
        def _odds(val: Any) -> float | None:
            if pd.isna(val) or val == "NA" or val == "":
                return None
            try:
                return float(val)
            except (TypeError, ValueError):
                return None

        out.append({
            "season": int(row["season"]),
            "games_played": int(row["Games played"]) if pd.notna(row["Games played"]) else 0,
            "fts": int(row["FTS"]) if "FTS" in row and pd.notna(row["FTS"]) else 0,
            "fts_historical_odds": _odds(row.get("FTS historical odds")),
            "ats": int(row["ATS"]) if "ATS" in row and pd.notna(row["ATS"]) else 0,
            "ats_historical_odds": _odds(row.get("ATS historical odds")),
            "lts": int(row["LTS"]) if "LTS" in row and pd.notna(row["LTS"]) else 0,
            "lts_historical_odds": _odds(row.get("LTS historical odds")),
            "fts2h": int(row["FTS2H"]) if "FTS2H" in row and pd.notna(row["FTS2H"]) else 0,
            "fts2h_historical_odds": _odds(row.get("FTS2H historical odds")),
            "two_plus": int(row[STAT_2PLUS]) if STAT_2PLUS in row and pd.notna(row[STAT_2PLUS]) else 0,
            "two_plus_historical_odds": _odds(row.get("2+ historical odds")),
        })
    return out


def get_player_positions(player_id: int) -> list[str] | None:
    """
    Return sorted list of distinct positions for the given player in the
    minutes-bands CSV (Over 20 mins band, all seasons).
    Returns None if player_id is not found; returns [] if no minutesbands data.
    """
    df_full, _, _ = load_data()
    mb_df = _get_minutesbands()

    match = df_full[df_full["player_id"] == player_id]
    if match.empty:
        return None
    player_name = str(match.iloc[0]["Player"]).strip()

    if mb_df.empty or "Player" not in mb_df.columns:
        return []

    sub = mb_df[
        (mb_df["Player"].astype(str).str.strip() == player_name)
        & (mb_df["Minutes_Band"] == DEFAULT_MINUTES_BAND)
    ]
    positions = sorted(sub["Position"].dropna().astype(str).str.strip().unique().tolist())
    return positions


def get_player_season_stats_minutesbands(
    player_id: int,
    minutes_bands: list[str] | None = None,
    positions: list[str] | None = None,
    seasons: list[int] | None = None,
) -> list[dict[str, Any]]:
    """
    Return per-season stats from the minutes-bands CSV, plus a Total row.

    Looks up player name via player_id from the full CSV, then filters
    the minutes-bands CSV by name, minutes_bands, positions, and seasons.
    When multiple bands or positions are provided, stats are aggregated across all of them.
    """
    df_full, _, _ = load_data()
    mb_df = _get_minutesbands()

    # Look up player name
    match = df_full[df_full["player_id"] == player_id]
    if match.empty:
        return []
    player_name = str(match.iloc[0]["Player"]).strip()

    if mb_df.empty or "Player" not in mb_df.columns:
        return []

    sub = mb_df[mb_df["Player"].astype(str).str.strip() == player_name].copy()
    if sub.empty:
        return []

    active_bands = [b.strip() for b in (minutes_bands or []) if b.strip()]
    if not active_bands:
        active_bands = [DEFAULT_MINUTES_BAND]
    band = ", ".join(active_bands)

    # Positions filter (list of positions)
    active_positions = [p.strip() for p in (positions or []) if p.strip().lower() not in ("", "all")]

    # --- Total Games: deduplicate per Season/Position (any band), then sum across positions ---
    sub_for_totals = sub.copy()
    if active_positions:
        sub_for_totals = sub_for_totals[sub_for_totals["Position"].astype(str).str.strip().isin(active_positions)]
    if seasons:
        sub_for_totals = sub_for_totals[sub_for_totals["Season"].isin(seasons)]

    has_total_col = "Total Games played" in sub_for_totals.columns
    if has_total_col and not sub_for_totals.empty:
        total_games_by_season: dict[int, int] = (
            sub_for_totals.groupby(["Season", "Position"])["Total Games played"]
            .first()
            .reset_index()
            .groupby("Season")["Total Games played"]
            .sum()
            .astype(int)
            .to_dict()
        )
    else:
        total_games_by_season = {}

    # --- Filtered stats: apply band(s), positions, seasons ---
    sub_band = sub[sub["Minutes_Band"].isin(active_bands)]
    if active_positions:
        sub_band = sub_band[sub_band["Position"].astype(str).str.strip().isin(active_positions)]
    if seasons:
        sub_band = sub_band[sub_band["Season"].isin(seasons)]

    def _agg_odds_val(hits: int, gp: int) -> float | None:
        return round(gp / hits, 2) if hits > 0 and gp > 0 else None

    def _agg_fmt(hits: int, gp: int) -> str:
        v = _agg_odds_val(hits, gp)
        return f"${v:.2f}" if v is not None else "—"

    # When filtering to a single position, keep it in the output; otherwise aggregate by season
    group_cols = ["Season"]
    if len(active_positions) == 1:
        group_cols = ["Season", "Position"]

    stat_sum_cols = ["Games played", "FTS", "ATS", "LTS", "FTS2H", "2+"]
    agg_df = sub_band.groupby(group_cols)[stat_sum_cols].sum().reset_index()
    agg_df = agg_df.sort_values("Season")

    out = []
    totals: dict[str, int] = {s: 0 for s in ["games_played", "total_games_played", "fts", "ats", "lts", "fts2h", "two_plus"]}

    for _, row in agg_df.iterrows():
        season_val = int(row["Season"])
        gp = int(row.get("Games played", 0))
        fts = int(row.get("FTS", 0))
        ats = int(row.get("ATS", 0))
        lts = int(row.get("LTS", 0))
        fts2h = int(row.get("FTS2H", 0))
        two_plus = int(row.get("2+", 0))
        tgp = int(total_games_by_season.get(season_val, 0))

        totals["games_played"] += gp
        totals["total_games_played"] += tgp
        totals["fts"] += fts
        totals["ats"] += ats
        totals["lts"] += lts
        totals["fts2h"] += fts2h
        totals["two_plus"] += two_plus

        row_position = str(row.get("Position", "All")).strip() if "Position" in agg_df.columns else (", ".join(active_positions) if active_positions else "All")

        out.append({
            "season": season_val,
            "position": row_position,
            "minutes_band": band,
            "games_played": gp,
            "total_games_played": tgp,
            "fts": fts,
            "fts_historical_odds": _agg_odds_val(fts, gp),
            "fts_odds_fmt": _agg_fmt(fts, gp),
            "ats": ats,
            "ats_historical_odds": _agg_odds_val(ats, gp),
            "ats_odds_fmt": _agg_fmt(ats, gp),
            "lts": lts,
            "lts_historical_odds": _agg_odds_val(lts, gp),
            "lts_odds_fmt": _agg_fmt(lts, gp),
            "fts2h": fts2h,
            "fts2h_historical_odds": _agg_odds_val(fts2h, gp),
            "fts2h_odds_fmt": _agg_fmt(fts2h, gp),
            "two_plus": two_plus,
            "two_plus_historical_odds": _agg_odds_val(two_plus, gp),
            "two_plus_odds_fmt": _agg_fmt(two_plus, gp),
        })

    # Append Total row — recompute odds from aggregates (hits/games)
    def _agg_odds(hits: int, gp: int) -> float | None:
        return round(gp / hits, 2) if hits > 0 and gp > 0 else None

    def _agg_odds_fmt(hits: int, gp: int) -> str:
        v = _agg_odds(hits, gp)
        return f"${v:.2f}" if v is not None else "—"

    tgp_total = totals["games_played"]
    out.append({
        "season": "Total",
        "position": ", ".join(active_positions) if active_positions else "All",
        "minutes_band": band,
        "games_played": tgp_total,
        "total_games_played": totals["total_games_played"],
        "fts": totals["fts"],
        "fts_historical_odds": _agg_odds(totals["fts"], tgp_total),
        "fts_odds_fmt": _agg_odds_fmt(totals["fts"], tgp_total),
        "ats": totals["ats"],
        "ats_historical_odds": _agg_odds(totals["ats"], tgp_total),
        "ats_odds_fmt": _agg_odds_fmt(totals["ats"], tgp_total),
        "lts": totals["lts"],
        "lts_historical_odds": _agg_odds(totals["lts"], tgp_total),
        "lts_odds_fmt": _agg_odds_fmt(totals["lts"], tgp_total),
        "fts2h": totals["fts2h"],
        "fts2h_historical_odds": _agg_odds(totals["fts2h"], tgp_total),
        "fts2h_odds_fmt": _agg_odds_fmt(totals["fts2h"], tgp_total),
        "two_plus": totals["two_plus"],
        "two_plus_historical_odds": _agg_odds(totals["two_plus"], tgp_total),
        "two_plus_odds_fmt": _agg_odds_fmt(totals["two_plus"], tgp_total),
    })

    return out


def _get_live_prices_df() -> pd.DataFrame:
    load_data()
    global _df_live_prices
    if _df_live_prices is None:
        _df_live_prices = pd.DataFrame()
    return _df_live_prices if not _df_live_prices.empty else pd.DataFrame()


def _get_best_prices_from_summary(player_name: str, market: str) -> list[dict[str, Any]]:
    """Return list of {website, price} from market summary CSV only. No fabrication."""
    market_norm = market.upper() if market != STAT_2PLUS else STAT_2PLUS
    filenames = _SUMMARY_CSV.get(market_norm)
    if not filenames:
        return []
    for base in (_BASE_DIR, _REPO_DATA):
        path = os.path.join(base, filenames[0])
        if not os.path.isfile(path):
            continue
        try:
            df = pd.read_csv(path)
            df.columns = [str(c).strip() for c in df.columns]
            if "Player" not in df.columns:
                return []
            name_key = player_name.strip().lower()
            mask = df["Player"].astype(str).str.strip().str.lower() == name_key
            rows = df.loc[mask]
            if rows.empty:
                return []
            offers = []
            for _, row in rows.iterrows():
                for col in df.columns:
                    if col not in _PRICE_COLUMNS:
                        continue
                    try:
                        v = row[col]
                        if pd.isna(v) or v == "":
                            continue
                        p = float(pd.to_numeric(v, errors="coerce"))
                        if p > 0:
                            offers.append({"website": col, "price": p})
                    except (TypeError, ValueError):
                        continue
            by_website: dict[str, float] = {}
            for o in offers:
                w = o["website"]
                if w not in by_website or o["price"] > by_website[w]:
                    by_website[w] = o["price"]
            out = [{"website": w, "price": p} for w, p in by_website.items()]
            out.sort(key=lambda x: -x["price"])
            return out
        except Exception:
            continue
    return []


def get_live_prices(player_name: str, market: str) -> list[dict[str, Any]]:
    return _get_best_prices_from_summary(player_name, market)


def _get_best_price_for_value(player_name: str, market: str) -> tuple[float | None, str | None]:
    offers = _get_best_prices_from_summary(player_name, market)
    if offers:
        return offers[0]["price"], offers[0]["website"]
    df = _get_live_prices_df()
    if df.empty or "Player" not in df.columns or "Market" not in df.columns or "Website" not in df.columns or "Price" not in df.columns:
        return None, None
    market_norm = market.upper() if market != STAT_2PLUS else STAT_2PLUS
    sub = df[
        (df["Player"].astype(str).str.strip().str.lower() == player_name.strip().lower())
        & (df["Market"].astype(str).str.strip().str.upper() == market_norm)
    ]
    if sub.empty:
        return None, None
    sub = sub.dropna(subset=["Price"])
    sub = sub[sub["Price"] > 0]
    if sub.empty:
        return None, None
    best = sub.loc[sub["Price"].idxmax()]
    return float(best["Price"]), str(best["Website"]).strip()


def format_best_prices_response(player_name: str, market: str, offers: list[dict[str, Any]]) -> str:
    if not offers:
        return (
            f"No price data available for {player_name} {market} this round. "
            "The player may not be listed in the current round's markets."
        )
    best = offers[0]
    lines = [
        f"{player_name} — {market} bookmaker prices this round:",
        "",
        f"Best price: ${best['price']:.2f} on {best['website']}",
        "",
        "All bookmakers:",
    ]
    for o in offers:
        marker = " ◄ best" if o["website"] == best["website"] and o["price"] == best["price"] else ""
        lines.append(f"• {o['website']}: ${o['price']:.2f}{marker}")
    return "\n".join(lines)


def compute_rankings(
    stat_type: str,
    season_from: int,
    season_to: int,
    positions: list[str],
    min_games: int | None,
    min_games_since_2024: int | None,
    top_n: int,
    ascending: bool = False,
    min_pct: float | None = None,
    minutes_band: str | None = None,
) -> list[dict[str, Any]]:
    """
    Rank players by try-scoring percentage using the minutes-bands CSV.
    Denominator = Total Games played (all games in season range, regardless of minutes band).
    Numerator = stat hits from the specified minutes_band (default Over 20 mins).
    Position filter applied directly from the minutes-bands CSV Position column.
    Inactive-since-2024 filter based on Total Games played >= 2024.
    """
    mb_df = _get_minutesbands()
    if mb_df.empty:
        # Fall back to legacy full CSV if minutes-bands data unavailable
        return _compute_rankings_legacy(
            stat_type, season_from, season_to, positions,
            min_games, min_games_since_2024, top_n, ascending, min_pct
        )

    band = minutes_band or DEFAULT_MINUTES_BAND
    stat_col_name = stat_type  # column name in minutes-bands CSV matches stat type directly

    # Season filter
    sub = mb_df[(mb_df["Season"] >= season_from) & (mb_df["Season"] <= season_to)].copy()

    # Position filter (from minutes-bands CSV Position column)
    if positions and "Position" in sub.columns:
        sub = sub[sub["Position"].astype(str).str.strip().isin(positions)]

    if sub.empty:
        return []

    # --- Total Games: deduplicate per Player/Season/Position, sum across positions per player ---
    # This gives us the true total games played regardless of minutes band
    if "Total Games played" in sub.columns:
        totals_df = (
            sub.groupby(["Player", "Season", "Position"])["Total Games played"]
            .first()
            .reset_index()
            .groupby("Player")["Total Games played"]
            .sum()
            .astype(int)
            .rename("total_games")
        )
    else:
        totals_df = pd.Series(dtype=int, name="total_games")

    # --- Stat hits: filter to specific minutes band ---
    band_sub = sub[sub["Minutes_Band"] == band].copy()
    if stat_col_name not in band_sub.columns:
        return []

    hits_df = (
        band_sub.groupby("Player")[stat_col_name]
        .sum()
        .astype(int)
        .rename("stat_hits")
    )

    # Merge total_games + stat_hits
    agg = pd.DataFrame({"total_games": totals_df, "stat_hits": hits_df}).fillna(0)
    agg["total_games"] = agg["total_games"].astype(int)
    agg["stat_hits"] = agg["stat_hits"].astype(int)
    agg = agg[agg["total_games"] > 0].copy()
    agg["pct"] = (agg["stat_hits"] / agg["total_games"] * 100).round(2)
    agg["hist_odds"] = (agg["total_games"] / agg["stat_hits"]).round(2).where(agg["stat_hits"] > 0, other=None)
    agg = agg.reset_index().rename(columns={"Player": "player_name"})

    # Inactive-since-2024 filter: player must have Total Games played >= 2024
    if "Total Games played" in mb_df.columns:
        recent_sub = mb_df[mb_df["Season"] >= 2024].copy()
        if positions and "Position" in recent_sub.columns:
            recent_sub = recent_sub[recent_sub["Position"].astype(str).str.strip().isin(positions)]
        recent_totals = (
            recent_sub.groupby(["Player", "Season", "Position"])["Total Games played"]
            .first()
            .reset_index()
            .groupby("Player")["Total Games played"]
            .sum()
            .astype(int)
        )
        agg["recent_2024"] = agg["player_name"].map(recent_totals).fillna(0).astype(int)
        agg = agg[agg["recent_2024"] >= (min_games_since_2024 or 0)]

    agg = agg[agg["total_games"] >= (min_games or 0)]
    if min_pct is not None:
        agg = agg[agg["pct"] >= min_pct]

    agg = agg.sort_values(
        by=["pct", "stat_hits", "total_games", "player_name"],
        ascending=[ascending, False, False, True],
    )
    rows = agg.head(top_n)
    result = []
    for _, r in rows.iterrows():
        name = r["player_name"]
        team, pos = _current_meta(name)
        # Prefer position from the minutes-bands data if _current_meta returns None
        if pos is None and positions:
            pos = positions[0] if len(positions) == 1 else None
        if pos is None and "Position" in band_sub.columns:
            player_positions = band_sub[band_sub["Player"].astype(str).str.strip() == str(name).strip()]["Position"].dropna().unique()
            if len(player_positions) == 1:
                pos = str(player_positions[0]).strip()
        result.append({
            "player_name": name,
            "team": team,
            "position": pos,
            "total_games": int(r["total_games"]),
            "stat_hits": int(r["stat_hits"]),
            "stat_type": stat_type,
            "pct": float(r["pct"]),
            "hist_odds": float(r["hist_odds"]) if r["hist_odds"] is not None and pd.notna(r["hist_odds"]) else None,
        })
    return result


def _compute_rankings_legacy(
    stat_type: str,
    season_from: int,
    season_to: int,
    positions: list[str],
    min_games: int | None,
    min_games_since_2024: int | None,
    top_n: int,
    ascending: bool = False,
    min_pct: float | None = None,
) -> list[dict[str, Any]]:
    """Fallback ranking using legacy full CSV. Used only when minutes-bands data is unavailable."""
    df_full, df_players, _ = load_data()
    col = _stat_col(stat_type)
    mask = _get_season_mask(df_full, season_from, season_to)
    sub = df_full.loc[mask]

    agg = sub.groupby("player_id").agg(
        total_games=("Games played", "sum"),
        stat_hits=(col, "sum"),
        player_name=("Player", "first"),
    ).reset_index()
    agg["total_games"] = agg["total_games"].astype(int)
    agg["stat_hits"] = agg["stat_hits"].astype(int)
    agg["pct"] = (agg["stat_hits"] / agg["total_games"] * 100).round(2)
    agg["hist_odds"] = (agg["total_games"] / agg["stat_hits"]).round(2).where(agg["stat_hits"] > 0, None)

    recent = df_full[df_full["season"] >= 2024].groupby("player_id")["Games played"].sum().reindex(agg["player_id"]).fillna(0).astype(int)
    agg["recent_2024"] = agg["player_id"].map(recent).fillna(0).astype(int)
    agg = agg[agg["recent_2024"] >= (min_games_since_2024 or 0)]
    agg = agg[agg["total_games"] >= (min_games or 0)]
    if min_pct is not None:
        agg = agg[agg["pct"] >= min_pct]

    if positions:
        if not df_players.empty and "Position" in df_players.columns:
            pos_map = df_players.set_index(df_players["Player"].astype(str).str.strip())["Position"].astype(str).str.strip().to_dict()
            def allowed(name: str) -> bool:
                p = pos_map.get(str(name).strip())
                return p in positions if p else False
            agg = agg[agg["player_name"].map(allowed)]

    agg = agg[agg["total_games"] > 0]
    agg = agg.sort_values(
        by=["pct", "stat_hits", "total_games", "player_name"],
        ascending=[ascending, False, False, True],
    )
    result = []
    for _, r in agg.head(top_n).iterrows():
        name = r["player_name"]
        team, pos = _current_meta(name)
        result.append({
            "player_name": name,
            "team": team,
            "position": pos,
            "total_games": int(r["total_games"]),
            "stat_hits": int(r["stat_hits"]),
            "stat_type": stat_type,
            "pct": float(r["pct"]),
            "hist_odds": float(r["hist_odds"]) if pd.notna(r["hist_odds"]) else None,
        })
    return result


VALUE_FLOOR_PCT = 0.80


def compute_value_analysis(
    player_id: int,
    stat_type: str,
    season_from: int,
    season_to: int,
    market_odds: float,
) -> dict[str, Any]:
    res = compute_player_stats(player_id, stat_type, season_from, season_to)
    total_games = res["total_games"]
    stat_hits = res["stat_hits"]
    hist_prob = (stat_hits / total_games) if total_games else 0.0
    hist_odds = (total_games / stat_hits) if stat_hits else None
    implied_prob = 1.0 / market_odds if market_odds else 0.0
    edge_pct = (hist_prob - implied_prob) * 100
    positive_value = hist_prob > implied_prob
    value_ratio = (market_odds / hist_odds) if (hist_odds and hist_odds > 0 and market_odds) else None
    value_floor = round(market_odds * VALUE_FLOOR_PCT, 2) if positive_value and market_odds else None
    return {
        **res,
        "market_odds": market_odds,
        "hist_odds": hist_odds,
        "hist_prob": round(hist_prob * 100, 2),
        "implied_prob": round(implied_prob * 100, 2),
        "edge_pct_points": round(edge_pct, 2),
        "positive_value": positive_value,
        "value_ratio": round(value_ratio, 2) if value_ratio is not None else None,
        "value_floor": value_floor,
    }


# ---------------------------------------------------------------------------
# Response formatters — bullet style (no *** markdown)
# ---------------------------------------------------------------------------

def _fmt_odds_str(hist_odds: float | None) -> str:
    return f"${hist_odds:.2f}" if hist_odds is not None else "—"


def _fmt_season_bullet(
    season_label: str,
    stat_type: str,
    gp: int,
    hits: int,
    odds_raw: Any,
    total_gp: int | None = None,
) -> str:
    """Build one bullet line: • 2023 — GP 20/24 | ATS 7/20 (35.0%, $2.86)"""
    pct = (hits / gp * 100) if gp else 0.0
    if pd.isna(odds_raw) or str(odds_raw).strip().upper() == "NA" or str(odds_raw).strip() == "":
        odds_str = "—"
    else:
        try:
            odds_str = f"${float(odds_raw):.2f}"
        except (TypeError, ValueError):
            odds_str = "—"
    gp_str = f"{gp}/{total_gp}" if total_gp is not None and total_gp > 0 else str(gp)
    return f"• {season_label} — GP {gp_str} | {stat_type} {hits}/{gp} ({pct:.1f}%, {odds_str})"


def _data_range_footer() -> str:
    """Return a one-line string describing the dataset's season coverage."""
    load_data()
    if _available_seasons:
        lo, hi = min(_available_seasons), max(_available_seasons)
        return f"Data covers {lo}–{hi} NRL seasons."
    return "Data covers all available NRL seasons."


def format_single_player_response(
    player_name: str,
    stat_type: str,
    season_from: int,
    season_to: int,
    stats: dict[str, Any],
    minutes_band: str | None = None,
    position: str | None = None,
) -> str:
    """
    Bullet-formatted per-season breakdown from the minutes-bands CSV.
    Falls back to aggregate summary if per-season data is unavailable.
    """
    team, pos = _current_meta(player_name)
    band = minutes_band or DEFAULT_MINUTES_BAND

    filter_parts = []
    if position:
        filter_parts.append(position)
    elif pos:
        filter_parts.append(pos)
    filter_parts.append(band)
    season_range = f"{season_from}–{season_to}" if season_from != season_to else str(season_from)
    filter_parts.append(season_range)
    header = f"{player_name} — {stat_type} stats ({', '.join(filter_parts)})"

    mb_df = _get_minutesbands()
    if not mb_df.empty and "Player" in mb_df.columns:
        player_rows = mb_df[mb_df["Player"].astype(str).str.strip() == player_name.strip()].copy()
        player_rows = player_rows[player_rows["Minutes_Band"] == band]
        if position and position.strip().lower() not in ("", "all"):
            player_rows = player_rows[player_rows["Position"].astype(str).str.strip() == position.strip()]
        player_rows = player_rows[
            (player_rows["Season"] >= season_from) & (player_rows["Season"] <= season_to)
        ]

        if not player_rows.empty:
            # Build total_games_by_season (ignoring band, across the positions already filtered)
            all_pos_rows = mb_df[mb_df["Player"].astype(str).str.strip() == player_name.strip()].copy()
            if position and position.strip().lower() not in ("", "all"):
                all_pos_rows = all_pos_rows[all_pos_rows["Position"].astype(str).str.strip() == position.strip()]
            all_pos_rows = all_pos_rows[
                (all_pos_rows["Season"] >= season_from) & (all_pos_rows["Season"] <= season_to)
            ]
            total_games_map: dict[int, int] = {}
            if "Total Games played" in all_pos_rows.columns and not all_pos_rows.empty:
                total_games_map = (
                    all_pos_rows.groupby(["Season", "Position"])["Total Games played"]
                    .first()
                    .reset_index()
                    .groupby("Season")["Total Games played"]
                    .sum()
                    .astype(int)
                    .to_dict()
                )

            # Aggregate multiple positions within the same season
            agg = player_rows.groupby("Season")[["Games played", stat_type]].sum().reset_index()
            agg = agg.sort_values("Season")

            lines = [header, ""]
            total_gp = 0
            total_hits = 0
            total_all_games = 0
            for _, row in agg.iterrows():
                season_int = int(row["Season"])
                gp = int(row.get("Games played", 0))
                hits = int(pd.to_numeric(row.get(stat_type, 0), errors="coerce") or 0)
                tgp = total_games_map.get(season_int)
                odds_raw = (gp / hits) if hits > 0 else None
                lines.append(_fmt_season_bullet(str(season_int), stat_type, gp, hits, odds_raw, total_gp=tgp))
                total_gp += gp
                total_hits += hits
                total_all_games += tgp if tgp is not None else gp
            total_odds_raw = (total_gp / total_hits) if total_hits else None
            lines.append(_fmt_season_bullet("Total", stat_type, total_gp, total_hits, total_odds_raw, total_gp=total_all_games if total_all_games else None))
            lines.append("")
            lines.append(_data_range_footer())
            return "\n".join(lines)

    # Fallback: aggregate from full CSV
    odds_str = _fmt_odds_str(stats.get("hist_odds"))
    pct = stats.get("pct", 0.0)
    hits = stats.get("stat_hits", 0)
    total_games = stats.get("total_games", 0)
    return (
        f"{header}\n\n"
        f"• {season_range} — GP {total_games} | {stat_type} {hits}/{total_games} ({pct:.2f}%, {odds_str})\n\n"
        f"{_data_range_footer()}"
    )


def format_all_stats_summary(
    player_name: str,
    season_from: int,
    season_to: int,
    minutes_band: str | None = None,
    position: str | None = None,
) -> str:
    """
    Show all 5 markets as a compact summary line, then per-season ATS breakdown.
    Used when the user asks for generic 'stats?' without specifying a market.
    """
    team, pos = _current_meta(player_name)
    band = minutes_band or DEFAULT_MINUTES_BAND
    season_range = f"{season_from}–{season_to}" if season_from != season_to else str(season_from)
    filter_parts = []
    if position:
        filter_parts.append(position)
    elif pos:
        filter_parts.append(pos)
    filter_parts.append(band)
    filter_parts.append(season_range)
    header = f"{player_name} — Try Scoring Stats ({', '.join(filter_parts)})"

    mb_df = _get_minutesbands()
    if mb_df.empty or "Player" not in mb_df.columns:
        return f"{header}\n\nNo data found.\n\n{_data_range_footer()}"

    player_rows = mb_df[mb_df["Player"].astype(str).str.strip() == player_name.strip()].copy()
    player_rows = player_rows[player_rows["Minutes_Band"] == band]
    if position and position.strip().lower() not in ("", "all"):
        player_rows = player_rows[player_rows["Position"].astype(str).str.strip() == position.strip()]
    player_rows = player_rows[
        (player_rows["Season"] >= season_from) & (player_rows["Season"] <= season_to)
    ]

    if player_rows.empty:
        return f"{header}\n\nNo data found for this filter combination.\n\n{_data_range_footer()}"

    stat_cols = [s for s in STAT_COLUMNS if s in player_rows.columns]
    totals: dict[str, int] = {}
    total_gp = int(player_rows["Games played"].sum())
    for s in stat_cols:
        totals[s] = int(pd.to_numeric(player_rows[s], errors="coerce").fillna(0).sum())

    # Summary line for all markets
    summary_parts = []
    for s in stat_cols:
        hits = totals[s]
        gp = total_gp
        pct = (hits / gp * 100) if gp else 0.0
        odds = f"${gp/hits:.2f}" if hits > 0 else "—"
        summary_parts.append(f"{s} {hits}/{gp} ({pct:.1f}%, {odds})")

    # Per-season ATS breakdown
    agg = player_rows.groupby("Season")[["Games played", "ATS"]].sum().reset_index()
    agg = agg.sort_values("Season")

    # Total games per season (for display as GP band/total)
    all_pos_rows = mb_df[mb_df["Player"].astype(str).str.strip() == player_name.strip()].copy()
    if position and position.strip().lower() not in ("", "all"):
        all_pos_rows = all_pos_rows[all_pos_rows["Position"].astype(str).str.strip() == position.strip()]
    all_pos_rows = all_pos_rows[
        (all_pos_rows["Season"] >= season_from) & (all_pos_rows["Season"] <= season_to)
    ]
    total_games_map: dict[int, int] = {}
    if "Total Games played" in all_pos_rows.columns and not all_pos_rows.empty:
        total_games_map = (
            all_pos_rows.groupby(["Season", "Position"])["Total Games played"]
            .first()
            .reset_index()
            .groupby("Season")["Total Games played"]
            .sum()
            .astype(int)
            .to_dict()
        )

    lines = [header, ""]
    lines.append("Summary (all markets):")
    for part in summary_parts:
        lines.append(f"• {part}")
    lines.append("")
    lines.append("Per season (ATS):")

    ats_total_gp = 0
    ats_total_hits = 0
    ats_total_all = 0
    for _, row in agg.iterrows():
        season_int = int(row["Season"])
        gp = int(row.get("Games played", 0))
        hits = int(pd.to_numeric(row.get("ATS", 0), errors="coerce") or 0)
        tgp = total_games_map.get(season_int)
        odds_raw = (gp / hits) if hits > 0 else None
        lines.append(_fmt_season_bullet(str(season_int), "ATS", gp, hits, odds_raw, total_gp=tgp))
        ats_total_gp += gp
        ats_total_hits += hits
        ats_total_all += tgp if tgp is not None else gp

    total_odds_raw = (ats_total_gp / ats_total_hits) if ats_total_hits else None
    lines.append(_fmt_season_bullet("Total", "ATS", ats_total_gp, ats_total_hits, total_odds_raw, total_gp=ats_total_all if ats_total_all else None))
    lines.append("")
    lines.append(_data_range_footer())
    return "\n".join(lines)


def format_ranking_response(
    rows: list[dict],
    stat_type: str,
    season_from: int,
    season_to: int,
    positions: list[str],
    min_games: int | None,
    min_games_since_2024: int | None,
    min_pct: float | None = None,
    minutes_band: str | None = None,
) -> str:
    """Numbered list with compact per-player line; then filter summary bullet."""
    season_range = f"{season_from}–{season_to}" if season_from != season_to else str(season_from)
    band = minutes_band or DEFAULT_MINUTES_BAND

    filter_parts = [f"stat={stat_type}", f"seasons={season_range}"]
    if positions:
        filter_parts.append(f"position={', '.join(positions)}")
    filter_parts.append(f"minutes={band}")
    if min_games is not None:
        filter_parts.append(f"min games={min_games}")
    if min_games_since_2024 is not None:
        filter_parts.append(f"min games since 2024={min_games_since_2024}")
    if min_pct is not None:
        filter_parts.append(f"min rate={min_pct}%")

    lines = [f"{stat_type} rankings ({season_range})", ""]
    for i, r in enumerate(rows, 1):
        label = r["player_name"]
        meta_parts = [p for p in [r.get("team"), r.get("position")] if p]
        if meta_parts:
            label += f" ({', '.join(meta_parts)})"
        odds_str = _fmt_odds_str(r.get("hist_odds"))
        # total_games is all games played (Total Games played); stat_hits scored in that window
        lines.append(
            f"{i}) {label} — {stat_type} {r['stat_hits']}/{r['total_games']} games ({r['pct']:.1f}%), hist {odds_str}"
        )
    lines.append("")
    lines.append(f"• Filters: {', '.join(filter_parts)}. GP = total games played in period. Inactive since 2024 excluded.")
    lines.append(_data_range_footer())
    return "\n".join(lines)


def format_value_response(
    player_name: str,
    value: dict[str, Any],
    season_from: int,
    season_to: int,
    website: str | None = None,
    minutes_band: str | None = None,
) -> str:
    """3-bullet value check: stat line, market vs historical, verdict."""
    team, pos = _current_meta(player_name)
    band = minutes_band or DEFAULT_MINUTES_BAND
    filter_parts = []
    if pos:
        filter_parts.append(pos)
    filter_parts.append(band)
    season_range = f"{season_from}–{season_to}" if season_from != season_to else str(season_from)
    filter_parts.append(season_range)

    stat_type = value["stat_type"]
    gp = value["total_games"]
    hits = value["stat_hits"]
    hist_pct = value["hist_prob"]
    hist_odds = value.get("hist_odds")
    hist_odds_str = _fmt_odds_str(hist_odds)
    implied_pct = value["implied_prob"]
    edge = value["edge_pct_points"]
    market_odds = value["market_odds"]
    price_str = f"${market_odds:.2f}"
    if website:
        price_str += f" on {website}"

    if abs(edge) < 0.5:
        verdict_label = "Fair value"
    elif value["positive_value"]:
        verdict_label = "Positive historical value"
    else:
        verdict_label = "Negative historical value"

    lines = [
        f"{player_name} — {stat_type} value check ({', '.join(filter_parts)})",
        "",
        f"• Stat: {hits}/{gp} games ({hist_pct:.2f}% hist), hist odds {hist_odds_str}",
        f"• Market: {price_str} (implied {implied_pct:.2f}%) vs historical {hist_pct:.2f}% — edge {edge:+.2f} pp",
        f"• Verdict: {verdict_label}",
    ]
    if value.get("value_ratio") is not None:
        lines.append(f"• Value by: {value['value_ratio']:.2f}x (market vs historical odds)")
    if value.get("value_floor") is not None:
        lines.append(f"• Still value down to ${value['value_floor']:.2f} (20% benchmark)")
    elif hist_odds is not None and hist_odds > 0:
        lines.append(f"• Would be value at or above ${hist_odds:.2f} (20% benchmark)")
    lines.append("")
    lines.append(_data_range_footer())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summary CSV helpers — model odds, historical, bookmaker prices
# ---------------------------------------------------------------------------

# Maps market key → (summary CSV historical col, summary CSV model col or None)
_SUMMARY_HIST_COL: dict[str, tuple[str, str | None]] = {
    "FTS": ("FTS Historical", "FTS Model"),
    "ATS": ("ATS Historical", "ATS Model"),
    "LTS": ("LTS Historical", None),
    "FTS2H": ("FTS2H Historical", None),
    "2+": ("TPT Historical", None),
}


def get_summary_row(player_name: str, stat_type: str) -> dict[str, Any]:
    """
    Read the appropriate summary CSV and return a dict with:
      hist_odds, model_odds (or None), highest_price, highest_bookie,
      value_vs_hist (%), value_vs_model (% or None), market_value (%),
      all_prices: list of {website, price}
    Returns an empty dict if no data found.
    """
    market_norm = stat_type.upper() if stat_type != STAT_2PLUS else STAT_2PLUS
    filenames = _SUMMARY_CSV.get(market_norm)
    if not filenames:
        return {}

    for base in (_BASE_DIR, _REPO_DATA):
        path = os.path.join(base, filenames[0])
        if not os.path.isfile(path):
            continue
        try:
            df = pd.read_csv(path)
            df.columns = [str(c).strip() for c in df.columns]
            if "Player" not in df.columns:
                return {}
            mask = df["Player"].astype(str).str.strip().str.lower() == player_name.strip().lower()
            rows = df.loc[mask]
            if rows.empty:
                return {}
            row = rows.iloc[0]

            hist_col, model_col = _SUMMARY_HIST_COL.get(market_norm, (None, None))

            def _safe_float(val: Any) -> float | None:
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    return None
                try:
                    v = float(pd.to_numeric(val, errors="coerce"))
                    return v if not pd.isna(v) and v > 0 else None
                except (TypeError, ValueError):
                    return None

            hist_odds = _safe_float(row.get(hist_col)) if hist_col and hist_col in df.columns else None
            model_odds = _safe_float(row.get(model_col)) if model_col and model_col in df.columns else None
            highest = _safe_float(row.get("Highest"))
            val_hist = _safe_float(row.get("Highest/Historical"))
            val_model = _safe_float(row.get("Highest/Model"))
            market_value = _safe_float(row.get("Market Value"))

            # Individual bookmaker prices
            all_prices: list[dict[str, Any]] = []
            for col in sorted(_PRICE_COLUMNS):
                if col in df.columns:
                    p = _safe_float(row.get(col))
                    if p is not None:
                        all_prices.append({"website": col, "price": p})
            all_prices.sort(key=lambda x: -x["price"])

            best = all_prices[0] if all_prices else None

            return {
                "player_name": str(row.get("Player", player_name)).strip(),
                "team": str(row.get("Team", "")).strip() or None,
                "market": market_norm,
                "hist_odds": hist_odds,
                "model_odds": model_odds,
                "highest_price": highest,
                "highest_bookie": best["website"] if best else None,
                "value_vs_hist": val_hist,
                "value_vs_model": val_model,
                "market_value": market_value,
                "all_prices": all_prices,
            }
        except Exception:
            continue
    return {}


def format_combined_bet_response(
    player_name: str,
    stat_type: str,
    summary: dict[str, Any],
    season_from: int,
    season_to: int,
    minutes_band: str | None = None,
) -> str:
    """
    Combined bet analysis: model odds first, then market prices, hist secondary.
    Used when user asks 'best price + is it a good bet?' in one query.
    """
    band = minutes_band or DEFAULT_MINUTES_BAND
    _, pos = _current_meta(player_name)
    filter_parts = []
    if pos:
        filter_parts.append(pos)
    filter_parts.append(band)
    season_range = f"{season_from}–{season_to}" if season_from != season_to else str(season_from)
    filter_parts.append(season_range)

    lines = [f"{player_name} — {stat_type} Bet Analysis ({', '.join(filter_parts)})", ""]

    hist_odds = summary.get("hist_odds")
    model_odds = summary.get("model_odds")
    val_hist = summary.get("value_vs_hist")
    val_model = summary.get("value_vs_model")
    all_prices = summary.get("all_prices", [])
    highest = summary.get("highest_price")
    best_bookie = summary.get("highest_bookie")

    # --- Historical record (for narrative context) ---
    df_full, _, _ = load_data()
    match = df_full[df_full["Player"].astype(str).str.strip() == player_name.strip()]
    gp_total = 0
    hits_total = 0
    if not match.empty:
        pid = int(match.iloc[0]["player_id"])
        stats = compute_player_stats(pid, stat_type, season_from, season_to)
        gp_total = stats.get("total_games", 0)
        hits_total = stats.get("stat_hits", 0)

    # --- 1. Model odds (primary signal) ---
    if model_odds:
        model_pct = round(100.0 / model_odds, 1)
        lines.append(f"Model odds: ${model_odds:.2f}  (model scores ~{model_pct:.1f}% of games)")
        if gp_total and hits_total:
            hist_pct_val = round(hits_total / gp_total * 100, 1)
            lines.append(
                f"Historical record: {stat_type} {hits_total}/{gp_total} ({hist_pct_val}%) over {season_range}"
            )
            if hist_odds:
                hist_pct_odds = round(100.0 / hist_odds, 1)
                if abs(model_pct - hist_pct_odds) < 2:
                    lines.append("Model and historical rates are closely aligned.")
                elif model_pct > hist_pct_odds:
                    lines.append(
                        f"Model is more optimistic than history ({model_pct:.1f}% vs {hist_pct_odds:.1f}%)."
                    )
                else:
                    lines.append(
                        f"Model is more conservative than history ({model_pct:.1f}% vs {hist_pct_odds:.1f}%)."
                    )
        lines.append("")

    # --- 2. Best market price vs model ---
    if all_prices:
        if highest and best_bookie:
            lines.append(f"Best available price: ${highest:.2f} on {best_bookie}  ◄")
        if model_odds and val_model is not None:
            lines.append(f"Best price is {val_model:.0f}% of model (model = ${model_odds:.2f})")
        lines.append("")
        lines.append("All bookmakers:")
        for o in all_prices:
            marker = "  ◄ best" if best_bookie and o["website"] == best_bookie and o["price"] == highest else ""
            lines.append(f"• {o['website']}: ${o['price']:.2f}{marker}")
        # Note large price spread across bookies
        if len(all_prices) >= 2:
            low = all_prices[-1]["price"]
            high = all_prices[0]["price"]
            if high > 0 and low > 0 and (high / low) >= 1.5:
                lines.append(
                    f"  (Wide spread: ${low:.2f}–${high:.2f} — shopping around makes a difference)"
                )
    else:
        lines.append("No bookmaker prices available this round.")

    # --- 3. Historical odds (secondary) ---
    if hist_odds:
        hist_pct = round(100.0 / hist_odds, 1)
        val_hist_str = f" ({val_hist:.0f}% of hist)" if val_hist is not None else ""
        lines.append("")
        lines.append(f"Historical odds: ${hist_odds:.2f}  (scores ~{hist_pct:.1f}% historically){val_hist_str}")

    # --- 4. Verdict ---
    lines.append("")
    is_value_model = val_model is not None and val_model >= 100
    is_value_hist = val_hist is not None and val_hist >= 100
    if is_value_model and is_value_hist:
        verdict = "Strong value signal — best price beats both model and historical rates."
    elif is_value_model:
        verdict = "Model says value. Best price beats model odds — worth considering."
    elif is_value_hist:
        verdict = "Historical value. Best price beats historical rate, but model suggests tighter odds."
    elif highest and model_odds and highest >= model_odds * 0.90:
        verdict = "Close to model fair value (within 10%). Marginal edge at best."
    elif highest and hist_odds and highest >= hist_odds * 0.90:
        verdict = "Close to historical fair value (within 10%). No clear edge."
    else:
        verdict = "No value — best price is below both model and historical rates."

    lines.append(f"Verdict: {verdict}")
    lines.append("")
    lines.append(_data_range_footer())
    return "\n".join(lines)


def get_round_value_bets(stat_type: str, top_n: int = 10, min_games: int = 5) -> list[dict[str, Any]]:
    """
    Return top-N value plays for this round from summary CSVs.
    Primary: model is available and best market price beats model odds (Highest/Model > 100).
    Fallback: best market price beats historical rate (Highest/Historical > 100).
    Sorted by Highest/Model desc first, then Highest/Historical desc.
    min_games filters out players with very few career games (noisy stats).
    """
    market_norm = stat_type.upper() if stat_type != STAT_2PLUS else STAT_2PLUS
    filenames = _SUMMARY_CSV.get(market_norm)
    if not filenames:
        return []

    for base in (_BASE_DIR, _REPO_DATA):
        path = os.path.join(base, filenames[0])
        if not os.path.isfile(path):
            continue
        try:
            df = pd.read_csv(path)
            df.columns = [str(c).strip() for c in df.columns]
            required = ["Player", "Highest", "Highest/Historical"]
            if not all(c in df.columns for c in required):
                return []

            df["Highest"] = pd.to_numeric(df["Highest"], errors="coerce")
            df["Highest/Historical"] = pd.to_numeric(df["Highest/Historical"], errors="coerce")

            hist_col, model_col = _SUMMARY_HIST_COL.get(market_norm, (None, None))
            if hist_col and hist_col in df.columns:
                df[hist_col] = pd.to_numeric(df[hist_col], errors="coerce")
            if model_col and model_col in df.columns:
                df[model_col] = pd.to_numeric(df[model_col], errors="coerce")
            if "Highest/Model" in df.columns:
                df["Highest/Model"] = pd.to_numeric(df["Highest/Model"], errors="coerce")
            if "Games" in df.columns:
                df["Games"] = pd.to_numeric(df["Games"], errors="coerce")

            # Base filter: market price must beat at least one signal
            has_model_col = "Highest/Model" in df.columns
            if has_model_col:
                value_mask = (
                    (
                        df["Highest/Model"].notna() & (df["Highest/Model"] > 100)
                    ) | (
                        df["Highest/Historical"].notna() & (df["Highest/Historical"] > 100)
                    )
                ) & df["Highest"].notna() & (df["Highest"] > 0)
            else:
                value_mask = (
                    df["Highest/Historical"].notna()
                    & (df["Highest/Historical"] > 100)
                    & df["Highest"].notna()
                    & (df["Highest"] > 0)
                )

            if "Games" in df.columns:
                value_mask = value_mask & (df["Games"].fillna(0) >= min_games)

            value_df = df[value_mask].copy()

            # Sort: Highest/Model desc first (NaN last), then Highest/Historical desc
            if has_model_col:
                value_df["_sort_model"] = value_df["Highest/Model"].fillna(0)
            else:
                value_df["_sort_model"] = 0.0
            value_df["_sort_hist"] = value_df["Highest/Historical"].fillna(0)
            value_df = value_df.sort_values(
                ["_sort_model", "_sort_hist"], ascending=[False, False]
            ).head(top_n)

            price_cols_present = sorted([c for c in _PRICE_COLUMNS if c in df.columns])

            result = []
            for _, row in value_df.iterrows():
                bookie_prices: list[dict[str, Any]] = []
                for col in price_cols_present:
                    try:
                        v = float(pd.to_numeric(row.get(col), errors="coerce"))
                        if not pd.isna(v) and v > 0:
                            bookie_prices.append({"website": col, "price": v})
                    except (TypeError, ValueError):
                        continue
                bookie_prices.sort(key=lambda x: -x["price"])
                best = bookie_prices[0] if bookie_prices else None

                result.append({
                    "player_name": str(row.get("Player", "")).strip(),
                    "team": str(row.get("Team", "")).strip() or None,
                    "hist_odds": float(row[hist_col]) if hist_col and hist_col in df.columns and pd.notna(row.get(hist_col)) else None,
                    "model_odds": float(row[model_col]) if model_col and model_col in df.columns and pd.notna(row.get(model_col)) else None,
                    "highest_price": float(row["Highest"]),
                    "highest_bookie": best["website"] if best else None,
                    "value_vs_hist": float(row["Highest/Historical"]) if pd.notna(row.get("Highest/Historical")) else None,
                    "value_vs_model": float(row["Highest/Model"]) if "Highest/Model" in df.columns and pd.notna(row.get("Highest/Model")) else None,
                    "stat_type": market_norm,
                })
            return result
        except Exception:
            continue
    return []


def format_round_value_response(rows: list[dict[str, Any]], stat_type: str) -> str:
    """Numbered list of this round's best value bets — model + market first, hist secondary."""
    if not rows:
        return (
            f"No value {stat_type} bets found this round.\n"
            "Try a different market (e.g. ATS, FTS, LTS)."
        )
    lines = [f"Best value {stat_type} bets this round:", ""]
    for i, r in enumerate(rows, 1):
        name = r["player_name"]
        team = f" ({r['team']})" if r.get("team") else ""
        best_price = r.get("highest_price")
        best_bookie = r.get("highest_bookie", "")
        price_str = f"${best_price:.2f} on {best_bookie}" if best_price and best_bookie else ("—" if not best_price else f"${best_price:.2f}")
        model_odds = r.get("model_odds")
        hist_odds = r.get("hist_odds")
        val_model = r.get("value_vs_model")
        val_hist = r.get("value_vs_hist")

        parts = []
        if model_odds:
            model_val_str = f" ({val_model:.0f}% of model)" if val_model else ""
            parts.append(f"model ${model_odds:.2f} | market {price_str}{model_val_str}")
        else:
            parts.append(f"market {price_str}")
        if hist_odds:
            hist_val_str = f" ({val_hist:.0f}% of hist)" if val_hist else ""
            parts.append(f"hist ${hist_odds:.2f}{hist_val_str}")

        lines.append(f"{i}) {name}{team} — {' | '.join(parts)}")
    lines.append("")
    lines.append("• Value = best available market price exceeds model odds and/or historical rate.")
    lines.append(_data_range_footer())
    return "\n".join(lines)


def _rule_based_response(pq: "ParsedQuery", message: str) -> str:
    """Run the deterministic rule-based logic given a pre-parsed query."""
    season_from, season_to = resolve_timeframe(pq)
    norm_msg = normalize_text(message)
    mb = pq.minutes_band

    # Round value bets — no player needed; checked before per-player routes
    if pq.round_value_request:
        stat = pq.stat_type or "ATS"
        rows = get_round_value_bets(stat, top_n=10)
        return format_round_value_response(rows, stat)

    # Combined bet analysis — best prices + value assessment in one response
    if pq.bet_analysis_request and pq.player_names:
        name = pq.player_names[0]
        stat = pq.stat_type or "FTS"
        summary = get_summary_row(name, stat)
        if summary:
            return format_combined_bet_response(name, stat, summary, season_from, season_to, minutes_band=mb)
        # Fall through to plain price response if no summary data
        offers = get_live_prices(name, stat)
        return format_best_prices_response(name, stat, offers)

    # Best available price — always deterministic
    if pq.best_price_request and pq.player_names:
        name = pq.player_names[0]
        stat = pq.stat_type or "FTS"
        offers = get_live_prices(name, stat)
        return format_best_prices_response(name, stat, offers)

    # Value question
    is_value_question = (
        (pq.market_odds is not None and pq.market_odds > 0 or "value" in norm_msg or "how much" in norm_msg)
        and pq.player_names
    )
    if is_value_question and pq.player_names:
        stat = pq.stat_type or "FTS"
        name = pq.player_names[0]
        df_full, _, _ = load_data()
        match = df_full[df_full["Player"].astype(str).str.strip() == name.strip()]
        if match.empty:
            return f"No player matched: '{name}'."
        pid = int(match.iloc[0]["player_id"])
        market_odds = pq.market_odds
        website = None
        if (market_odds is None or market_odds <= 0) and ("value" in norm_msg or "how much" in norm_msg):
            market_odds, website = _get_best_price_for_value(name, stat)
        if market_odds is None or market_odds <= 0:
            return "Please give a price (e.g. Is $17 for Payne Haas FTS value?, or '31s LTS' for $31) or ensure the player/market appears in the summary CSVs."
        value = compute_value_analysis(pid, stat, season_from, season_to, market_odds)
        return format_value_response(name, value, season_from, season_to, website=website, minutes_band=mb)
    elif ("value" in norm_msg or "how much" in norm_msg) and not pq.player_names:
        return "Please specify a player name for value analysis (e.g. Is $17 for Payne Haas FTS value?)."

    # Single or few players
    if pq.player_names and not pq.top_n:
        # When no stat type specified, show all-markets summary
        if pq.stat_type is None:
            pos = pq.positions[0] if pq.positions else None
            parts = []
            for name in pq.player_names[:5]:
                parts.append(format_all_stats_summary(name, season_from, season_to, minutes_band=mb, position=pos))
            return "\n\n".join(parts)

        stat = pq.stat_type
        parts = []
        for name in pq.player_names[:5]:
            df_full, _, _ = load_data()
            match = df_full[df_full["Player"].astype(str).str.strip() == name.strip()]
            if match.empty:
                parts.append(f"No player matched: '{name}'.")
                continue
            pid = int(match.iloc[0]["player_id"])
            stats = compute_player_stats(pid, stat, season_from, season_to)
            pos = pq.positions[0] if pq.positions else None
            parts.append(format_single_player_response(name, stat, season_from, season_to, stats, minutes_band=mb, position=pos))
        return "\n\n".join(parts)

    # Ranking
    if pq.top_n or (pq.positions and pq.stat_type):
        stat = pq.stat_type or "ATS"
        min_g, min_2024, min_pct = resolve_games_filters(pq)
        n = resolve_top_n(pq)
        ascending = "worst" in normalize_text(message) or "lowest" in normalize_text(message)
        rows = compute_rankings(
            stat, season_from, season_to,
            pq.positions, min_g, min_2024, n, ascending=ascending, min_pct=min_pct,
            minutes_band=mb,
        )
        if not rows:
            return "No players match all filters. Try relaxing minimum games or position."
        return format_ranking_response(
            rows, stat, season_from, season_to,
            pq.positions, min_g, min_2024, min_pct, minutes_band=mb,
        )

    # Fallback
    stat = pq.stat_type or "ATS"
    if pq.positions:
        min_g, min_2024, min_pct = resolve_games_filters(pq)
        rows = compute_rankings(
            stat, season_from, season_to,
            pq.positions, min_g, min_2024, 10, ascending=False, min_pct=min_pct,
        )
        if rows:
            return format_ranking_response(
                rows, stat, season_from, season_to,
                pq.positions, min_g, min_2024, min_pct, minutes_band=mb,
            )
    return "Please ask about a specific player, a ranking (e.g. top 5 edge forwards for LTS since 2022), or a value question (e.g. Is $17 for Payne Haas FTS value?)."


def _build_disambiguation_message(ambiguous_map: dict[str, list[str]]) -> str:
    """Return a numbered disambiguation prompt for the first ambiguous partial name found."""
    partial, candidates = next(iter(ambiguous_map.items()))
    lines = [
        f"I found multiple players matching '{partial}'. Which one did you mean?",
    ]
    for i, name in enumerate(candidates, 1):
        lines.append(f"{i}) {name}")
    lines.append("Please re-ask using the full name from the list above.")
    return "\n".join(lines)


def get_chat_response(message: str, history: list[dict]) -> str:
    """Produce full response text. Price/bookmaker questions always handled deterministically."""
    load_data()
    pq = parse_query(message)

    # Disambiguation takes priority — ask user to clarify before doing anything else
    if pq.ambiguous_name_map and not pq.player_names:
        return _build_disambiguation_message(pq.ambiguous_name_map)

    # Round value bets — deterministic, never delegate to RAG
    if pq.round_value_request:
        return _rule_based_response(pq, message)

    # Combined bet analysis — deterministic (prices + value), never delegate to RAG
    if pq.bet_analysis_request and pq.player_names:
        return _rule_based_response(pq, message)

    # Always handle price/bookmaker requests deterministically — never delegate to RAG
    if pq.best_price_request and pq.player_names:
        name = pq.player_names[0]
        stat = pq.stat_type or "FTS"
        offers = get_live_prices(name, stat)
        return format_best_prices_response(name, stat, offers)

    # For non-price questions, prefer RAG when available
    try:
        import rag
        if rag.is_rag_available():
            return rag.get_rag_response(message, history)
    except Exception:
        pass

    # Rule-based fallback (reuse already-parsed query)
    return _rule_based_response(pq, message)


def stream_chat_response(message: str, history: list[dict]):
    """Yield full response as chunks for SSE. Preserves newlines and bullet characters."""
    full = get_chat_response(message, history)
    lines = full.split("\n")
    for line_idx, line in enumerate(lines):
        words = line.split()
        for i, w in enumerate(words):
            yield w + (" " if i < len(words) - 1 else "")
        if line_idx < len(lines) - 1:
            yield "\n"
