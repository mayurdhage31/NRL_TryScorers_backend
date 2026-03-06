"""
NRL tryscorers chatbot: deterministic stats, rankings, value analysis.
All calculations in code; LLM optional for phrasing only.
"""
import os
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Data paths and cache
# ---------------------------------------------------------------------------
# Data dir: backend/data, or DATA_DIR env, or fallback to repo data/
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
_available_seasons: list[int] = []

# Stat column name for "2+" in CSV
STAT_2PLUS = "2+"
STAT_COLUMNS = ["FTS", "ATS", "LTS", "FTS2H", STAT_2PLUS]

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
    "ats": "ATS",
    "anytime try scorer": "ATS",
    "anytime tryscorer": "ATS",
    "lts": "LTS",
    "last try scorer": "LTS",
    "last tryscorer": "LTS",
    "fts2h": "FTS2H",
    "first try scorer 2nd half": "FTS2H",
    "first try scorer second half": "FTS2H",
    "first try in second half": "FTS2H",
    "first 2nd half tryscorer": "FTS2H",
    "2+": STAT_2PLUS,
    "2+ tries": STAT_2PLUS,
    "two or more tries": STAT_2PLUS,
    "2 or more tries": STAT_2PLUS,
    "multi try": STAT_2PLUS,
    "multi-try": STAT_2PLUS,
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """Load full tryscorers CSV and players/teams CSV; return (df_full, df_players, seasons)."""
    global _df_full, _df_players, _df_live_prices, _available_seasons
    if _df_full is not None:
        return _df_full, _df_players if _df_players is not None else pd.DataFrame(), _available_seasons

    if not os.path.isfile(_FULL_CSV):
        raise FileNotFoundError(f"Tryscorers data not found: {_FULL_CSV}")

    _df_full = pd.read_csv(_FULL_CSV)
    # Ensure numeric
    for col in ["season", "Games played"] + STAT_COLUMNS:
        if col in _df_full.columns:
            _df_full[col] = pd.to_numeric(_df_full[col], errors="coerce").fillna(0).astype(int)

    _available_seasons = sorted(_df_full["season"].dropna().unique().astype(int).tolist())

    if os.path.isfile(_PLAYERS_CSV):
        _df_players = pd.read_csv(_PLAYERS_CSV)
        # handle unnamed first column
        if _df_players.columns[0].strip() == "" or _df_players.columns[0].startswith("Unnamed"):
            _df_players = _df_players.iloc[:, 1:]
        _df_players.columns = [c.strip() for c in _df_players.columns]
        if "Player" not in _df_players.columns:
            _df_players = pd.DataFrame()
    else:
        _df_players = pd.DataFrame()

    # Live prices (optional): Player, Market, Website, Price
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

    return _df_full, _df_players, _available_seasons


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
    stat_type: str | None = None
    season_from: int | None = None
    season_to: int | None = None
    top_n: int | None = None
    min_games: int | None = None
    min_games_since_2024: int | None = None
    min_pct: float | None = None  # e.g. "better than 1 in 10" -> 10.0
    positions: list[str] = field(default_factory=list)
    market_odds: float | None = None
    best_price_request: bool = False  # "best available price" for player-market
    raw_message: str = ""


def parse_query(message: str) -> ParsedQuery:
    """Extract intent from user message (rule-based + regex)."""
    load_data()  # ensure _available_seasons etc. set
    norm = normalize_text(message)
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
        pq.top_n = pq.top_n or 5  # will sort ascending later

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

    # Better than 1 in N (e.g. better than 1 in 10 -> min 10% rate)
    better = re.search(r"better\s+than\s+1\s+in\s+(\d+)", norm)
    if better:
        n = int(better.group(1))
        pq.min_pct = 100.0 / n if n else None

    # Position
    pq.positions = resolve_positions(norm)

    # Market odds: $17, 17 for/ats/fts/odds, odds of X, or "31s"/"41s"/"81s" (same as $31, $41, $81)
    dollar = re.search(r"\$\s*(\d+(?:\.\d+)?)", message)  # $17 or $17.00
    if dollar:
        pq.market_odds = float(dollar.group(1))
    else:
        # e.g. "31s LTS", "41s FTS", "81s" - number + "s" means price in dollars
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

    # Player names: resolve after we have data
    pq.player_names = resolve_player_names(message, norm)

    # Best available price: "best price", "best available price", "prices for X FTS"
    if re.search(r"best\s+(available\s+)?price", norm) or re.search(r"price[s]?\s+for\s+.+\s+(fts|ats|lts|fts2h|2\+)", norm):
        pq.best_price_request = True
    if pq.stat_type and pq.player_names and ("price" in norm or "odds" in norm):
        if re.search(r"(what|best|available|current)\s+(is\s+)?(the\s+)?(best\s+)?(price|odds)", norm):
            pq.best_price_request = True

    return pq


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
    """Return list of Position values (e.g. ['2nd Row'] for edge forwards)."""
    out = set()
    for alias, pos in POSITION_ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", norm_text):
            out.add(pos)
    for group_name, positions in POSITION_GROUPS.items():
        if re.search(r"\b" + re.escape(group_name.replace(" ", r"\s+")) + r"\b", norm_text):
            out.update(positions)
    return list(out)


def resolve_player_names(message: str, norm_text: str) -> list[str]:
    """Match player names from message against data. Prefer exact then fuzzy."""
    df_full, df_players, _ = load_data()
    all_names = set()
    if "Player" in df_full.columns:
        all_names.update(df_full["Player"].dropna().astype(str).unique())
    if not df_players.empty and "Player" in df_players.columns:
        all_names.update(df_players["Player"].dropna().astype(str).unique())

    candidates = []
    for name in all_names:
        name_clean = name.strip()
        if len(name_clean) < 3:
            continue
        # exact substring (word boundary) in original message (case-insensitive)
        if re.search(re.escape(name_clean), message, re.I):
            candidates.append((name_clean, len(name_clean)))
    # prefer longer matches
    candidates.sort(key=lambda x: -x[1])
    seen = set()
    result = []
    for name, _ in candidates:
        if name not in seen and not any(name in s and name != s for s in seen):
            seen.add(name)
            result.append(name)
    return result[:10]


def resolve_top_n(pq: ParsedQuery) -> int:
    """Default 5 for 'best' if not specified."""
    return pq.top_n if pq.top_n is not None else 5


def resolve_games_filters(pq: ParsedQuery) -> tuple[int | None, int | None, float | None]:
    """Return (min_games, min_games_since_2024, min_pct)."""
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
    """Aggregate by player_id in range; return total_games, stat_hits, pct, hist_odds."""
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
    """Return list of unique players (player_id, name) sorted by name."""
    df_full, _, _ = load_data()
    if df_full.empty or "Player" not in df_full.columns or "player_id" not in df_full.columns:
        return []
    # Only keep players who have played at least 1 game since the start of the 2024 season
    recent = (
        df_full[df_full["season"] >= 2024]
        .groupby("player_id")["Games played"]
        .sum()
    )
    recent_games = recent.to_dict()

    # One row per player_id, take first Player name
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
    """Return per-season stats for one player: Games played, FTS, FTS historical odds, ATS, etc."""
    df_full, _, _ = load_data()
    sub = df_full[df_full["player_id"] == player_id].copy()
    if sub.empty:
        return []
    # CSV has "2+", "2+ historical odds" - use STAT_2PLUS
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


def _get_live_prices_df() -> pd.DataFrame:
    """Return live prices DataFrame (may be empty). Load data if needed. Used only for value Q when user didn't give price."""
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
    """Return list of {website, price} for player-market. Best-price questions use summary CSVs only."""
    return _get_best_prices_from_summary(player_name, market)


def _get_best_price_for_value(player_name: str, market: str) -> tuple[float | None, str | None]:
    """Get single best price for value question when user didn't give price. Prefer summary CSVs, else live_prices.csv."""
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
    """Format best available prices from data only. Never fabricate."""
    if not offers:
        return f"No price data available for {player_name} {market} in the current data. Prices are only taken from the summary CSVs (e.g. fts_summary.csv, lts_summary.csv); if this player/market is not listed there, we cannot show a price."
    lines = [f"Best available prices for {player_name} {market}:", ""]
    for o in offers:
        lines.append(f"${o['price']:.2f} on {o['website']}")
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
) -> list[dict[str, Any]]:
    """Rank players by percentage; apply all filters; exclude inactive since 2024 for rankings."""
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

    # Exclude players with 0 games since 2024 for ranking
    recent = df_full[df_full["season"] >= 2024].groupby("player_id")["Games played"].sum().reindex(agg["player_id"]).fillna(0).astype(int)
    agg["recent_2024"] = agg["player_id"].map(recent).fillna(0).astype(int)
    agg = agg[agg["recent_2024"] >= (min_games_since_2024 or 0)]
    agg = agg[agg["total_games"] >= (min_games or 0)]
    if min_pct is not None:
        agg = agg[agg["pct"] >= min_pct]

    if positions:
        if df_players.empty or "Position" not in df_players.columns:
            pass  # no position filter
        else:
            # current position from players file (Player column may have extra spaces)
            pos_map = df_players.set_index(df_players["Player"].astype(str).str.strip())["Position"].astype(str).str.strip().to_dict()
            def allowed(name):
                p = pos_map.get(str(name).strip() if isinstance(name, str) else str(name).strip())
                if p is None:
                    return False
                return p in positions
            agg = agg[agg["player_name"].map(allowed)]

    agg = agg[agg["total_games"] > 0]
    agg = agg.sort_values(
        by=["pct", "stat_hits", "total_games", "player_name"],
        ascending=[ascending, False, False, True],
    )
    rows = agg.head(top_n)
    result = []
    for _, r in rows.iterrows():
        name = r["player_name"]
        team, pos = _current_meta(name)
        result.append({
            "player_id": int(r["player_id"]),
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


# 20% benchmark: available price can drop up to 20% below a value price and still be "value"
VALUE_FLOOR_PCT = 0.80


def compute_value_analysis(
    player_id: int,
    stat_type: str,
    season_from: int,
    season_to: int,
    market_odds: float,
) -> dict[str, Any]:
    """Historical prob vs implied prob; edge in percentage points; value ratio (live/historical)."""
    res = compute_player_stats(player_id, stat_type, season_from, season_to)
    total_games = res["total_games"]
    stat_hits = res["stat_hits"]
    hist_prob = (stat_hits / total_games) if total_games else 0.0
    hist_odds = (total_games / stat_hits) if stat_hits else None
    implied_prob = 1.0 / market_odds if market_odds else 0.0
    edge_pct = (hist_prob - implied_prob) * 100
    positive_value = hist_prob > implied_prob
    value_ratio = (market_odds / hist_odds) if (hist_odds and hist_odds > 0 and market_odds) else None
    # 20% rule: still value if price >= this value price * VALUE_FLOOR_PCT
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


def _format_stat_block_lines(
    label: str,
    stat_type: str,
    season_from: int,
    season_to: int,
    stats: dict[str, Any],
) -> list[str]:
    """Build multiline stat block: Player, Stat, Seasons, Games played, [STAT] hits, Rate, Historical odds."""
    odds_str = f"${stats['hist_odds']:.2f}" if stats.get("hist_odds") is not None else "—"
    season_range = f"{season_from}–{season_to}" if season_from != season_to else str(season_from)
    pct = stats["pct"] if isinstance(stats.get("pct"), (int, float)) else 0.0
    return [
        f"Player: {label}",
        f"Stat: {stat_type}",
        f"Seasons: {season_range}",
        f"Games played: {stats['total_games']}",
        f"{stat_type} hits: {stats['stat_hits']}",
        f"Rate: {pct:.2f}%",
        f"Historical odds: {odds_str}",
    ]


def format_single_player_response(
    player_name: str,
    stat_type: str,
    season_from: int,
    season_to: int,
    stats: dict[str, Any],
) -> str:
    """Compact multiline: each metric on its own line."""
    team, pos = _current_meta(player_name)
    label = player_name
    if team or pos:
        parts = [p for p in [team, pos] if p]
        label = f"{player_name} ({', '.join(parts)})"
    lines = _format_stat_block_lines(label, stat_type, season_from, season_to, stats)
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
) -> str:
    """Numbered list with one stat per line; then filter summary."""
    season_range = f"{season_from}–{season_to}" if season_from != season_to else str(season_from)
    lines = [
        f"Stat: {stat_type} (seasons {season_range})",
        "",
    ]
    for i, r in enumerate(rows, 1):
        label = r["player_name"]
        if r.get("team") or r.get("position"):
            parts = [p for p in [r.get("team"), r.get("position")] if p]
            label = f"{r['player_name']} ({', '.join(parts)})"
        odds_str = f"${r['hist_odds']:.2f}" if r.get("hist_odds") is not None else "—"
        lines.append(f"{i}. {label}")
        lines.append(f"   Games: {r['total_games']} · {stat_type}: {r['stat_hits']} ({r['pct']}%) · Hist odds: {odds_str}")
    filters = []
    if positions:
        filters.append(f"position = {', '.join(positions)}")
    filters.append(f"stat = {stat_type}")
    filters.append(f"seasons = {season_range}")
    if min_games is not None:
        filters.append(f"min games = {min_games}")
    if min_games_since_2024 is not None:
        filters.append(f"min games since 2024 = {min_games_since_2024}")
    if min_pct is not None:
        filters.append(f"min rate = {min_pct}%")
    lines.append("")
    lines.append("Filters: " + ", ".join(filters) + ". Inactive since 2024 excluded.")
    return "\n".join(lines)


def format_value_response(
    player_name: str,
    value: dict[str, Any],
    season_from: int,
    season_to: int,
    website: str | None = None,
) -> str:
    """Value comparison: stat block, market odds (with website if provided), implied prob, verdict, value-by-Xx, 20% rule."""
    team, pos = _current_meta(player_name)
    label = player_name
    if team or pos:
        parts = [p for p in [team, pos] if p]
        label = f"{player_name} ({', '.join(parts)})"
    lines = _format_stat_block_lines(
        label, value["stat_type"], season_from, season_to, value
    )
    edge = value["edge_pct_points"]
    if abs(edge) < 0.5:
        verdict = "Fair value"
    elif value["positive_value"]:
        verdict = "Value"
    else:
        verdict = "Not value"
    price_str = f"${value['market_odds']:.2f}"
    if website:
        price_str += f" on {website}"
    lines.append(f"Live price: {price_str}")
    lines.extend([
        f"Implied probability: {value['implied_prob']:.2f}%",
        f"Verdict: {verdict}",
        f"Edge: {edge:+.2f} percentage points",
    ])
    # Always include value-by multiplier and 20% benchmark when we have the numbers (Payne Haas format)
    if value.get("value_ratio") is not None:
        lines.append(f"Value by: {value['value_ratio']:.2f}x (live price vs historical odds)")
    if value.get("value_floor") is not None:
        lines.append(f"Using 20% benchmark: still value down to ${value['value_floor']:.2f}; below that is not value.")
    elif value.get("hist_odds") is not None and value["hist_odds"] > 0 and value["market_odds"]:
        # Show 20% benchmark for non-value / fair value: price would need to be at or above historical for value
        lines.append(f"Using 20% benchmark: would be value at or above ${value['hist_odds']:.2f}; below that is not value.")
    return "\n".join(lines)


def get_chat_response(message: str, history: list[dict]) -> str:
    """Produce full response text (no streaming). Prefer RAG when OPENAI_API_KEY is set; else use rules."""
    load_data()
    # Prefer RAG when available so answers come from RAG pipeline
    try:
        import rag
        if rag.is_rag_available():
            return rag.get_rag_response(message, history)
    except Exception:
        pass
    # Rule-based path when RAG not available or failed
    pq = parse_query(message)
    season_from, season_to = resolve_timeframe(pq)
    norm_msg = normalize_text(message)

    # Best available price: always include website with price (e.g. $81 on Topsport)
    if pq.best_price_request and pq.player_names and pq.stat_type:
        name = pq.player_names[0]
        stat = pq.stat_type or "ATS"
        offers = get_live_prices(name, stat)
        return format_best_prices_response(name, stat, offers)

    # Value question: use market_odds from message or best live price; include website and value-by-Xx, 20% rule
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
            return "Please give a price (e.g. Is $17 for Payne Haas FTS value?, or '31s LTS' for $31) or ensure the player/market appears in the summary CSVs so I can use the best available price."
        value = compute_value_analysis(pid, stat, season_from, season_to, market_odds)
        return format_value_response(name, value, season_from, season_to, website=website)
    elif ("value" in norm_msg or "how much" in norm_msg) and not pq.player_names:
        return "Please specify a player name for value analysis (e.g. Is $17 for Payne Haas FTS value?)."

    # Single or few players
    if pq.player_names and not pq.top_n:
        stat = pq.stat_type or "ATS"
        parts = []
        for name in pq.player_names[:5]:
            df_full, _, _ = load_data()
            match = df_full[df_full["Player"].astype(str).str.strip() == name.strip()]
            if match.empty:
                parts.append(f"No player matched: '{name}'.")
                continue
            pid = int(match.iloc[0]["player_id"])
            stats = compute_player_stats(pid, stat, season_from, season_to)
            parts.append(format_single_player_response(name, stat, season_from, season_to, stats))
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
        )
        if not rows:
            return "No players match all filters. Try relaxing minimum games or position."
        return format_ranking_response(
            rows, stat, season_from, season_to,
            pq.positions, min_g, min_2024, min_pct,
        )

    # Fallback: try to answer with stat type + timeframe
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
                pq.positions, min_g, min_2024, min_pct,
            )
    return "Please ask about a specific player, a ranking (e.g. top 5 edge forwards for LTS since 2022), or a value question (e.g. Is $17 for Payne Haas FTS value?)."


def stream_chat_response(message: str, history: list[dict]):
    """Yield full response as chunks (word/sentence) for SSE. Preserves newlines."""
    full = get_chat_response(message, history)
    lines = full.split("\n")
    for line_idx, line in enumerate(lines):
        words = line.split()
        for i, w in enumerate(words):
            yield w + (" " if i < len(words) - 1 else "")
        if line_idx < len(lines) - 1:
            yield "\n"
