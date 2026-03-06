"""
Quick tests for chat: "31s" price parsing, value format, and best price from summary CSVs only.
Requires backend/data (or repo data/) with Nrl_tryscorers_2020_2025_full.csv. Summary CSVs
(fts_summary.csv etc.) optional for best-price tests.
Run from backend: python test_chat.py
"""
import os
import sys

# run from backend directory so data/ is found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tryscorers_chat as chat


def test_31s_parsed_as_price():
    """'31s', '41s', '81s' should be parsed as $31, $41, $81."""
    for msg, expected in [
        ("Is Braydon Trindall 31s LTS good value?", 31.0),
        ("41s FTS value?", 41.0),
        ("Payne Haas 81s ATS", 81.0),
        ("what about 17.5s for LTS", 17.5),
    ]:
        pq = chat.parse_query(msg)
        assert pq.market_odds == expected, f"message {msg!r}: expected market_odds={expected}, got {pq.market_odds}"


def test_31s_lts_value_response_format():
    """Value answer for '31s LTS' must include value-by multiplier and 20% benchmark (Payne Haas format)."""
    msg = "Is Braydon Trindall 31s LTS good value?"
    response = chat.get_chat_response(msg, [])
    assert "Value by:" in response or "value by" in response.lower(), (
        f"Expected 'Value by:' in value response. Got: {response[:500]}..."
    )
    assert "20% benchmark" in response, (
        f"Expected '20% benchmark' in value response. Got: {response[:500]}..."
    )
    assert "31" in response, "Expected live price 31 in response"


def test_best_price_request_parsed():
    """'Best price for X FTS' should set best_price_request and stat FTS."""
    msg = "Best price for Payne Haas FTS?"
    pq = chat.parse_query(msg)
    assert pq.best_price_request is True, f"Expected best_price_request=True for {msg!r}"
    assert (pq.stat_type or "").upper() == "FTS", f"Expected stat_type FTS, got {pq.stat_type}"
    assert pq.player_names, f"Expected player name parsed, got {pq.player_names}"


def test_best_price_response_from_data_only():
    """Best price answer must be from summary CSVs only; if unavailable, say so (no fabrication)."""
    msg = "Best price for Payne Haas FTS?"
    response = chat.get_chat_response(msg, [])
    # Either we have real prices (from summary CSV) or the explicit "no data" message
    if "No price data available" in response:
        assert "summary" in response.lower() or "cannot show" in response.lower() or "not listed" in response.lower(), (
            "When no price data, message should mention summary data / not available."
        )
    else:
        # Should list dollar amounts and a bookmaker name, not made-up numbers
        assert "Best available prices" in response or "best available" in response.lower()
        assert "$" in response
        assert " on " in response, "Prices should be 'X on Website' format"


if __name__ == "__main__":
    test_31s_parsed_as_price()
    print("test_31s_parsed_as_price passed")
    test_31s_lts_value_response_format()
    print("test_31s_lts_value_response_format passed")
    test_best_price_request_parsed()
    print("test_best_price_request_parsed passed")
    test_best_price_response_from_data_only()
    print("test_best_price_response_from_data_only passed")
    print("All tests passed.")
