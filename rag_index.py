#!/usr/bin/env python3
"""
Build or refresh the Gemini File Search RAG indexes from NRL tryscorers CSVs.
Run from backend dir: python rag_index.py

Builds two stores:
  - nrl_tryscorers          (legacy, full CSV)
  - nrl_tryscorers_minutesbands  (primary, position + minutes-bands dataset)

Requires GEMINI_API_KEY in .env and tryscorers data in backend/data/.
"""
import os
import sys

# Run from backend directory so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

if not os.environ.get("GEMINI_API_KEY"):
    print("GEMINI_API_KEY not set. Add it to backend/.env and try again.")
    sys.exit(1)

from rag import ensure_index, ensure_minutesbands_index

if __name__ == "__main__":
    print("Building primary index (minutes-bands dataset)...")
    n_mb = ensure_minutesbands_index()
    if n_mb == 0:
        print("  Minutes-bands store already exists and is valid. No re-indexing needed.")
    else:
        print(f"  Indexed {n_mb} chunks into nrl_tryscorers_minutesbands store.")

    print("Building legacy index (full CSV)...")
    n = ensure_index()
    if n == 0:
        print("  Legacy store already exists and is valid. No re-indexing needed.")
    else:
        print(f"  Indexed {n} chunks into nrl_tryscorers store.")
