#!/usr/bin/env python3
"""
Build or refresh the Gemini File Search RAG index from NRL tryscorers CSV.
Run from backend dir: python rag_index.py
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

from rag import ensure_index

if __name__ == "__main__":
    n = ensure_index()
    if n == 0:
        print("File Search store already exists and is valid. No re-indexing needed.")
    else:
        print(f"Indexed {n} chunks into Gemini File Search store.")
