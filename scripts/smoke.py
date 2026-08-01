"""Manual end-to-end smoke script for Phase 1.

Requires the API reachable on the host at :8000 (`uvicorn prorag.main:app`
directly, or `docker compose up postgres` + running the api outside compose —
the full `docker compose up` stack no longer publishes 8000, only caddy's
80/443) against a real Postgres with the migration applied, and a working
LiteLLM provider (OPENAI_API_KEY or equivalent set in .env).

Usage:
    python scripts/smoke.py path/to/some.pdf "What does the document say about X?"
"""

import sys
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000"


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: python scripts/smoke.py <file> <question>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    question = sys.argv[2]

    print("1) healthz")
    r = httpx.get(f"{BASE_URL}/healthz", timeout=10)
    r.raise_for_status()
    print("   ", r.json())

    print("2) ingest", file_path)
    with open(file_path, "rb") as f:
        r = httpx.post(
            f"{BASE_URL}/ingest",
            files={"file": (file_path.name, f)},
            timeout=120,
        )
    r.raise_for_status()
    ingest_body = r.json()
    print("   ", ingest_body)
    doc_id = ingest_body["doc_id"]

    print("3) chat:", question)
    r = httpx.post(f"{BASE_URL}/chat", json={"message": question}, timeout=120)
    r.raise_for_status()
    chat_body = r.json()
    print("    answer:", chat_body["answer"])
    print("    sources:", chat_body["sources"])

    print("4) fetch original file for doc", doc_id)
    r = httpx.get(f"{BASE_URL}/files/{doc_id}/original", timeout=30)
    r.raise_for_status()
    print("   ", len(r.content), "bytes,", r.headers.get("content-type"))

    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
