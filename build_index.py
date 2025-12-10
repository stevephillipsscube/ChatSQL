# build_index.py

import json
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings

# Paths
JSONL_PATH = Path("embeddings_output.jsonl")
CHROMA_DIR = Path("chroma_underwriting_db")


def load_records(jsonl_path: Path) -> List[Dict[str, Any]]:
    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            records.append(rec)
    return records


def main():
    if not JSONL_PATH.exists():
        raise FileNotFoundError(f"{JSONL_PATH} not found")

    print(f"[INFO] Loading records from {JSONL_PATH}...")
    records = load_records(JSONL_PATH)
    print(f"[INFO] Loaded {len(records)} records.")

    if not records:
        print("[INFO] No records to index, exiting.")
        return

    # Initialize Chroma client (local persistent DB)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False)
    )

    # Create or get collection
    collection = client.get_or_create_collection(
        name="underwriting",
        metadata={"hnsw:space": "cosine"}
    )

    # Prepare data for Chroma
    ids = []
    texts = []
    metadatas = []
    embeddings = []

    for rec in records:
        ids.append(rec["id"])
        texts.append(rec["text"])
        metadatas.append(rec.get("metadata", {}))
        embeddings.append(rec["embedding"])

    # Chroma API: add with precomputed embeddings
    print("[INFO] Adding vectors to Chroma collection 'underwriting'...")
    # If you re-run and get duplicate IDs, you can use upsert instead
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"[INFO] Indexed {len(ids)} documents into Chroma at {CHROMA_DIR}.")


if __name__ == "__main__":
    main()
