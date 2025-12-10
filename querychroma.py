from pathlib import Path
import chromadb
from chromadb.config import Settings

CHROMA_DIR = Path("chroma_underwriting_db")

def main():
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(name="underwriting")

    # First: inspect metadata keys
    sample = collection.get(limit=1, include=["metadatas"])
    print("Sample metadata keys:", sample["metadatas"][0].keys())

    # Now: find the row with Class Code 54121
    res = collection.get(
        where={"Class Code": 54121},        # note: int, not string
        include=["documents", "metadatas"],
        limit=10,
    )

    print(f"Found {len(res['ids'])} records with Class Code 54121\n")

    for i, (doc, meta) in enumerate(
        zip(res["documents"], res["metadatas"]), start=1
    ):
        print(f"#{i}")
        print("  Company:", meta.get("company"))
        print("  File:   ", meta.get("filename"))
        print("  Sheet:  ", meta.get("sheet"))
        print("  Row:    ", meta.get("row_index"))
        print("  Business:", meta.get("Business"))
        print("  Class Code:", meta.get("Class Code"))
        print("  --- snippet ---")
        snippet = doc[:400].replace("\n", " ")
        print(" ", snippet, "\n")

if __name__ == "__main__":
    main()
