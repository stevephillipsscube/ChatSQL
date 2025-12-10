from pathlib import Path
import chromadb
from chromadb.config import Settings

CHROMA_DIR = Path("chroma_underwriting_db")

def main():
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(name="underwriting")

    # 1) How many vectors exist total?
    count = collection.count()
    print(f"Collection 'underwriting' has {count} vectors")

    # 2) Run a dummy query to see how many neighbors we actually get back
    #    Use a zero-vector with the same dimension as your embeddings (3072)
    dummy_emb = [0.0] * 3072

    res = collection.query(
        query_embeddings=[dummy_emb],
        n_results=20,
        include=["metadatas"],   # any valid include; ids are always present
    )

    ids = res["ids"][0]
    print(f"Dummy query asked for 20, Chroma returned {len(ids)} results")

if __name__ == "__main__":
    main()
