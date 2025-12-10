# screen_risk.py
#
# Given a questionnaire (key/value info about a business),
# look up underwriting docs and return which carriers will / won't cover it.

from typing import Dict, List
from pathlib import Path

import chromadb
from chromadb.config import Settings

from dotenv import load_dotenv
from openai import OpenAI

# ---- Config (reuse same settings as query_underwriting.py) ----

CHROMA_DIR = Path("chroma_underwriting_db")

EMBEDDING_MODEL = "text-embedding-3-large"  # must match ETL
CHAT_MODEL = "gpt-4.1-mini"                 # or gpt-4.1 / gpt-4o, etc.

# ---- Init OpenAI + env ----

load_dotenv()
client = OpenAI()


# === Helper: build a natural-language risk description from questionnaire ===

def build_risk_description(q: Dict[str, str]) -> str:
    """
    Turn a questionnaire dict into a compact natural-language summary
    that will work well as an embedding/query.
    """
    first = q.get("First Name", "").strip()
    last = q.get("Last Name", "").strip()
    name = " ".join([p for p in [first, last] if p])

    company = q.get("Company Name", "").strip()
    size = q.get("Company Size", "").strip()
    industry = q.get("Industry", "").strip()
    business = q.get("Business", "").strip()
    res_com = q.get("Residental or Commercial", q.get("Residential or Commercial", "")).strip()

    # You can extend this later with address, revenues, class code, etc.
    parts = []
    if name:
        parts.append(f"Contact name: {name}.")
    if company:
        parts.append(f"Company name: {company}.")
    if size:
        parts.append(f"Number of employees: {size}.")
    if industry:
        parts.append(f"Industry: {industry}.")
    if business:
        parts.append(f"Business type: {business}.")
    if res_com:
        parts.append(f"Operations are primarily {res_com.lower()}.")

    # High-level underwriting question:
    parts.append(
        "Question: Based on the underwriting guidelines, which insurance carriers "
        "will write this risk, and under which lines (BOP, package, WC, umbrella, etc.)? "
        "Also identify carriers that explicitly exclude or decline this class."
    )

    return " ".join(parts)


# === Embedding function (MUST match ETL) ===

def embed_query(text: str) -> List[float]:
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text]
    )
    return resp.data[0].embedding


# === LLM synthesis: decide who will cover it ===

def summarize_eligibility(questionnaire: Dict[str, str],
                          docs: List[str],
                          metadatas: List[dict]) -> str:
    """
    Ask GPT to read retrieved underwriting chunks and decide
    which carriers are likely to cover / maybe / not cover.
    """
    risk_desc = build_risk_description(questionnaire)

    # Build context blocks with carrier + file info
    ctx_blocks = []
    for doc, meta in zip(docs, metadatas):
        carrier = meta.get("company", "UNKNOWN")
        filename = meta.get("filename", "")
        ctx_blocks.append(
            f"[Company: {carrier} | File: {filename}]\n{doc}"
        )

    context_text = "\n\n---\n\n".join(ctx_blocks)

    system_prompt = (
        "You are an underwriting assistant. You are given:\n"
        "1) A description of a specific risk (business).\n"
        "2) Excerpts from carrier appetite guides and underwriting manuals.\n\n"
        "Your job is to determine, for EACH carrier mentioned in the excerpts:\n"
        "- Will they write this risk? (Yes / Maybe / No)\n"
        "- Under which lines (BOP, package, GL, property, WC, umbrella, etc.)\n"
        "- Any important restrictions or exclusions (e.g., roof % limits, residential vs commercial only).\n\n"
        "Rules:\n"
        "- ONLY use the information in the provided excerpts; do not guess beyond them.\n"
        "- If the excerpts clearly exclude the class, mark the carrier as 'No' and explain briefly.\n"
        "- If the excerpts clearly include the class, mark 'Yes' and list the eligible lines.\n"
        "- If it's ambiguous, mark 'Maybe' and explain why.\n"
        "- If a carrier is not mentioned at all in the excerpts, do NOT speculate about them.\n"
        "- Return a concise, readable summary, preferably as a table and then bullet notes."
    )

    user_content = (
        f"RISK DESCRIPTION:\n{risk_desc}\n\n"
        f"UNDERWRITING EXCERPTS:\n\n{context_text}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )

    return resp.choices[0].message.content.strip()


# === Main: screen one risk ===

def screen_risk(questionnaire: Dict[str, str]) -> None:
    """
    High-level function:
    - Embed the risk
    - Query Chroma for relevant underwriting chunks
    - Ask GPT to summarize which carriers will / won't cover it
    """
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Chroma directory {CHROMA_DIR} not found. "
            f"Run build_index.py after generating embeddings_output.jsonl."
        )

    client_chroma = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client_chroma.get_collection(name="underwriting")

    risk_desc = build_risk_description(questionnaire)
    print("=== RISK DESCRIPTION ===")
    print(risk_desc)
    print()

    # 1) Embed the risk description
    q_emb = embed_query(risk_desc)

    # 2) Query Chroma for a larger batch to improve recall
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=40,  # pull a lot so multiple carriers show up
        include=["documents", "metadatas", "distances"],
        where=None,   # no company filter – we want everyone
    )

    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    print(f"[INFO] Retrieved {len(docs)} relevant chunks from Chroma.\n")

    # Optional: show a quick carrier summary before LLM
    carriers_seen = sorted({m.get("company", "UNKNOWN") for m in metadatas})
    print("Carriers present in retrieved chunks:")
    for c in carriers_seen:
        print(f" - {c}")
    print()

    # 3) Ask GPT to summarize eligibility across carriers
    summary = summarize_eligibility(questionnaire, docs, metadatas)

    print("=== UNDERWRITING ELIGIBILITY SUMMARY ===")
    print(summary)


if __name__ == "__main__":
    # Example: your Al's Roofing questionnaire
    questionnaire_example = {
        "First Name": "Al",
        "Last Name": "Rivera",
        "Company Name": "Als Liquor",
        "Company Size": "20",
        "Industry": "Hospitality",
        "Business": "Bar",
        "Residental or Commercial": "Commercial",
    }

    screen_risk(questionnaire_example)
