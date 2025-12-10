# screen_risk_chat.py
#
# Stateful underwriting risk screening:
# 1) Use all carriers to decide who will cover the risk (table + notes).
# 2) For follow-up questions, ONLY use carriers classified as Yes/Maybe.

from typing import Dict, List, Tuple, Set, Any
from pathlib import Path
import json

import chromadb
from chromadb.config import Settings

from dotenv import load_dotenv
from openai import OpenAI

# ---- Config ----

CHROMA_DIR = Path("chroma_underwriting_db")
EMBEDDING_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4.1-mini"  # or gpt-4.1 / gpt-4o


# ---- Init ----

load_dotenv()
client = OpenAI()


# -----------------------------------------------------
# Build natural-language risk description
# -----------------------------------------------------

def build_risk_description(q: Dict[str, str]) -> str:
    parts = []

    first = q.get("First Name", "").strip()
    last = q.get("Last Name", "").strip()
    if first or last:
        parts.append(f"Contact name: {first} {last}".strip() + ".")

    company = q.get("Company Name", "").strip()
    if company:
        parts.append(f"Company name: {company}.")

    size = q.get("Company Size", "").strip()
    if size:
        parts.append(f"Number of employees: {size}.")

    industry = q.get("Industry", "").strip()
    if industry:
        parts.append(f"Industry: {industry}.")

    business = q.get("Business", "").strip()
    if business:
        parts.append(f"Business type: {business}.")

    res_com = q.get("Residental or Commercial", q.get("Residential or Commercial", "")).strip()
    if res_com:
        parts.append(f"Operations are primarily {res_com.lower()}.")

    parts.append(
        "Question: Based on the underwriting guidelines, which insurance carriers will write this risk, "
        "and under which lines (BOP, package, WC, umbrella, etc.)? Also identify carriers that explicitly "
        "exclude or decline this class."
    )

    return " ".join(parts)


# -----------------------------------------------------
# Embedding
# -----------------------------------------------------

def embed_query(text: str) -> List[float]:
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text]
    )
    return resp.data[0].embedding


# -----------------------------------------------------
# First GPT call: classify carriers as Yes / Maybe / No (JSON)
# -----------------------------------------------------

def classify_carriers_json(
    risk_desc: str,
    docs: List[str],
    metas: List[dict]
) -> Tuple[Set[str], Set[str], List[Dict[str, Any]], str]:
    """
    Ask GPT to read all retrieved chunks and return a JSON classification
    of carriers: Yes / Maybe / No.

    Returns:
      (eligible_carriers, maybe_carriers, carriers_list, summary_markdown)

      carriers_list is a list of:
        {
          "name": str,
          "eligibility": "yes" | "maybe" | "no",
          "lines": [str],
          "notes": str
        }
    """

    # Build context with carrier + file info
    ctx_blocks = []
    for d, m in zip(docs, metas):
        carrier = m.get("company", "UNKNOWN")
        fname = m.get("filename", "")
        ctx_blocks.append(f"[Company: {carrier} | File: {fname}]\n{d}")

    context_text = "\n\n---\n\n".join(ctx_blocks)

    system_prompt = (
        "You are an underwriting assistant. You will be given:\n"
        "1) A description of a specific insurance risk.\n"
        "2) Excerpts from multiple carriers' underwriting manuals and appetite guides.\n\n"
        "Your job in THIS STEP is ONLY to classify each carrier as:\n"
        "  - 'yes' (will write this risk),\n"
        "  - 'maybe' (unclear / conditional),\n"
        "  - 'no' (will not write this risk).\n\n"
        "You MUST return STRICT JSON with this exact schema and nothing else:\n"
        "{\n"
        "  \"summary_markdown\": \"human readable summary of your reasoning (markdown ok)\",\n"
        "  \"carriers\": [\n"
        "    {\n"
        "      \"name\": \"Carrier Name\",\n"
        "      \"eligibility\": \"yes|maybe|no\",\n"
        "      \"lines\": [\"BOP\", \"GL\", \"WC\", ...],\n"
        "      \"notes\": \"short explanation from the docs\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Do NOT wrap the JSON in markdown fences. Do NOT add commentary outside the JSON."
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
        temperature=0.1,
    )

    raw = resp.choices[0].message.content.strip()

    # Extract JSON object defensively
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Expected JSON object, got:\n{raw}")

    json_str = raw[start:end+1]
    data = json.loads(json_str)

    carriers_data: List[Dict[str, Any]] = data.get("carriers", [])
    summary_markdown: str = data.get("summary_markdown", "")

    eligible: Set[str] = set()
    maybe: Set[str] = set()

    for c in carriers_data:
        name = (c.get("name") or "").strip()
        elig = (c.get("eligibility") or "").lower()
        if not name:
            continue
        if elig == "yes":
            eligible.add(name)
        elif elig == "maybe":
            maybe.add(name)

    return eligible, maybe, carriers_data, summary_markdown


# -----------------------------------------------------
# Pretty-print classification like your previous output
# -----------------------------------------------------

def print_eligibility_table(carriers_data: List[Dict[str, Any]], summary_markdown: str) -> None:
    """
    Format the carrier classification in a table + notes,
    matching the style you liked from query_underwriting.py.
    """

    print("=== UNDERWRITING ELIGIBILITY SUMMARY ===")

    # Table header
    print("| Carrier           | Write Risk? | Eligible Lines                          | Restrictions / Notes |")
    print("|-------------------|-------------|-----------------------------------------|-----------------------|")

    for c in carriers_data:
        name = c.get("name", "")
        elig = (c.get("eligibility", "") or "").capitalize()
        lines = c.get("lines", []) or []
        notes = c.get("notes", "") or ""

        lines_str = ", ".join(lines) if lines else "-"

        # Truncate notes a bit for table row, full details can be in Notes section
        row_notes = notes.replace("\n", " ")
        if len(row_notes) > 120:
            row_notes = row_notes[:117] + "..."

        print(f"| {name:<17} | {elig:<11} | {lines_str:<37} | {row_notes} |")

    print("\n---\n")
    if summary_markdown:
        print("### Notes:\n")
        print(summary_markdown)
        print()


# -----------------------------------------------------
# Follow-up chat: only use eligible carriers' chunks
# -----------------------------------------------------

def run_chat_session(questionnaire: Dict[str, str]) -> None:
    # ----- Connect to Chroma -----
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            "Chroma index not found. Run build_index.py first."
        )

    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = chroma_client.get_collection(name="underwriting")

    # ----- Risk description -----
    risk_desc = build_risk_description(questionnaire)
    print("=== RISK DESCRIPTION ===")
    print(risk_desc)
    print()

    # ----- Retrieve from ALL carriers for the initial classification -----
    emb = embed_query(risk_desc)

    res = collection.query(
        query_embeddings=[emb],
        n_results=60,                     # big grab for high recall
        include=["documents", "metadatas"],
        where=None,                      # NO filter here
    )

    docs_all = res["documents"][0]
    metas_all = res["metadatas"][0]

    print(f"[INFO] Retrieved {len(docs_all)} relevant chunks from Chroma.\n")

    carriers_seen = sorted({m.get("company", "UNKNOWN") for m in metas_all})
    print("Carriers present in retrieved chunks:")
    for c in carriers_seen:
        print(f" - {c}")
    print()

    # ----- First GPT pass: classify carriers (Yes / Maybe / No) -----
    eligible, maybe, carriers_data, summary = classify_carriers_json(
        risk_desc, docs_all, metas_all
    )

    eligible_union = eligible | maybe

    # Pretty-print like your earlier example
    print_eligibility_table(carriers_data, summary)

    print("Carriers with YES / MAYBE:")
    for c in sorted(eligible_union):
        print(f" - {c}")
    print()

    if not eligible_union:
        print("No carriers were classified as yes/maybe. Nothing to chat about.")
        return

    # ----- Filter docs/metas to ONLY eligible carriers -----
    filtered_docs: List[str] = []
    filtered_metas: List[dict] = []

    for d, m in zip(docs_all, metas_all):
        carrier = m.get("company", "UNKNOWN")
        if carrier in eligible_union:
            filtered_docs.append(d)
            filtered_metas.append(m)

    print(f"[INFO] Retaining {len(filtered_docs)} chunks for follow-up chat "
          f"from carriers: {', '.join(sorted(eligible_union))}\n")

    # Build static context from filtered chunks
    context_blocks = []
    for d, m in zip(filtered_docs, filtered_metas):
        carrier = m.get("company", "UNKNOWN")
        fname = m.get("filename", "")
        context_blocks.append(f"[Company: {carrier} | File: {fname}]\n{d}")

    STATIC_CONTEXT = "\n\n---\n\n".join(context_blocks)

    # ----- System prompt for follow-up chat -----
    carriers_list_str = ", ".join(sorted(eligible_union))

    SYSTEM_PROMPT = (
        "You are an underwriting assistant.\n\n"
        f"The ONLY carriers to consider in this conversation are: {carriers_list_str}.\n"
        "You have a fixed risk profile and a fixed set of underwriting excerpts "
        "for those carriers.\n\n"
        "Rules:\n"
        "- Answer ALL questions using ONLY these excerpts; do not invent coverage.\n"
        "- If the excerpts don't say something, say 'not specified in the documents'.\n"
        "- Make it clear in your answers which carrier(s) you are talking about.\n"
    )

    # Initial conversation history for follow-ups
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"RISK PROFILE:\n{risk_desc}\n\n"
                f"UNDERWRITING EXCERPTS (eligible carriers only):\n\n{STATIC_CONTEXT}"
            ),
        },
    ]

    print("You can now ask follow-up questions about coverage for this risk.")
    print("These answers will consider ONLY carriers classified as Yes/Maybe.")
    print("Type 'exit' to quit.\n")

    # REPL loop: each question uses the same cached STATIC_CONTEXT
    while True:
        q = input("> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("\nSession ended.")
            break

        messages.append({"role": "user", "content": q})

        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.2,
        )

        answer = resp.choices[0].message.content.strip()
        print("\n" + answer + "\n")

        messages.append({"role": "assistant", "content": answer})


# -----------------------------------------------------
# Example invocation
# -----------------------------------------------------

if __name__ == "__main__":
    # Example: Al's Roofing
    risk_form = {
        "First Name": "Al",
        "Last Name": "Rivera",
        "Company Name": "Als Roofing",
        "Company Size": "10",
        "Industry": "Construction",
        "Business": "Roofing",
        "Residental or Commercial": "Commercial",
    }

    run_chat_session(risk_form)
