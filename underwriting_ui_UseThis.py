import json
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any

import streamlit as st
import chromadb
from chromadb.config import Settings

from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd

def render_table(carriers_data, carrier_to_files=None):
    rows = []
    for c in carriers_data:
        name = c.get("name", "")
        rows.append({
            "Carrier": name,
            "Write Risk?": (c.get("eligibility", "") or "").capitalize(),
            "Eligible Lines": ", ".join(c.get("lines", []) or []),
            "Restrictions / Notes": c.get("notes", "") or "",
            "Sources": "; ".join(carrier_to_files.get(name, [])) if carrier_to_files else "",
        })

    df = pd.DataFrame(rows)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

# ---------------- CONFIG ---------------- #

CHROMA_DIR = Path("chroma_underwriting_db")
EMBEDDING_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4.1"  # or gpt-4.1 / gpt-4o

# Known carriers in the corpus (directory names / metadata "company" values)
CARRIER_LIST = [
    "AmTrust",
    "Travelers",
    "MSA",
    "Merchants",
    "CNA",
    "Chautauqua Patrons",
    "Security Mutual",
    "Preferred Mutual",
    "Utica First",
    "Guard",
]

# --------------- INIT CLIENTS --------------- #

load_dotenv()
openai_client = OpenAI()


@st.cache_resource
def get_chroma_collection():
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Chroma directory {CHROMA_DIR} not found. Run build_index.py first."
        )
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(name="underwriting")


# --------------- CORE HELPERS --------------- #

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


def embed_query(text: str) -> List[float]:
    resp = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text]
    )
    return resp.data[0].embedding


def classify_carriers_json(
    risk_desc: str,
    docs: List[str],
    metas: List[dict]
) -> Tuple[Set[str], Set[str], List[Dict[str, Any]], str]:
    """
    GPT call that classifies carriers as yes/maybe/no.
    Returns:
      eligible (yes), maybe, carriers_data(list), summary_markdown
    """

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

    resp = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.1,
    )

    raw = resp.choices[0].message.content.strip()

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


def build_table_markdown(carriers_data: List[Dict[str, Any]]) -> str:
    """
    Build the markdown table like your console output.
    """
    lines = []
    lines.append("| Carrier           | Write Risk? | Eligible Lines                          | Restrictions / Notes |")
    lines.append("|-------------------|-------------|-----------------------------------------|-----------------------|")

    for c in carriers_data:
        name = c.get("name", "")
        elig = (c.get("eligibility", "") or "").capitalize()
        lines_list = c.get("lines", []) or []
        notes = c.get("notes", "") or ""

        lines_str = ", ".join(lines_list) if lines_list else "-"

        row_notes = notes.replace("\n", " ")
        if len(row_notes) > 120:
            row_notes = row_notes[:117] + "..."

        lines.append(
            f"| {name:<17} | {elig:<11} | {lines_str:<37} | {row_notes} |"
        )

    return "\n".join(lines)


# --------------- STREAMLIT APP --------------- #

def main():
    st.set_page_config(page_title="Underwriting Screener", layout="wide")
    st.title("Underwriting Screener + Follow-up Chat")

    collection = get_chroma_collection()

    # ---- INIT SESSION STATE ----
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "eligible_carriers" not in st.session_state:
        st.session_state.eligible_carriers = []
    if "static_context" not in st.session_state:
        st.session_state.static_context = ""
    if "risk_desc" not in st.session_state:
        st.session_state.risk_desc = ""

    # ---- Questionnaire Form ----
    st.subheader("Risk Questionnaire")

    col1, col2 = st.columns(2)

    with col1:
        first_name = st.text_input("First Name", value="Al")
        last_name = st.text_input("Last Name", value="Rivera")
        company_name = st.text_input("Company Name", value="Als Diner")
        company_size = st.text_input("Company Size", value="10")

    with col2:
        industry = st.text_input("Industry", value="Hospitality")
        business = st.text_input("Business", value="Bar Resturant")
        res_com = st.selectbox(
            "Residental or Commercial",
            options=["Commercial", "Residential", "Both"],
            index=0,
        )

    # ---- Screen Risk Button ----
    if st.button("Screen Risk"):
        # New questionnaire for THIS risk
        questionnaire = {
            "First Name": first_name,
            "Last Name": last_name,
            "Company Name": company_name,
            "Company Size": company_size,
            "Industry": industry,
            "Business": business,
            "Residental or Commercial": res_com,
        }

        # Build and store risk description
        risk_desc = build_risk_description(questionnaire)
        st.session_state.risk_desc = risk_desc

        # Clear previous conversation whenever we screen a new risk
        st.session_state.chat_messages = []
        st.session_state.eligible_carriers = []
        st.session_state.static_context = ""

        st.markdown("### Risk Description")
        st.write(risk_desc)

        # --- Embed the risk description ---
        q_emb = embed_query(risk_desc)

        # We want: top K per carrier + fill remainder up to TOTAL_TARGET from global search
        PER_CARRIER_TOP_K = 5
        TOTAL_TARGET = 100

        docs_all: List[str] = []
        metas_all: List[dict] = []

        # For deduping (since we don't have IDs in include, use a composite key)
        seen_keys = set()

        def add_results(docs: List[str], metas: List[dict]):
            for d, m in zip(docs, metas):
                key = (
                    m.get("company", ""),
                    m.get("filename", ""),
                    m.get("chunk_index", ""),
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                docs_all.append(d)
                metas_all.append(m)

        # --- 1) Top-K per known carrier ---
        for carrier in CARRIER_LIST:
            try:
                res_carrier = collection.query(
                    query_embeddings=[q_emb],
                    n_results=PER_CARRIER_TOP_K,
                    include=["documents", "metadatas"],
                    where={"company": carrier},
                )
            except Exception:
                continue  # if carrier has no vectors, skip

            docs_c = res_carrier["documents"][0]
            metas_c = res_carrier["metadatas"][0]
            add_results(docs_c, metas_c)

        # --- 2) Global fill to reach TOTAL_TARGET ---
        remaining = max(TOTAL_TARGET - len(docs_all), 0)
        if remaining > 0:
            res_global = collection.query(
                query_embeddings=[q_emb],
                n_results=remaining,
                include=["documents", "metadatas"],
                where=None,
            )
            docs_g = res_global["documents"][0]
            metas_g = res_global["metadatas"][0]
            add_results(docs_g, metas_g)

        st.info(
            f"Retrieved {len(docs_all)} relevant chunks from Chroma "
            f"(per-carrier top {PER_CARRIER_TOP_K} + global fill)."
        )

        # Show which carriers appeared
        carriers_present = sorted({m.get("company", "UNKNOWN") for m in metas_all})
        st.markdown(
            "**Carriers present in retrieved chunks:**  " + ", ".join(carriers_present)
        )

        # 🔹 Build carrier -> files map from all retrieved metas
        carrier_to_files_raw: Dict[str, set] = {}
        for m in metas_all:
            carrier = m.get("company", "UNKNOWN")
            rel_path = m.get("relative_path") or m.get("filename", "")
            if not rel_path:
                continue
            carrier_to_files_raw.setdefault(carrier, set()).add(rel_path)

        # --- Classify carriers (yes/maybe/no) ---
        eligible, maybe, carriers_data, summary = classify_carriers_json(
            risk_desc, docs_all, metas_all
        )

        eligible_union = eligible | maybe
        st.session_state.eligible_carriers = sorted(list(eligible_union))

        # Restrict sources to just carriers in YES/MAYBE
        carrier_to_files: Dict[str, List[str]] = {
            carrier: sorted(files)
            for carrier, files in carrier_to_files_raw.items()
            if carrier in eligible_union
        }

        # Show table + notes
        st.markdown("### Underwriting Eligibility Summary")
        render_table(carriers_data, carrier_to_files)

        if summary:
            st.markdown("---")
            st.markdown("#### Notes")
            st.markdown(summary)

        st.markdown("---")
        if eligible_union:
            st.markdown("**Carriers with YES / MAYBE:**")
            st.write(", ".join(sorted(eligible_union)))
        else:
            st.warning("No carriers were classified as YES / MAYBE for this risk.")

        # --- Build filtered static context for follow-up chat ---
        filtered_docs: List[str] = []
        filtered_metas: List[dict] = []

        for d, m in zip(docs_all, metas_all):
            carrier = m.get("company", "UNKNOWN")
            if carrier in eligible_union:
                filtered_docs.append(d)
                filtered_metas.append(m)

        context_blocks = []
        for d, m in zip(filtered_docs, filtered_metas):
            carrier = m.get("company", "UNKNOWN")
            fname = m.get("filename", "")
            context_blocks.append(f"[Company: {carrier} | File: {fname}]\n{d}")

        static_context = "\n\n---\n\n".join(context_blocks)
        st.session_state.static_context = static_context

    st.markdown("---")

    # ---- Follow-up Chat (only if we have context from last Screen Risk) ----
    if st.session_state.static_context and st.session_state.eligible_carriers:
        st.subheader("Follow-up Questions (Eligible Carriers Only)")

        st.caption(
            "Answering using **only** these carriers: "
            + ", ".join(st.session_state.eligible_carriers)
        )

        # Render chat history
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("Ask a question about coverage for this risk...")
        if user_input:
            # Add user message
            st.session_state.chat_messages.append(
                {"role": "user", "content": user_input}
            )

            carriers_list_str = ", ".join(st.session_state.eligible_carriers)
            system_prompt = (
                "You are an underwriting assistant.\n\n"
                f"The ONLY carriers to consider in this conversation are: {carriers_list_str}.\n"
                "You have a fixed risk profile and a fixed set of underwriting excerpts "
                "for those carriers.\n\n"
                "Rules:\n"
                "- Answer ALL questions using ONLY these excerpts; do not invent coverage.\n"
                "- If the excerpts don't say something, say 'not specified in the documents'.\n"
                "- Make it clear in your answers which carrier(s) you are talking about.\n"
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"RISK PROFILE:\n{st.session_state.risk_desc}\n\n"
                        f"UNDERWRITING EXCERPTS (eligible carriers only):\n\n"
                        f"{st.session_state.static_context}"
                    ),
                },
            ]

            # Append conversation turns for THIS risk only
            for msg in st.session_state.chat_messages:
                messages.append({"role": msg["role"], "content": msg["content"]})

            resp = openai_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                temperature=0.2,
            )
            answer = resp.choices[0].message.content.strip()

            st.session_state.chat_messages.append(
                {"role": "assistant", "content": answer}
            )

            with st.chat_message("assistant"):
                st.markdown(answer)

    else:
        st.info(
            "Fill out the questionnaire and click **Screen Risk** to start, "
            "then you can ask follow-up questions."
        )


if __name__ == "__main__":
    main()
