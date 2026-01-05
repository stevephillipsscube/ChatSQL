import json
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
import html

import streamlit as st
import chromadb
from chromadb.config import Settings
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI

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
        temperature=0.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        seed=42
    )

    raw = resp.choices[0].message.content.strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        # Fallback: sometimes models wrap in ```json ... ```
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


def identify_ineligible_carriers(
    chat_history: List[Dict[str, str]],
    current_eligible_carriers: List[str],
    context_text: str
) -> List[str]:
    """
    Asks the LLM to identify which of the 'current_eligible_carriers' are definitively
    ineligible based on the 'chat_history' constraints (e.g. alcohol %).
    """
    
    # Convert chat history to a string block
    history_block = ""
    for msg in chat_history:
        role = msg["role"].upper()
        content = msg["content"]
        history_block += f"{role}: {content}\n"

    system_prompt = (
        "You are an expert underwriting analyst helper.\n"
        "Your goal is to identifying carriers that are NO LONGER ELIGIBLE based on new constraints revealed in the conversation.\n\n"
        "You have:\n"
        "1. A list of currently eligible carriers.\n"
        "2. The text excerpts (guidelines) for these carriers.\n"
        "3. The conversation history where the user might have added new constraints (e.g. 'Alcohol sales are 45%', 'There is a dance floor').\n\n"
        "TASK:\n"
        "- Review the conversation for any NEW facts that pose a hard knockout.\n"
        "- Compare these facts against the provided excerpts.\n"
        "- Return a JSON object with a list of 'ineligible_carriers' names.\n"
        "- STRICT RULE: Only list a carrier if it is explicitly disqualified by the new facts. If uncertain, leave it eligible.\n"
        "- If no carriers are disqualified, return an empty list.\n\n"
        "JSON SCHEMA:\n"
        "{\n"
        "  \"ineligible_carriers\": [\"Carrier A\", \"Carrier B\"]\n"
        "}"
    )

    user_content = (
        f"CURRENT ELIGIBLE CARRIERS: {', '.join(current_eligible_carriers)}\n\n"
        f"UNDERWRITING GUIDELINES:\n{context_text}\n\n"
        f"CONVERSATION HISTORY:\n{history_block}\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o", # Use a smart model for reasoning
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        raw = resp.choices[0].message.content.strip()
        data = json.loads(raw)
        return data.get("ineligible_carriers", [])
    except Exception as e:
        # On fail, don't filter anything to be safe
        print(f"Filter error: {e}")
        return []



def render_html_table(carriers_data, carrier_to_files=None):
    """
    Renders a custom HTML table to allow full control over wrapping, column widths,
    and clickable file links.
    """
    if not carriers_data:
        st.write("No carrier data available.")
        return

    # Start HTML table with styles
    html_str = """
<style>
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        font-family: sans-serif;
        font-size: 0.95rem;
        color: #333;
    }
    .custom-table th {
        background-color: #f0f2f6;
        color: #31333F;
        font-weight: 600;
        text-align: left;
        padding: 12px 8px;
        border-bottom: 2px solid #ddd;
    }
    .custom-table td {
        padding: 10px 8px;
        border-bottom: 1px solid #eee;
        vertical-align: top;
        line-height: 1.5;
    }
    .custom-table tr:hover {
        background-color: #f9f9f9;
    }
    
    /* Column Widths */
    .col-carrier { width: 15%; }
    .col-eligibility { width: 10%; }
    .col-lines { width: 25%; }
    .col-notes { width: 35%; /* Rest of the logic handles wrap */ }
    .col-sources { width: 15%; /* Reduced width as requested */ }
    
    /* Text Wrapping */
    .wrap-text {
        white-space: normal;
        word-wrap: break-word;
    }
    
    /* Sources - No Link */
    .source-item {
        display: block;
        margin-bottom: 4px;
        color: #555;
        font-size: 0.85rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
</style>
<table class="custom-table">
    <thead>
        <tr>
            <th class="col-carrier">Carrier</th>
            <th class="col-eligibility">Write Risk?</th>
            <th class="col-lines">Eligible Lines</th>
            <th class="col-notes">Restrictions / Notes</th>
            <th class="col-sources">Sources</th>
        </tr>
    </thead>
    <tbody>
"""

    for c in carriers_data:
        name = c.get("name", "")
        # Safe HTML escaping
        safe_name = html.escape(name)
        
        eligibility = (c.get("eligibility", "") or "").capitalize()
        safe_eligibility = html.escape(eligibility)
        
        lines = ", ".join(c.get("lines", []) or [])
        safe_lines = html.escape(lines)
        
        notes = c.get("notes", "") or ""
        safe_notes = html.escape(notes).replace("\n", "<br>")
        
        # Format Sources with links
        sources_html = ""
        if carrier_to_files and name in carrier_to_files:
            file_paths = carrier_to_files[name]
            for idx, fp in enumerate(file_paths):
                # Constructing file display name
                display_name = Path(fp).name
                sources_html += f'<span class="source-item" title="{html.escape(fp)}">📄 {html.escape(display_name)}</span>'
        
        html_str += f"""
<tr>
    <td class="col-carrier"><strong>{safe_name}</strong></td>
    <td class="col-eligibility">{safe_eligibility}</td>
    <td class="col-lines wrap-text">{safe_lines}</td>
    <td class="col-notes wrap-text">{safe_notes}</td>
    <td class="col-sources">{sources_html}</td>
</tr>
"""

    html_str += """
</tbody>
</table>
"""

    st.markdown(html_str, unsafe_allow_html=True)


# --------------- STREAMLIT APP --------------- #

def main():
    st.set_page_config(page_title="Underwriting Screener", layout="wide")

    # Custom CSS for App Styling
    st.markdown("""
        <style>
        /* Main Container Padding */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        /* Dashboard Card Style */
        .stForm {
            border: 1px solid #e0e0e0;
            padding: 20px;
            border-radius: 10px;
            box_shadow: 0 2px 5px rgba(0,0,0,0.05);
            background-color: #ffffff;
        }
        /* Headings */
        h1, h2, h3 {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            color: #1f1f1f;
        }
        /* Sidebar/Expander text */
        .streamlit-expanderHeader {
            font-size: 0.9rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Underwriting Screener")
    st.caption("AI-Powered Eligibility Check & Follow-Up")

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
    if "screening_complete" not in st.session_state:
        st.session_state.screening_complete = False

    # ---- Questionnaire Form ----
    with st.form("risk_questionnaire_form"):
        st.subheader("Risk Questionnaire")
        st.markdown("Enter details about the prospect to screen against carrier guidelines.")
        
        col1, col2 = st.columns(2)

        with col1:
            first_name = st.text_input("First Name", value="Al")
            last_name = st.text_input("Last Name", value="Rivera")
            company_name = st.text_input("Company Name", value="Als Diner")
            company_size = st.text_input("Company Size", value="10", help="Total number of full-time equivalent employees")

        with col2:
            industry = st.text_input("Industry", value="Hospitality", help="e.g. Construction, retail, service")
            business = st.text_input("Business", value="Bar Resturant")
            res_com = st.selectbox(
                "Residential or Commercial",
                options=["Commercial", "Residential", "Both"],
                index=0,
            )

        submitted = st.form_submit_button("Screen Risk", type="primary")

    if submitted:
        # Reset state for new run
        st.session_state.chat_messages = []
        st.session_state.eligible_carriers = []
        st.session_state.static_context = ""
        st.session_state.screening_complete = False
        
        with st.spinner("Analyzing carrier guidelines..."):
            # build questionnaire data
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

            # --- Embed the risk description ---
            q_emb = embed_query(risk_desc)

            PER_CARRIER_TOP_K = 5
            TOTAL_TARGET = 100

            docs_all: List[str] = []
            metas_all: List[dict] = []
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
                    continue

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

            # --- Classify carriers (yes/maybe/no) ---
            eligible, maybe, carriers_data, summary = classify_carriers_json(
                risk_desc, docs_all, metas_all
            )

            eligible_union = eligible | maybe
            st.session_state.eligible_carriers = sorted(list(eligible_union))

            # Store results in session state to persist after rerun
            st.session_state.last_results = {
                "carriers_data": carriers_data,
                "summary": summary,
                "eligible_union": eligible_union,
                "docs_all": docs_all,
                "metas_all": metas_all,
            }
            st.session_state.screening_complete = True
            
            st.success("Analysis Complete!")

    # Display Results (if available)
    if st.session_state.screening_complete and "last_results" in st.session_state:
        res = st.session_state.last_results
        carriers_data = res["carriers_data"]
        summary = res["summary"]
        eligible_union = res["eligible_union"]
        metas_all = res["metas_all"]
        docs_all = res["docs_all"]

        st.markdown("### Risk Description")
        st.info(st.session_state.risk_desc)
        
        # --- Prepare "Sources" map for the table ---
        carrier_to_files_raw: Dict[str, set] = {}
        for m in metas_all:
            carrier = m.get("company", "UNKNOWN")
            rel_path = m.get("relative_path") or m.get("filename", "")
            if not rel_path:
                continue
            carrier_to_files_raw.setdefault(carrier, set()).add(rel_path)
            
        carrier_to_files: Dict[str, List[str]] = {
            carrier: sorted(files)
            for carrier, files in carrier_to_files_raw.items()
            if carrier in eligible_union  # Or show all? User likely wants eligible. logic was eligible only.
        }

        # Show table + notes
        st.markdown("### Underwriting Eligibility Summary")
        render_html_table(carriers_data, carrier_to_files)

        if summary:
            st.markdown("---")
            st.markdown("#### Detailed Notes")
            st.markdown(summary)

        # --- Debug Info in Expander ---
        with st.expander("Show detailed debug info (Retrieved Chunks & Sources)"):
            st.write(f"**Total relevant chunks:** {len(docs_all)}")
            carriers_present = sorted({m.get("company", "UNKNOWN") for m in metas_all})
            st.write(f"**Carriers Scanned:** {', '.join(carriers_present)}")
            # Could list all files here too if needed

        # --- Build static context for follow-up chat ---
        if not st.session_state.static_context:
            filtered_docs = []
            filtered_metas = []
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

            st.session_state.static_context = "\n\n---\n\n".join(context_blocks)


    # ---- Follow-up Chat ----
    st.markdown("---")
    
    # Only show chat if we have context
    if st.session_state.static_context and st.session_state.eligible_carriers:
        st.subheader("Follow-up Questions (Eligible Carriers Only)")
        
        st.caption(
            "Answering using eligible carriers."
        )

        # Render chat history
        for msg in st.session_state.chat_messages:
            # No avatars, plain cleaner look
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("Ask a question about coverage for this risk...")
        if user_input:
            # Add user message
            st.session_state.chat_messages.append(
                {"role": "user", "content": user_input}
            )
            with st.chat_message("user"):
                st.markdown(user_input)

            # --- TWO-STEP LOGIC ---
            
            # 1. Reconstruct current context from session state (all currently eligible chunks)
            # We access the raw data from last_results to ensure we have the full pool
            if "last_results" in st.session_state:
                res = st.session_state.last_results
                docs_pool = res["docs_all"]
                metas_pool = res["metas_all"]
                eligible_pool = res["eligible_union"] # The baseline eligible list
            else:
                # Fallback (shouldn't happen if we are here)
                docs_pool = []
                metas_pool = []
                eligible_pool = []

            # Filter pool to only what was originally eligible (baseline)
            current_docs = []
            current_metas = []
            for d, m in zip(docs_pool, metas_pool):
                if m.get("company", "UNKNOWN") in eligible_pool:
                    current_docs.append(d)
                    current_metas.append(m)

            # Construct the context text for the filtering step
            filter_context_blocks = []
            for d, m in zip(current_docs, current_metas):
                carrier = m.get("company", "UNKNOWN")
                filter_context_blocks.append(f"[Company: {carrier}]\n{d}")
            filter_context_str = "\n---\n".join(filter_context_blocks)

            # 2. RUN FILTERING STEP
            # We pass the full history including the latest user message
            with st.spinner("Checking eligibility against new info..."):
                current_chat_history = st.session_state.chat_messages
                
                ineligible_names = identify_ineligible_carriers(
                    current_chat_history, 
                    list(eligible_pool), 
                    filter_context_str
                )
            
            # 3. APPLY FILTER (Code Level)
            final_eligible_carriers = [c for c in eligible_pool if c not in ineligible_names]
            
            # If everybody got filtered out, we should warn, but technically we just proceed with empty/limited context
            # Rebuild context for the final answer using ONLY valid carriers
            final_docs = []
            final_metas = []
            for d, m in zip(docs_pool, metas_pool):
                c_name = m.get("company", "UNKNOWN")
                if c_name in final_eligible_carriers:
                    final_docs.append(d)
                    final_metas.append(m)

            final_context_blocks = []
            for d, m in zip(final_docs, final_metas):
                carrier = m.get("company", "UNKNOWN")
                fname = m.get("filename", "") 
                final_context_blocks.append(f"[Company: {carrier} | File: {fname}]\n{d}")
            
            final_context_str = "\n\n---\n\n".join(final_context_blocks)
            
            # Update the display list string for the prompt
            final_carriers_str = ", ".join(final_eligible_carriers)
            
            # 4. GENERATE ANSWER
            system_prompt = (
                "You are an underwriting assistant.\n\n"
                f"The ONLY eligible carriers remaining are: {final_carriers_str}.\n"
                "You have a fixed risk profile and a fixed set of underwriting excerpts.\n\n"
                "Rules:\n"
                "- Answer questions using ONLY the provided excerpts.\n"
                "- Exclude any mentions of carriers NOT in the eligible list above.\n"
                "- FORMATTING: BE CONCISE. Use summary tables. Avoid long paragraphs.\n"
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"RISK PROFILE:\n{st.session_state.risk_desc}\n\n"
                        f"VALID UNDERWRITING EXCERPTS:\n\n"
                        f"{final_context_str}"
                    ),
                },
            ]

            # Add chat history (optional: technically we re-injected context so we might not need full history, 
            # but it helps for conversational flow. However, we MUST prevent the LLM from hallucinating 
            # based on old 'ineligible' turns. 
            # SIMPLIFICATION: We just append the history. The system prompt 'ONLY eligible carriers' 
            # is a strong constraint, and the context *physically* lacks the other carriers now.)
            for msg in st.session_state.chat_messages:
                messages.append({"role": msg["role"], "content": msg["content"]})

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    resp = openai_client.chat.completions.create(
                        model=CHAT_MODEL,
                        messages=messages,
                        temperature=0.2,
                    )
                    answer = resp.choices[0].message.content.strip()
                    
                    # Optional: Add a debug note about who was filtered
                    if ineligible_names:
                        debug_msg = f"\n\n*(Filtered out based on criteria: {', '.join(ineligible_names)})*"
                        answer += debug_msg
                        
                    st.markdown(answer)

            st.session_state.chat_messages.append(
                {"role": "assistant", "content": answer}
            )
    elif not st.session_state.screening_complete:
        st.info("Please fill out the questionnaire above and click **Screen Risk** to start.")


if __name__ == "__main__":
    main()
