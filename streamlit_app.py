# streamlit_app.py
import streamlit as st
import pandas as pd
import appetite_core as core  # the file above

st.set_page_config(page_title="GPT Appetite SQL Chat", layout="wide")
st.title("GPT Appetite SQL Chat (CSV + PDF tables)")

# --- DB + data load ---
con = core.create_connection()
core.load_guard_appetite_indicators(con)
core.load_small_business(con)
core.load_ana_bob(con)
core.recreate_unified_view(con)

with st.expander("Loaded tables / row counts", expanded=False):
    st.write(core.get_row_counts(con))

# --- Sidebar: PDF ingestion ---
st.sidebar.header("PDF ingestion")
uploaded_pdfs = st.sidebar.file_uploader(
    "Upload appetite / underwriting PDFs",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_pdfs:
    files = [(f.name, f.read()) for f in uploaded_pdfs]
    total_rows = core.ingest_pdf_blobs(con, files)
    st.sidebar.success(f"Ingested {len(uploaded_pdfs)} PDF(s), {total_rows} normalized row(s).")
    core.rebuild_pdf_index(con)
    st.sidebar.info(f"PDF embedding index rebuilt.")

mode = st.radio(
    "Query mode",
    ["SQL over all tables (CSV + PDFs)", "Semantic Q&A over PDF tables"],
    horizontal=True,
)

# --- SQL mode ---
if mode == "SQL over all tables (CSV + PDFs)":
    prompt = st.chat_input("Ask about appetite (e.g., hotels with bop >= 0.5 across sources)")
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            try:
                df, summary, sql, relaxed = core.sql_chat(con, prompt)
            except ValueError as e:
                st.error(str(e))
                df = pd.DataFrame()
                summary = ""

            if df.empty:
                st.warning("No rows found.")
            else:
                st.subheader("Result")
                st.dataframe(df, use_container_width=True)
                st.download_button(
                    "Download CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    file_name="result.csv",
                    mime="text/csv"
                )
                st.subheader("Summary")
                st.write(summary)

# --- Semantic PDF mode ---
else:
    q = st.chat_input("Ask a question about the appetite / underwriting PDFs...")
    if q:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            answer, hits_df = core.semantic_pdf_answer(q)
            if hits_df is None or hits_df.empty:
                st.warning("No relevant PDF rows found.")
            else:
                st.subheader("Most relevant rows from PDF tables")
                st.dataframe(hits_df, use_container_width=True)
            st.subheader("Answer")
            st.write(answer)
