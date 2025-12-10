# appetite_core.py
import os, re, json, duckdb, pandas as pd, sqlparse, io
from dotenv import load_dotenv
from typing import List, Dict, Any
import numpy as np
from openai import OpenAI

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
EMBED_MODEL  = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
DB_PATH      = os.getenv("DUCKDB_PATH", "tables.duckdb")

CSV_GUARD = os.getenv("CSV_GUARD", "/mnt/data/GUARD_Appetite_Indicators_Table1.csv")
CSV_SMALL = os.getenv("CSV_SMALL", "/mnt/data/Appetite_Merged_small_business_table-1.csv")
CSV_ANAB  = os.getenv("CSV_ANAB",  "/mnt/data/ANA_BOB_Appetite_Guide_Merged_Table-1.csv")

client = OpenAI()

# ---------------- DB INIT ----------------

def create_connection(db_path: str = DB_PATH) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(db_path)
    con.execute("CREATE SCHEMA IF NOT EXISTS raw;")
    return con

# ---------- Helpers (NO streamlit here) ----------

def clean_cols(cols):
    return [c.strip().lower().replace(" ", "_") for c in cols]

def numify(series):
    return (
        series.astype(str)
              .str.strip()
              .str.replace(",", "", regex=False)
              .str.replace(r"[^0-9.\-]", "", regex=True)
              .replace({"": None})
              .pipe(pd.to_numeric, errors="coerce")
    )

def pick(df, *candidates, default=None):
    cols = set(df.columns)
    for c in candidates:
        if c and c in cols:
            return c
    return default

def read_csv_as_str(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        # core should NOT touch UI; just return None
        return None
    df = pd.read_csv(path, dtype=str).fillna("")
    df.columns = clean_cols(df.columns)
    return df

# ---------- CSV LOADERS (same logic, without st.*) ----------

def load_guard_appetite_indicators(con: duckdb.DuckDBPyConnection, path=CSV_GUARD):
    df = read_csv_as_str(path)
    if df is None:
        con.execute("DROP TABLE IF EXISTS raw.guard_appetite;")
        return

    c_provider = pick(df, "provider")
    c_industry = pick(df, "industry")
    c_cob = pick(df, "class_of_business", "business", "class", "class_name", "business_class", "cob")
    c_wc = pick(df, "wc", "workers_comp", "workers_compensation")
    c_bop = pick(df, "bop")
    c_umb = pick(df, "comm_umb", "umb", "umbrella")
    c_auto = pick(df, "comm_auto", "auto", "commercial_auto")

    out = pd.DataFrame({
        "provider": df[c_provider] if c_provider else "",
        "industry": df[c_industry] if c_industry else "",
        "class_of_business": df[c_cob] if c_cob else "",
        "wc": numify(df[c_wc]) if c_wc else None,
        "bop": numify(df[c_bop]) if c_bop else None,
        "comm_umb": numify(df[c_umb]) if c_umb else None,
        "comm_auto": numify(df[c_auto]) if c_auto else None,
    })

    con.execute("DROP TABLE IF EXISTS raw.guard_appetite;")
    con.execute("""
        CREATE TABLE raw.guard_appetite (
          provider TEXT,
          industry TEXT,
          class_of_business TEXT,
          wc DOUBLE,
          bop DOUBLE,
          comm_umb DOUBLE,
          comm_auto DOUBLE
        );
    """)
    con.register("df_guard", out)
    con.execute("INSERT INTO raw.guard_appetite SELECT * FROM df_guard;")


def load_small_business(con: duckdb.DuckDBPyConnection, path=CSV_SMALL):
    df = read_csv_as_str(path)
    if df is None:
        con.execute("DROP TABLE IF EXISTS raw.small_business_appetite;")
        return

    c_provider = pick(df, "provider")
    c_industry = pick(df, "industry")
    c_cob = pick(df, "business", "class_of_business", "class", "class_name", "business_class", "cob")

    c_bop = pick(df, "bop")
    c_pkg = pick(df, "pkg", "package")
    c_wc = pick(df, "wc", "workers_comp")
    c_umb = pick(df, "umb", "comm_umb", "umbrella")
    c_auto = pick(df, "auto", "comm_auto", "commercial_auto")
    c_im = pick(df, "im", "inland_marine", "inland")
    c_cyber = pick(df, "cyber")
    c_epli = pick(df, "epli")

    out = pd.DataFrame({
        "provider": df[c_provider] if c_provider else "",
        "industry": df[c_industry] if c_industry else "",
        "class_of_business": df[c_cob] if c_cob else "",
        "bop": numify(df[c_bop]) if c_bop else None,
        "pkg": numify(df[c_pkg]) if c_pkg else None,
        "wc": numify(df[c_wc]) if c_wc else None,
        "umb": numify(df[c_umb]) if c_umb else None,
        "auto": numify(df[c_auto]) if c_auto else None,
        "im": numify(df[c_im]) if c_im else None,
        "cyber": numify(df[c_cyber]) if c_cyber else None,
        "epli": numify(df[c_epli]) if c_epli else None,
    })

    con.execute("DROP TABLE IF EXISTS raw.small_business_appetite;")
    con.execute("""
        CREATE TABLE raw.small_business_appetite (
          provider TEXT,
          industry TEXT,
          class_of_business TEXT,
          bop DOUBLE,
          pkg DOUBLE,
          wc DOUBLE,
          umb DOUBLE,
          auto DOUBLE,
          im DOUBLE,
          cyber DOUBLE,
          epli DOUBLE
        );
    """)
    con.register("df_small", out)
    con.execute("INSERT INTO raw.small_business_appetite SELECT * FROM df_small;")


def load_ana_bob(con: duckdb.DuckDBPyConnection, path=CSV_ANAB):
    df = read_csv_as_str(path)
    if df is None:
        con.execute("DROP TABLE IF EXISTS raw.ana_bob_appetite;")
        return

    c_provider = pick(df, "provider")
    c_industry = pick(df, "industry")
    c_cob = pick(df, "business", "class_of_business", "class", "class_name", "business_class", "cob")
    c_bop = pick(df, "bop")
    c_wc = pick(df, "wc", "workers_comp")
    c_umb = pick(df, "umb", "comm_umb", "umbrella")
    c_auto = pick(df, "auto", "comm_auto", "commercial_auto")
    c_im = pick(df, "im", "inland_marine", "inland")
    c_cyber = pick(df, "cyber")

    out = pd.DataFrame({
        "provider": df[c_provider] if c_provider else "",
        "industry": df[c_industry] if c_industry else "",
        "class_of_business": df[c_cob] if c_cob else "",
        "bop": numify(df[c_bop]) if c_bop else None,
        "wc": numify(df[c_wc]) if c_wc else None,
        "umb": numify(df[c_umb]) if c_umb else None,
        "auto": numify(df[c_auto]) if c_auto else None,
        "im": numify(df[c_im]) if c_im else None,
        "cyber": numify(df[c_cyber]) if c_cyber else None,
    })

    con.execute("DROP TABLE IF EXISTS raw.ana_bob_appetite;")
    con.execute("""
        CREATE TABLE raw.ana_bob_appetite (
          provider TEXT,
          industry TEXT,
          class_of_business TEXT,
          bop DOUBLE,
          wc DOUBLE,
          umb DOUBLE,
          auto DOUBLE,
          im DOUBLE,
          cyber DOUBLE
        );
    """)
    con.register("df_anab", out)
    con.execute("INSERT INTO raw.ana_bob_appetite SELECT * FROM df_anab;")


# ---------- PDF INGESTION (core only) ----------

def ensure_pdf_table(con: duckdb.DuckDBPyConnection):
    con.execute("""
        CREATE TABLE IF NOT EXISTS raw.pdf_appetite (
          provider TEXT,
          industry TEXT,
          class_of_business TEXT,
          bop DOUBLE,
          pkg DOUBLE,
          wc DOUBLE,
          umb DOUBLE,
          auto DOUBLE,
          im DOUBLE,
          cyber DOUBLE,
          epli DOUBLE,
          source_pdf TEXT,
          page INTEGER,
          row_idx INTEGER
        );
    """)

def extract_tables_from_pdf_bytes(file_bytes: bytes, filename: str) -> pd.DataFrame:
    import pdfplumber
    rows: list[dict] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for t_idx, table in enumerate(tables):
                if not table or len(table) < 2:
                    continue
                header_raw = table[0]
                header = clean_cols([(h or "").strip() for h in header_raw])
                for r_idx, row in enumerate(table[1:], start=1):
                    rec: Dict[str, Any] = {}
                    for i in range(min(len(header), len(row))):
                        rec[header[i]] = (row[i] or "").strip()
                    rec["_pdf_name"] = filename
                    rec["_page"] = page_idx
                    rec["_row"] = r_idx
                    rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.columns = clean_cols(df.columns)
    return df

def normalize_pdf_tables(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    c_provider = pick(raw_df, "provider", "carrier", "company")
    c_industry = pick(raw_df, "industry", "segment", "category")
    c_cob = pick(
        raw_df,
        "class_of_business", "business", "business_description",
        "description", "class", "class_name", "naics_description"
    )
    c_bop = pick(raw_df, "bop")
    c_pkg = pick(raw_df, "pkg", "package")
    c_wc = pick(raw_df, "wc", "workers_comp", "workers_compensation")
    c_umb = pick(raw_df, "umb", "umbrella", "comm_umb")
    c_auto = pick(raw_df, "auto", "comm_auto", "commercial_auto")
    c_im = pick(raw_df, "im", "inland_marine", "inland")
    c_cyber = pick(raw_df, "cyber")
    c_epli = pick(raw_df, "epli")

    out = pd.DataFrame({
        "provider": raw_df[c_provider] if c_provider else "",
        "industry": raw_df[c_industry] if c_industry else "",
        "class_of_business": raw_df[c_cob] if c_cob else "",
        "bop": numify(raw_df[c_bop]) if c_bop else None,
        "pkg": numify(raw_df[c_pkg]) if c_pkg else None,
        "wc": numify(raw_df[c_wc]) if c_wc else None,
        "umb": numify(raw_df[c_umb]) if c_umb else None,
        "auto": numify(raw_df[c_auto]) if c_auto else None,
        "im": numify(raw_df[c_im]) if c_im else None,
        "cyber": numify(raw_df[c_cyber]) if c_cyber else None,
        "epli": numify(raw_df[c_epli]) if c_epli else None,
        "_pdf_name": raw_df.get("_pdf_name", ""),
        "_page": raw_df.get("_page", None),
        "_row": raw_df.get("_row", None),
    })
    return out

def insert_pdf_rows(con: duckdb.DuckDBPyConnection, df_norm: pd.DataFrame):
    if df_norm is None or df_norm.empty:
        return
    ensure_pdf_table(con)
    tmp = df_norm.rename(columns={
        "_pdf_name": "source_pdf",
        "_page": "page",
        "_row": "row_idx",
    })
    con.register("df_pdf_norm", tmp)
    con.execute("""
        INSERT INTO raw.pdf_appetite
        (provider, industry, class_of_business,
         bop, pkg, wc, umb, auto, im, cyber, epli,
         source_pdf, page, row_idx)
        SELECT
          provider, industry, class_of_business,
          bop, pkg, wc, umb, auto, im, cyber, epli,
          _pdf_name, _page, _row
        FROM df_pdf_norm;
    """)

# ---------- UNIFIED VIEW ----------

def recreate_unified_view(con: duckdb.DuckDBPyConnection):
    ensure_pdf_table(con)
    con.execute("DROP VIEW IF EXISTS raw.appetite_all;")
    con.execute("""
    CREATE VIEW raw.appetite_all AS
    SELECT
      'guard' AS source,
      provider,
      industry,
      class_of_business AS business,
      wc,
      bop,
      comm_umb AS umb,
      comm_auto AS auto,
      NULL::DOUBLE AS im,
      NULL::DOUBLE AS cyber,
      NULL::DOUBLE AS epli,
      NULL::DOUBLE AS pkg
    FROM raw.guard_appetite
    UNION ALL
    SELECT
      'small_business' AS source,
      provider,
      industry,
      class_of_business AS business,
      wc,
      bop,
      umb,
      auto,
      im,
      cyber,
      epli,
      pkg
    FROM raw.small_business_appetite
    UNION ALL
    SELECT
      'ana_bob' AS source,
      provider,
      industry,
      class_of_business AS business,
      wc,
      bop,
      umb,
      auto,
      im,
      cyber,
      NULL::DOUBLE AS epli,
      NULL::DOUBLE AS pkg
    FROM raw.ana_bob_appetite
    UNION ALL
    SELECT
      'pdf' AS source,
      provider,
      industry,
      class_of_business AS business,
      wc,
      bop,
      umb,
      auto,
      im,
      cyber,
      epli,
      pkg
    FROM raw.pdf_appetite;
    """)

def get_row_counts(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute("""
      SELECT 'raw.guard_appetite' AS table, COUNT(*) AS rows FROM raw.guard_appetite
      UNION ALL SELECT 'raw.small_business_appetite', COUNT(*) FROM raw.small_business_appetite
      UNION ALL SELECT 'raw.ana_bob_appetite', COUNT(*) FROM raw.ana_bob_appetite
      UNION ALL SELECT 'raw.pdf_appetite', COUNT(*) FROM raw.pdf_appetite
      UNION ALL SELECT 'raw.appetite_all', COUNT(*) FROM raw.appetite_all
    """).df()
