import os
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple, Optional

import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

# ---- OpenAI setup ----
load_dotenv()          # reads OPENAI_API_KEY from .env
client = OpenAI()      # Pylance now knows 'client' exists


# ========== CONFIG ==========

ROOT_DIR = r"C:\Users\StevePhillips\Downloads\Underwriting\Underwriting"
OUTPUT_JSONL = "embeddings_output.jsonl"
EMBEDDING_MODEL = "text-embedding-3-large"


# ========== EMBEDDING BACKEND (PLUG YOURS HERE) ==========

def embed_texts(texts: List[str]) -> List[List[float]]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]


# ========== CHUNKING FOR PDFs (RECURSIVE) ==========

def recursive_chunk_text(
    text: str,
    max_chars: int = 2000,
    overlap_chars: int = 200
) -> List[str]:
    """
    Recursively splits text by separators to keep chunks under max_chars.
    Separators: \n\n (paragraphs), \n (lines), . (sentences), space, empty.
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    separators = ["\n\n", "\n", ". ", " ", ""]
    
    # Find the best separator that works
    for sep in separators:
        if sep == "":
            # Fallback: hard cut
            chunks = []
            for i in range(0, len(text), max_chars - overlap_chars):
                chunks.append(text[i : i + max_chars])
            return chunks

        splits = text.split(sep)
        # If splitting by this separator doesn't help reduce size (e.g. one giant block), try next
        if any(len(s) > max_chars for s in splits) and len(splits) < 2:
            continue
        
        # Merge splits back into chunks
        chunks = []
        current_chunk = []
        current_len = 0
        
        for s in splits:
            s_len = len(s) + len(sep)
            if current_len + s_len > max_chars:
                if current_chunk:
                    chunks.append(sep.join(current_chunk))
                current_chunk = [s]
                current_len = s_len
            else:
                current_chunk.append(s)
                current_len += s_len
        
        if current_chunk:
            chunks.append(sep.join(current_chunk))
            
        # If we successfully chunked it, return (with overlap logic if needed, 
        # but recursive usually handles context well enough. We can add overlap 
        # by re-joining previous tail, but for simplicity we'll stick to clean breaks 
        # or simple overlap if requested).
        
        # Simple overlap implementation for the resulting chunks
        if overlap_chars > 0 and len(chunks) > 1:
            overlapped_chunks = []
            prev_tail = ""
            for ch in chunks:
                if prev_tail:
                    # Prepend overlap
                    new_chunk = (prev_tail + sep + ch).strip()
                    # If adding overlap makes it too big, we might just take the tail
                    if len(new_chunk) > max_chars + overlap_chars: 
                        # Soft limit: allow it to go a bit over to preserve context
                        pass
                    overlapped_chunks.append(new_chunk)
                else:
                    overlapped_chunks.append(ch)
                
                # Update tail
                prev_tail = ch[-overlap_chars:]
            return overlapped_chunks
            
        return chunks

    return [text] # Should not reach here due to empty separator fallback


# ========== PDF PROCESSING (PDFPLUMBER) ==========

def extract_text_from_pdf(path: Path) -> List[Tuple[int, str]]:
    """
    Extract text per page using pdfplumber for better table layout preservation.
    Returns a list of (page_number, page_text) with 1-based page numbers.
    """
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is not installed. Run: pip install pdfplumber")

    page_blocks: List[Tuple[int, str]] = []

    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                # extract_text(layout=True) attempts to preserve physical layout
                text = page.extract_text(layout=True) or ""
                text = text.strip()
                if text:
                    page_blocks.append((i + 1, text))
    except Exception as e:
        print(f"[WARN] Error reading {path} with pdfplumber: {e}")
        return []

    return page_blocks


# ========== SPREADSHEET PROCESSING (ROW-LEVEL) ==========

def try_cast_numeric(val: Any) -> Any:
    """
    Try to cast a value to int/float; if it fails, return original.
    This lets numeric columns live as real numbers in metadata
    when possible.
    """
    if isinstance(val, (int, float)):
        return val
    s = str(val).strip()
    if s == "":
        return ""
    # Try int
    try:
        i = int(s)
        return i
    except Exception:
        pass
    # Try float
    try:
        f = float(s)
        return f
    except Exception:
        pass
    return val


def extract_rows_from_spreadsheet(path: Path) -> List[Dict[str, Any]]:
    """
    Read XLS/XLSX/CSV and return a list of row records:
      [
        {
          "text": "...natural language row description...",
          "metadata_extras": { ... sheet, row_index, columns ... }
        }
      ]
    """
    ext = path.suffix.lower()
    if ext in [".xls", ".xlsx"]:
        sheets = pd.read_excel(path, sheet_name=None)
    elif ext == ".csv":
        try:
            sheets = {"Sheet1": pd.read_csv(path)}
        except UnicodeDecodeError:
            sheets = {"Sheet1": pd.read_csv(path, encoding="latin1", on_bad_lines="skip")}
    else:
        return []

    row_records: List[Dict[str, Any]] = []

    for sheet_name, df in sheets.items():
        df = df.fillna("")  # avoid NaNs
        headers = list(map(str, df.columns))

        for idx, row in df.iterrows():
            # Build natural language row description
            cells = []
            meta_row: Dict[str, Any] = {
                "sheet": sheet_name,
                "row_index": int(idx),
            }

            for col, val in row.items():
                col_str = str(col)
                val_cast = try_cast_numeric(val)
                meta_row[col_str] = val_cast
                cells.append(f"{col_str} = {val_cast}")

            header_str = ", ".join(headers)
            row_str = "; ".join(cells)
            text_block = (
                f"Sheet: {sheet_name}. "
                f"Columns: {header_str}. "
                f"Row {idx + 1}: {row_str}."
            )

            row_records.append({
                "text": text_block,
                "metadata_extras": meta_row
            })

    return row_records


# ========== METADATA & DIRECTORY WALKING ==========

def guess_company_from_path(path: Path, root: Path) -> str:
    """
    Assume directory structure:
      root / Company / ... / file
    We use the first path component after root as company.
    """
    rel = path.relative_to(root)
    parts = rel.parts
    if len(parts) >= 2:
        return parts[0]
    else:
        return "UNKNOWN_COMPANY"


def iter_files(root_dir: Path) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            yield Path(dirpath) / fname


# ========== MAIN PER-FILE PROCESSING ==========

def process_pdf(path: Path, root_dir: Path) -> List[Dict[str, Any]]:
    """
    Process a PDF into text chunks (section-level) with PAGE metadata.
    """
    company = guess_company_from_path(path, root_dir)

    base_metadata = {
        "company": company,
        "filename": path.name,
        "relative_path": str(path.relative_to(root_dir)),
        "file_type": "pdf"
    }

    try:
        page_blocks = extract_text_from_pdf(path)
    except Exception as e:
        print(f"[WARN] Failed to read PDF {path}: {e}")
        return []

    if not page_blocks:
        print(f"[INFO] No readable text for {path}, skipping PDF.")
        return []

    records: List[Dict[str, Any]] = []

    # Optional: prepend a simple header to each chunk for better retrieval
    doc_header = (
        f"Document: {path.name} "
        f"(Company: {company}).\n\n"
    )

    global_chunk_index = 0

    for page_number, page_text in page_blocks:
        chunks = recursive_chunk_text(page_text)
        if not chunks:
            continue

        for local_idx, ch in enumerate(chunks):
            text_with_header = doc_header + f"[PAGE {page_number}]\n\n" + ch

            meta = dict(base_metadata)
            meta.update({
                "chunk_index": global_chunk_index,
                "content_type": "pdf_text",
                "page_start": page_number,
                "page_end": page_number
            })

            records.append({
                "id": str(uuid.uuid4()),
                "text": text_with_header,
                "metadata": meta
            })

            global_chunk_index += 1

    return records


def process_spreadsheet(path: Path, root_dir: Path) -> List[Dict[str, Any]]:
    """
    Process a spreadsheet into one record per row.
    """
    company = guess_company_from_path(path, root_dir)

    base_metadata = {
        "company": company,
        "filename": path.name,
        "relative_path": str(path.relative_to(root_dir)),
        "file_type": path.suffix.lower().lstrip(".")
    }

    try:
        row_records = extract_rows_from_spreadsheet(path)
    except Exception as e:
        print(f"[WARN] Failed to read spreadsheet {path}: {e}")
        return []

    if not row_records:
        print(f"[INFO] No rows found in {path}.")
        return []

    records: List[Dict[str, Any]] = []

    for i, row_rec in enumerate(row_records):
        text_block = row_rec["text"]
        metadata_extras = row_rec["metadata_extras"]

        meta = dict(base_metadata)
        meta.update(metadata_extras)
        meta.update({
            "chunk_index": i,
            "content_type": "spreadsheet_row"
        })

        records.append({
            "id": str(uuid.uuid4()),
            "text": text_block,
            "metadata": meta
        })

    return records


def process_file(path: Path, root_dir: Path) -> List[Dict[str, Any]]:
    """
    Dispatch based on file extension.
    Returns a list of {
      "id": str,
      "text": str,
      "metadata": dict
    }
    """
    ext = path.suffix.lower()
    if ext == ".pdf":
        return process_pdf(path, root_dir)
    elif ext in [".xls", ".xlsx", ".csv"]:
        return process_spreadsheet(path, root_dir)
    else:
        # Unknown / unsupported type
        return []


# ========== MAIN SCRIPT ==========

def main():
    root = Path(ROOT_DIR)
    all_records: List[Dict[str, Any]] = []

    print(f"[INFO] Walking root directory: {root}")
    for file_path in iter_files(root):
        ext = file_path.suffix.lower()
        if ext not in [".pdf", ".xls", ".xlsx", ".csv"]:
            continue

        print(f"[INFO] Processing {file_path}")
        file_records = process_file(file_path, root)

        if not file_records:
            continue

        # Batch embed for this file’s chunks/rows
        texts = [r["text"] for r in file_records]
        embeddings = embed_texts(texts)

        if len(embeddings) != len(file_records):
            raise RuntimeError("Embedding count does not match chunk/row count")

        for rec, emb in zip(file_records, embeddings):
            rec["embedding"] = emb

        all_records.extend(file_records)

    # Write to JSONL
    if all_records:
        with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
            for rec in all_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[INFO] Wrote {len(all_records)} records to {OUTPUT_JSONL}")
    else:
        print("[INFO] No records generated.")


if __name__ == "__main__":
    main()
