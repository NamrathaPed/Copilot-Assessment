import re, pickle, pathlib
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer

SIZE, OVERLAP = 800, 150

SIZE, OVERLAP_LINES = 1200, 2   # bigger chunks: the docs are short and tables stay together

def split_page(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    out, cur = [], []
    for line in lines:
        if cur and sum(len(l) + 1 for l in cur) + len(line) > SIZE:
            out.append("\n".join(cur))
            cur = cur[-OVERLAP_LINES:]          # repeat the last 2 lines so context isn't lost
        cur.append(line)
    if cur:
        out.append("\n".join(cur))
    return out

chunks = []
for pdf in sorted(pathlib.Path("docs").glob("*.pdf")):
    for pageno, page in enumerate(fitz.open(pdf), start=1):
        for t in split_page(page.get_text()):
            chunks.append({"file": pdf.name, "page": pageno, "text": t})

model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode([c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=True)
pickle.dump({"chunks": chunks, "emb": np.array(emb)}, open("index.pkl", "wb"))
print(f"Indexed {len(chunks)} chunks")