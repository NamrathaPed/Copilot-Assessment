
import os
import re
import json
import pickle
import pathlib

import numpy as np
import requests

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


ROOT = pathlib.Path(__file__).resolve().parent

# Load environment variables from .env
load_dotenv(ROOT / ".env")

MIN_SCORE = float(os.getenv("MIN_SCORE", 0.30))
BACKEND = os.getenv("LLM_BACKEND", "anthropic").lower()


# Load document index
with open(ROOT / "index.pkl", "rb") as f:
    data = pickle.load(f)

CHUNKS = data["chunks"]
EMB = data["emb"]

embedder = SentenceTransformer("all-MiniLM-L6-v2")

tok = lambda s: re.findall(r"\w+", s.lower())
bm25 = BM25Okapi([tok(c["text"]) for c in CHUNKS])
norm = lambda s: re.sub(r"\s+", " ", s).lower()


SAFETY = re.compile(
    r"\b(lockout|tagout|loto|warning|danger|caution|hazard|chemical|ppe|emergency|energy)\b",
    re.I
)


SYSTEM = """You answer questions for a factory shift supervisor using ONLY the numbered context passages.

Rules:
- If the passages do not clearly answer the question, return found=false.
- Never guess or use outside knowledge.
- Be brief and practical.
- Use short numbered steps if it is a procedure.
- Support the answer with evidence: copy short quotes EXACTLY (word for word) from the passages.

Return ONLY JSON:
{"found": true/false, "answer": "...", "evidence": [{"source": 1, "quote": "..."}]}
"""


# Retrieve relevant document chunks
def retrieve(q, k=5):
    qv = embedder.encode([q], normalize_embeddings=True)[0]

    dense = EMB @ qv

    d_rank = list(np.argsort(-dense)[:20])
    b_rank = list(np.argsort(-bm25.get_scores(tok(q)))[:20])

    rrf = {}

    for ranking in (d_rank, b_rank):
        for r, i in enumerate(ranking):
            rrf[i] = rrf.get(i, 0) + 1 / (60 + r)

    top = sorted(rrf, key=rrf.get, reverse=True)[:k]

    return top, float(dense.max())


# Call the selected LLM
def call_llm(user_msg):

    if BACKEND == "nvidia":

        api_key = os.getenv("NVIDIA_API_KEY")
        model = os.getenv("NVIDIA_MODEL")

        if not api_key:
            raise ValueError("NVIDIA_API_KEY is missing from .env")

        if not model:
            raise ValueError("NVIDIA_MODEL is missing from .env")

        r = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "temperature": 0,
                "max_tokens": 800,
                "stream": False,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_msg}
                ]
            },
            timeout=120
        )

        if not r.ok:
            print("NVIDIA API error:", r.status_code, r.text)
            r.raise_for_status()

        return r.json()["choices"][0]["message"]["content"]


    if BACKEND == "ollama":

        r = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
                "stream": False,
                "format": "json",
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_msg}
                ]
            },
            timeout=120
        )

        if not r.ok:
            print("Ollama API error:", r.status_code, r.text)
            r.raise_for_status()

        return r.json()["message"]["content"]


    if BACKEND == "anthropic":

        api_key = os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is missing from .env")

        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        msg = client.messages.create(
            model=os.getenv("CLAUDE_MODEL", "claude-haiku-5-5"),
            max_tokens=800,
            system=SYSTEM,
            messages=[
                {"role": "user", "content": user_msg}
            ]
        )

        return msg.content[0].text


    raise ValueError(
        f"Unsupported LLM_BACKEND: {BACKEND}. "
        "Choose anthropic, nvidia, or ollama."
    )


# Parse JSON response from LLM
def parse(raw):
    try:
        return json.loads(
            raw[raw.index("{"): raw.rindex("}") + 1]
        )
    except Exception:
        return None


NOT_FOUND = {
    "found": False,
    "answer": "I couldn't find this in the provided documents. "
              "Please check with your manager or the document owner.",
    "sources": [],
    "safety": False
}

UNAVAILABLE = {
    "found": False,
    "answer": "The answer service is temporarily unavailable. "
              "Please retry, or check with your manager.",
    "sources": [],
    "safety": False
}


class Q(BaseModel):
    question: str


app = FastAPI()


@app.post("/ask")
def ask(req: Q):

    idx, best = retrieve(req.question)

    if best < MIN_SCORE:
        return NOT_FOUND

    ctx = "\n\n".join(
        f"[{n}] ({CHUNKS[i]['file']}, page {CHUNKS[i]['page']})\n"
        f"{CHUNKS[i]['text']}"
        for n, i in enumerate(idx, 1)
    )

    try:
        out = parse(
            call_llm(f"Context:\n{ctx}\n\nQuestion: {req.question}")
        )
    except Exception as e:
        print("LLM backend error:", type(e).__name__, e)
        return UNAVAILABLE

    if not out or not out.get("found"):
        return NOT_FOUND

    sources = []

    # Verify that each LLM quote exists in its original passage
    for ev in out.get("evidence", []):

        n = ev.get("source", 0)
        quote = ev.get("quote", "")

        if (
            isinstance(n, int)
            and 1 <= n <= len(idx)
            and quote
            and norm(quote) in norm(CHUNKS[idx[n - 1]]["text"])
        ):
            c = CHUNKS[idx[n - 1]]

            sources.append({
                "file": c["file"],
                "page": c["page"],
                "quote": quote
            })

    # Reject answers without verifiable evidence
    if not sources:

        related_docs = sorted({
            f"{CHUNKS[i]['file']} p.{CHUNKS[i]['page']}"
            for i in idx[:3]
        })

        return {
            "found": False,
            "safety": False,
            "sources": [],
            "answer": "I found related material but couldn't verify an exact answer. "
                      "Please check: " + ", ".join(related_docs)
        }

    safety = bool(
        SAFETY.search(
            out["answer"] + " ".join(s["quote"] for s in sources)
        )
    )

    return {
        "found": True,
        "answer": out["answer"],
        "sources": sources,
        "safety": safety
    }


@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")


app.mount(
    "/static",
    StaticFiles(directory=ROOT / "static"),
    name="static"
)
