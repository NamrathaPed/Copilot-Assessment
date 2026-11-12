# Shop-Floor Copilot

A document-grounded question-answering assistant for manufacturing shift supervisors.
Ask a question in plain English, get a short answer with the exact source passage and
page number — or an explicit "I couldn't find this" when the documents do not support an
answer.

Built for the Kaveri Precision Components (Hosur Plant) document set: incoming material
inspection, lockout/tagout safety, and TX-200 CNC preventive maintenance.

---

## 1. Quick Start

### Prerequisites

- Python 3.9+ (developed on 3.9.21)
- macOS, Linux, or Windows
- ~2 GB disk for the sentence-transformer model and torch

### Install

```bash
cd copilot
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
```

Edit `.env` and set at least one backend:

```ini
LLM_BACKEND=nvidia
NVIDIA_API_KEY=nvapi-...            # from https://build.nvidia.com
NVIDIA_MODEL=nvidia/nemotron-3.5-bolt-e4b
```

Any of these backends works — pick one:

| `LLM_BACKEND` | Required settings |
|---|---|
| `nvidia` | `NVIDIA_API_KEY`, `NVIDIA_MODEL` |
| `anthropic` | `ANTHROPIC_API_KEY`, `CLAUDE_MODEL` |
| `ollama` | local Ollama on `:11434`, `OLLAMA_MODEL` |

Keep `.env` out of version control. Restrict it: `chmod 600 .env`.

### Build the index

`index.pkl` is already included, so you can skip this. To rebuild it from the PDFs:

```bash
python ingest.py
# -> Indexed 14 chunks
```

### Run

```bash
uvicorn app:app --reload --port 8000
```

Open <http://localhost:8000>.

Example request:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many samples should be inspected for a lot of 200 steel bars?"}'
```

Response:

```json
{
  "found": true,
  "answer": "13",
  "sources": [
    {
      "file": "QC-SOP-014_Rev04_Incoming_Material_Inspection.pdf",
      "page": 1,
      "quote": "91 - 500\n13"
    }
  ],
  "safety": false
}
```

---

## 2. Architecture

```
                    ┌──────────────────────────────────────┐
   Browser  ──────► │  FastAPI  app.py                     │
   (index.html)     │                                      │
                    │  POST /ask                           │
                    │    │                                 │
                    │    ├─ retrieve(question)             │
                    │    │     ├─ dense: MiniLM cosine    │
                    │    │     ├─ sparse: BM25            │
                    │    │     └─ fuse: Reciprocal Rank   │
                    │    │        Fusion (k = 60)         │
                    │    │                                 │
                    │    ├─ score < MIN_SCORE ? ──► not    │
                    │    │                          found   │
                    │    ├─ call_llm(context, question)    │
                    │    │     └─ NVIDIA / Anthropic /    │
                    │    │        Ollama                  │
                    │    ├─ parse JSON answer              │
                    │    └─ verify each quote appears in   │
                    │       the source passage ──► drop    │
                    │       unverified evidence            │
                    └──────────────────────────────────────┘
                                      │
                              index.pkl (chunks + embeddings)
                                      ▲
                              built by ingest.py
```

### Why this design

**Hybrid retrieval.** Dense embeddings catch paraphrases ("how many pieces do I pull"
→ "samples to inspect"), while BM25 catches exact tokens (part numbers, form IDs,
`AL-305`). Each contributes its top 20; Reciprocal Rank Fusion merges them so neither
ranking dominates.

**Grounded generation.** The system prompt forbids outside knowledge and requires JSON
with an `evidence` list of `{source, quote}` pairs. The model sees numbered passages and
must cite by number.

**Evidence verification.** The backend never trusts the model's citations. Every quote is
normalised (whitespace-collapsed, lower-cased) and must be a literal substring of the
passage it claims to come from. If no quote survives verification, the request returns
`found: false` with the related documents listed — the answer is rejected rather than
shown unverified.

**Abstention by default.** A cosine-score floor (`MIN_SCORE`, default 0.30) short-
circuits retrieval before any LLM call. Combined with evidence verification, the system
has three independent chances to say "I don't know."

**Safety flagging.** A regex over safety-critical terms (`lockout`, `tagout`, `loto`,
`ppe`, `hazard`, `chemical`, `emergency`, `energy`, …) sets `safety: true`, which makes the
UI render a prominent warning banner.

### Files

| Path | Purpose |
|---|---|
| `copilot/app.py` | FastAPI app: retrieval, LLM call, verification, endpoints |
| `copilot/ingest.py` | PDF → chunk → embed → `index.pkl` |
| `copilot/static/index.html` | Single-page chat UI (vanilla JS, no build step) |
| `copilot/index.pkl` | Prebuilt index: 14 chunks, 384-dim embeddings |
| `copilot/docs/` | Source PDFs |
| `copilot/.env` | API keys and backend selection (not committed) |

### Chunking

Pages are split on line boundaries into chunks of ~1200 characters with a 2-line overlap,
so tables (like the sampling plan) stay intact and context is not lost at a boundary.
Page numbers are preserved per chunk so every quote can be traced to a page.

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BACKEND` | `anthropic` | `nvidia`, `anthropic`, or `ollama` |
| `MIN_SCORE` | `0.30` | Cosine floor for answering at all |
| `NVIDIA_MODEL` | — | Any model provisioned on your NVIDIA account |
| `CLAUDE_MODEL` | `claude-haiku-5-5` | Anthropic model ID |
| `OLLAMA_MODEL` | `llama3.1:8b` | Local Ollama model |

---

## 3. Test Questions and Actual Answers

Run against `nvidia/nemotron-3.5-bolt-e4b` with the included `index.pkl`.
Each row is real output from `POST /ask`.

### Answerable — direct lookup

**Q1.** For a lot of 200 incoming steel bars, how many samples should be inspected?

> **13**
> Source: `QC-SOP-014_Rev04_Incoming_Material_Inspection.pdf` — page 1
> Evidence: *"91 - 500 / 13"*

**Q2.** What should the QC team do when a supplier delivers material without a Material
Test Certificate (MTC)?

> The lot is put on HOLD and must not be inspected or accepted until the MTC is received.
> Source: `QC-SOP-014_Rev04_Incoming_Material_Inspection.pdf` — page 2
> Evidence: *"If it is missing, the lot is put on HOLD and must not be inspected or
> accepted until the MTC is received."*

### Answerable — multi-hop / numeric

**Q4.** How frequently should the TX-200 CNC machine's spindle runout be checked, and what
is the maximum permitted runout?

> Spindle runout should be checked monthly, and the maximum permitted runout is 0.005 mm.
> Sources: `MNT-MAN-CNC-02_TX-200_Preventive_Maintenance.pdf` — page 1
> Evidence: *"Monthly / Check spindle runout…"* and *"Spindle runout (at chuck face) /
> 0.005 mm max."*

### Correctly refused — outside the corpus

**Q5.** What cutting speed should operators use when machining titanium on the TX-200 CNC
machine?

> I couldn't find this in the provided documents. Please check with your manager or the
> document owner.
> **Correct refusal** — the maintenance manual lists hydraulic pressure, chuck pressure,
> coolant concentration and alarms, but no cutting-speed table. This is the single most
> important behaviour: a weaker system would invent a plausible SFM value.

**Q3.** An employee who applied a LOTO lock is absent, and production wants to restart
the machine. Can the shift supervisor remove the lock?

> I couldn't find this in the provided documents. Please check with your manager or the
> document owner.

> **Known limitation — see THINKING.md §5.** This is an over-abstention, not a correct
> refusal. `SAF-SOP-003` §6 does answer it: the supervisor must first try to phone the
> owner, and if unreachable, **both** the Production Manager **and** Safety Officer must
> authorise removal. The passage ranks #1 (cosine 0.711), so retrieval works; the model
> declined because the question is framed as a yes/no permission and the passage answers
> with a conditional procedure. Rephrased as *"Can a shift supervisor remove a lock when
> the owner is absent?"* the system answers correctly.

### Behaviour summary

| # | Question type | Expected | Actual |
|---|---|---|---|
| Q1 | Table lookup | 13 | 13 |
| Q2 | Conditional rule | HOLD until MTC | HOLD until MTC |
| Q3 | Permission, conditional | Supervisor cannot act alone | Abstained |
| Q4 | Frequency + numeric limit | Monthly, 0.005 mm | Monthly, 0.005 mm |
| Q5 | Not in corpus | Refuse | Refused |

---

## 4. Limitations

- **Revision handling.** `QC-SOP-014` exists as Rev 03 and Rev 04, and both are indexed.
  Rev 04 (effective 2026-07-01) supersedes Rev 03 (2023-03-15). Retrieval currently has no
  revision awareness, so a query can surface Rev 03 text. Mitigation in place: Rev 04 wins
  on the tested queries because its chunks score higher, but this is not guaranteed. A
  production system should filter to the latest effective revision per document number.
- **Over-abstention on yes/no framing.** See Q3 above.
- **English only.** PDFs and questions are English; no multilingual handling.
- **No live operational data.** Task management and prioritisation were deliberately
  dropped — see THINKING.md §3.
- **Single-process, in-memory index.** The 14-chunk index is loaded at startup; there is
  no incremental update, authentication, or rate limiting.
- **LLM-dependent.** Answer quality and JSON reliability vary by backend and model.