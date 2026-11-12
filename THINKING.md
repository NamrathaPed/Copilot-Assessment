# THINKING — Design Rationale and Reflection

This document records how the Shop-Floor Copilot was scoped, what feedback changed, and
why the implementation looks the way it does.

---

## 1. The User

### Who the supervisor is

A shop-floor supervisor owns the shift. They are the link between management and the
people running the machines, accountable for hitting production targets while keeping
procedures, quality standards, and safety rules intact.

The work is interrupt-driven. A supervisor coordinates several activities at once, assigns
work, handles the unexpected, and makes calls quickly — often standing next to a machine
with an operator waiting.

### The actual problem

The information they need is not in one place. It is spread across SOPs, maintenance
manuals, quality procedures, and safety documents. Answering a single question can mean
opening three PDFs, searching for a table, and checking whether the copy in hand is the
current revision — all while the shift clock runs.

### The two candidate uses of a copilot

**1. Task management and prioritisation.** Group the shift's work into quality checks,
maintenance, and safety tasks; order it by urgency; surface what needs attention now.

**2. Document-based question answering.** Ask in plain English instead of reading PDFs;
get the answer plus its source so the original instruction can be checked.

### What could go wrong

Trust is the whole problem. A model that produces confident, plausible, wrong instructions
is worse than no tool at all in a plant — a bad maintenance step or safety instruction can
injure someone or scrap product.

So the system had to be built to abstain: answer only when the documents support it, show
the source, and say plainly when it does not know rather than guess. Where documents are
ambiguous or conflict, it must push the decision back to a human.

---

## 2. Original Journey

The first concept combined both uses above. The intended flow:

1. Supervisor opens the copilot at the start of a shift.
2. Copilot proposes a prioritised task list for the shift.
3. Supervisor asks document questions throughout the shift.
4. Every answer carries a citation back to the source PDF and page.

The implicit assumption was that "understanding the shift" and "answering document
questions" were one product.

---

## 3. AI Critique

Two adversarial reviews were run: a skeptical shop-floor supervisor and a skeptical product
manager.

### Perspective A — the supervisor

1. **Where does task data come from?** SOPs describe *how* to do a task, not *which*
   tasks are due this shift. Without the production schedule, machine status, and who is
   available, prioritisation is guesswork — and a wrong priority on a safety task is
   harmful.
2. **Is it the current revision?** "Show me the source" is not enough if two documents
   disagree because one is obsolete. A supervisor needs revision number, effective date,
   and page — not just a filename.
3. **I need the answer now.** Standing at a machine, a long paragraph with five citations
   is useless. The answer must be short, with the steps scannable and safety warnings
   visually loud.
4. **"Not found" is not enough.** Say which documents were checked and who to ask next.

### Perspective B — the product manager

1. **Two problems, not one.** Document Q&A is buildable from the provided PDFs. Task
   management needs live operational data that is out of scope. Building both doubles the
   surface area and the risk. Ship Q&A first.
2. **Revision conflict is a first-class requirement.** If Rev 04 replaces Rev 03, the
   system must prefer Rev 04. This is most critical exactly where it matters most —
   quality and safety.
3. **How do we know it is useful?** A working chatbot is not a working product. Measure:
   correct retrieval, answers supported by sources, correct abstention, correct handling
   of revisions, and whether a supervisor can read the answer in seconds.
4. **What happens when it is wrong?** Retrieval can miss; generation can misread a
   relevant passage. The system must show traceable evidence and never stand in for
   established safety procedure or trained judgement.

### Overall critique

| Area | Gap identified |
|---|---|
| Task management | Needs live operational data, not SOPs |
| Retrieval | Must account for document revisions |
| UX | Answers must be short and scannable |
| Reliability | Every answer needs verifiable evidence |
| Safety | Unsupported instructions must not be guessed |
| Evaluation | Must test accuracy, abstention, and version handling |

---

## 4. Reflection

The feedback moved the work from features to the supervisor's actual moment of need.

The sharpest point was that task management and document Q&A are different products.
Prioritisation depends on live data the corpus does not contain, so it was cut from v1 and
recorded as a future enhancement rather than faked with plausible-looking output.

The revision point changed the design more than expected. "Retrieve the relevant passage"
is not the same as "retrieve the current instruction." Both `QC-SOP-014` Rev 03 and Rev 04
are in the corpus; without revision awareness the system can quote a 2023 procedure in
2026. This is a real, documented gap (README §4), not a solved problem — and that honesty
matters more than a claim of completeness.

The UX critique produced concrete rules: short answers, numbered steps for procedures, a
prominent safety banner, and an explicit statement of which documents were checked when the
answer is not found.

Finally, the reliability critique set the architecture. The system does not take the
model's word for its own citations. It re-checks every quote against the source passage and
drops any answer whose evidence cannot be verified.

---

## 5. Design Decisions

### D1 — Document Q&A only; task management deferred

**Decision.** Build grounded Q&A. Leave prioritisation as a future enhancement.

**Why.** Task management requires the production schedule, machine status, and staff
availability — none of which are in the PDFs. Shipping a prioritisation feature with no
data source would produce confident fiction, which is precisely the failure mode the
persona cannot tolerate.

### D2 — Hybrid retrieval with Reciprocal Rank Fusion

**Decision.** Retrieve with dense embeddings and BM25, take the top 20 from each, fuse with
RRF (`k = 60`).

**Why.** Dense retrieval handles paraphrase — "how many pieces do I pull" matches "samples
to inspect." BM25 handles exact tokens that embeddings blur: part numbers, `AL-305`,
`SF-12`, `QF-07A`. RRF merges the two without needing score calibration across incomparable
scales. Each list contributes its top 20 so a document ranked well by only one method still
surfaces.

### D3 — Chunking that preserves tables

**Decision.** Split each page on line boundaries into ~1200-character chunks with a 2-line
overlap; keep the page number with every chunk.

**Why.** The sampling plan is a table keyed by lot size. Character-window chunking would
cut `91 - 500 | 13` from its neighbours and make the answer unanswerable or wrong. Line
accumulation keeps the table whole. The overlap prevents losing a rule that straddles a
boundary. Page numbers are retained because every answer must be checkable against the
original document.

### D4 — Abstention at three independent layers

**Decision.** Refuse when (a) the best dense score is below `MIN_SCORE = 0.30`, (b) the
model returns `found: false`, or (c) no quote survives verification.

**Why.** Any single check can fail. A score floor stops retrieval from ever reaching the
model on an unrelated question. The model can recognise that a passage is adjacent but not
responsive. Verification catches fabricated citations. Three chances to say "I don't know"
is the core safety property, and it is what makes Q5 (titanium cutting speed, absent from
the corpus) return a refusal instead of an invented number.

### D5 — Never trust model-supplied citations

**Decision.** Every returned quote is normalised and must appear literally in the passage
it cites; failures are dropped; if none survive, return `found: false` plus the related
document names.

**Why.** A model that writes `{"source": 2, "quote": "..."}` can misremember the quote or
misattribute the source. Re-checking is cheap and converts an unverifiable answer into a
visible refusal. This is the difference between "the model said so" and "the document says
so."

### D6 — Backend-agnostic LLM layer

**Decision.** Support NVIDIA, Anthropic, and Ollama behind one `call_llm()` function chosen
by `LLM_BACKEND`.

**Why.** The plant cannot assume a single vendor, and a local model is attractive for
confidential documents. Swapping backends is an env change, not a code change. The
trade-off is that each backend has different JSON reliability, which is why `parse()`
extracts the outermost `{...}` rather than assuming a clean body.

### D7 — Graceful degradation on backend failure

**Decision.** Wrap only the LLM call in `try/except` and return an "unavailable" response.

**Why.** A 500 with a stack trace is useless to a supervisor mid-shift, and the frontend
would show a generic error. Returning the same `found: false` shape means a backend outage
looks like "no answer right now — retry or ask your manager," which is the correct
behaviour. Retrieval is deliberately left outside the handler so retrieval bugs still
surface loudly during development.

### D8 — Paths resolved from `__file__`

**Decision.** Resolve `.env`, `index.pkl`, and `static/` relative to the module file.

**Why.** CWD-relative paths break `uvicorn` launched from the repo root and break any
process manager that does not `cd` first. This cost real debugging time and is a
one-line-per-path fix.

---

## 6. Known Limitations

Documented rather than hidden, because an assessment of a safety tool should be honest
about its edges.

1. **Revision handling is not implemented.** Rev 03 and Rev 04 of `QC-SOP-014` are both
   indexed. Rev 04 won on the tested queries by score, not by design. A production system
   must filter to the latest effective revision per document number.
2. **Over-abstention on yes/no framing (Q3).** `SAF-SOP-003` §6 answers the absent-owner
   question, and the correct passage ranks #1 (0.711), but the model declines because the
   question asks for permission and the passage gives a conditional procedure. Rephrasing
   resolves it. A production fix would add a "answer from the best passage even if the
   framing is yes/no" instruction, or a query-rewriting step.
3. **No multilingual support.**
4. **No incremental indexing, auth, or rate limiting.** The index is in memory and loaded
   at startup.
5. **Answer quality depends on the chosen model.** JSON-mode and instruction-following vary
   by backend.

---

## 7. Evaluation Approach

Testing deliberately spans three categories, per the critique:

| Category | Question | What it tests |
|---|---|---|
| Direct lookup | Q1 (sample count) | Table retrieval and numeric accuracy |
| Conditional rule | Q2 (missing MTC) | Retaining the condition, not just the action |
| Conditional permission | Q3 (absent lock owner) | Abstention vs. conditional reasoning |
| Multi-hop numeric | Q4 (frequency + limit) | Combining two facts into one answer |
| Out of corpus | Q5 (titanium cutting speed) | Refusal instead of fabrication |

The most informative result is Q5: a system that invents a cutting speed would look more
capable and be more dangerous. Q3 is the most informative failure: retrieval succeeded and
the model still declined, which isolates the problem to abstention policy rather than
retrieval.