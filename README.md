# SCSE '26 — Graded Exercise #3: RAG for the University IT Support Agent

This is the RAG part of Graded Exercise #3. A fine-tuned Qwen3-0.6B model
tends to hallucinate an IT Service Desk phone number because no verified
number exists in its training data. Retrieval-Augmented Generation grounds
every answer in a single knowledge-base file so that the model instead says
it does not know the number.

## Files

| File | Status | Purpose |
|---|---|---|
| `policy_loader.py` | **Do not edit** (provided) | Parses `university_it_policies.txt` into LangChain `Document` objects |
| `university_it_policies.txt` | knowledge base | The single source file (1,196 policy records) |
| `rag_tool.py` | completed | Loads the documents, embeds them with `nomic-embed-text`, stores them in Chroma, and exposes the `search_policy` retrieval tool |
| `agent.py` | completed | qwen3:0.6b agent with the retrieve → draft → review → revise reflection loop and a phone-number hallucination harness |
| `rag.py` | completed | Entry point; asks the graded challenge question (plus sanity questions) |
| `check_data.py` | provided | Verifies the knowledge base reproduces the original JSONL records |

## How it works

1. **Retrieve** — the question is embedded with `nomic-embed-text` and the
   top policy excerpts are retrieved from a Chroma collection
   (`R` of RAG).
2. **Augment** — the excerpts are placed in the model prompt as
   `RETRIEVED CONTEXT` (`A`).
3. **Generate** — `qwen3:0.6b` (served by Ollama) drafts an answer using
   only that context (`G`).
4. **Reflect** (Lecture 11) — the same model reviews the draft as a strict
   reviewer and returns `PASS` or `REVISE`; a revision is produced when
   needed. A deterministic harness additionally rejects any phone number
   that does not appear verbatim in the retrieved context.

## Setup

```powershell
uv venv --python 3.11
.venv\Scripts\activate
uv pip install -r requirements.txt

ollama pull qwen3:0.6b
ollama pull nomic-embed-text
```

## Run

```powershell
python rag.py
```

## Expected result

For the question

> What exact phone number should I call for the University IT Service Desk?

the agent answers that it does not know the number because it is not in the
policy knowledge base. It does **not** invent a number and does not give an
irrelevant answer. Questions that *are* covered by the knowledge base (print
job retention, account lockout, etc.) are still answered normally.
