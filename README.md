# Brand Guardian AI

An LLMOps portfolio project that audits video ads for brand/regulatory compliance
(e.g. FTC disclosure rules) using a RAG pipeline built with LangGraph.

Given a YouTube video URL, the system:
1. Extracts a transcript (captions, or local Whisper if none exist)
2. Retrieves relevant compliance rules from a knowledge base (FTC influencer
   guidelines, ad specs, etc.)
3. Asks an LLM to judge the content against those rules
4. Returns a structured PASS/FAIL compliance report with specific violations

## Why this project

Brand safety and regulatory compliance (e.g. FTC disclosure rules) are usually
checked by hand, which doesn't scale as ad volume grows. This project explores how
far an LLM + RAG pipeline can go in automating that first pass, while also serving
as a hands-on **LLMOps** portfolio piece — not just calling an LLM API, but building
out the surrounding practice: retrieval, evaluation, CI/CD, infra-as-code, and
observability around it. It originally ran on paid Azure services and was rebuilt
from scratch on a free-tier AWS stack to keep it accessible to run and extend.


## Architecture

```
YouTube URL
     │
     ▼
[ Indexer Node ]  → captions API, or local Whisper fallback
     │  transcript
     ▼
[ Auditor Node ]  → RAG retrieval (pgvector) + LLM (Groq) judges against
     │                compliance rules (FTC PDFs, etc.)
     ▼
Compliance Report (PASS/FAIL + violations list)
```

Orchestrated as a two-node [LangGraph](https://github.com/langchain-ai/langgraph)
workflow (`backend/src/graph/workflow.py`), exposed via a FastAPI endpoint
(`backend/src/api/server.py`) and a CLI entry point (`main.py`).

## Known limitations

- **No OCR**: on-screen-only disclosures (e.g. a "#ad" banner shown as text but never
  spoken) won't be caught — only spoken claims in the transcript are audited.
- **Short-form videos only**: the sync Lambda deploy target is sized for ~30–60 second
  ads. Longer-form video would need an async (SQS + worker) pipeline, which was
  considered and deliberately deferred — see [PROBLEMS.md](./PROBLEMS.md).

## Tech stack

**AI / Pipeline**
- Orchestration: LangGraph, LangChain
- LLM: Groq
- Embeddings: sentence-transformers (local, `all-MiniLM-L6-v2`)
- Vector store: pgvector on AWS RDS (planned; currently Chroma)
- Transcription: youtube-transcript-api, faster-whisper (fallback)

**App / API**
- API: FastAPI
- CLI: `main.py`
- Demo frontend: Streamlit (planned)

**Infra / LLMOps (planned)**
- Compute: AWS Lambda + API Gateway
- Storage: AWS S3
- IaC: Terraform
- CI/CD: GitHub Actions
- Evaluation: RAGAS or DeepEval
- Tracing: LangSmith
- Monitoring: AWS CloudWatch

**Tooling**
- Package management: uv

## Setup

```bash
# Install dependencies
uv sync

# Configure environment
cp .env.example .env   # then fill in GROQ_API_KEY, etc.

# Index the compliance knowledge base (PDFs in backend/data/)
uv run python -m backend.scripts.index_documents

# Run a one-off audit via CLI
uv run python main.py

# Or run the API server
uv run uvicorn backend.src.api.server:app --reload
```

## Project layout

```
backend/
  data/                 compliance rule PDFs (knowledge base source)
  scripts/index_documents.py   loads PDFs into the vector store
  src/
    api/server.py        FastAPI app
    graph/                LangGraph workflow, nodes, state schema
    services/             transcript extraction service
main.py                  CLI entry point
PROBLEMS.md               log of blockers/decisions hit along the way
```
