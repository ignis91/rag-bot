# rag-bot

Semantic search and Q&A over Chinese-language lecture videos.
A transcription-first RAG pipeline: video → transcript → chunks → embeddings → vector search.

## Stack

| Layer          | Tool                                    |
|----------------|-----------------------------------------|
| Transcription  | faster-whisper (CTranslate2, CUDA, int8)|
| Embeddings     | BGE-M3 via sentence-transformers        |
| Vector store   | pgvector (PostgreSQL)                   |
| API            | FastAPI                                 |
| Frontend       | Next.js                                 |

## Requirements

- Python, managed via [uv](https://github.com/astral-sh/uv)
- CUDA 12.4, GPU with ≥3 GB VRAM
- PostgreSQL with the pgvector extension

## Setup

1. `uv sync`
2. Copy `.env.example` → `.env` and fill in the values.
3. The first run downloads BGE-M3 from HuggingFace and requires network access.
   Once the model is cached locally, set `HF_HUB_OFFLINE=1` to run fully offline
   (no revalidation calls to the Hub, no rate-limit exposure).

## Pipeline

Run every stage from the repository root as a module:

```bash
python3 -m app.ingestion.transcribe   # video (.mp4) → transcript JSON
python3 -m app.ingestion.chunk        # transcript → overlapping chunks
# embed → not implemented yet
# vector storage → not implemented yet
# retrieval / API → not implemented yet
```

Both ingestion stages are idempotent: rerunning skips work whose output is
already up to date, and rebuilds only when the source has changed.

## Project structure

```
app/
  core/        # config (constants), logging setup
  ingestion/   # transcribe, chunk  (embed pending)
  retrieval/   # pending
  api/         # pending
```

## Status

| Stage      | State                                                        |
|------------|-------------------------------------------------------------|
| transcribe | ✓ atomic writes, existence-based idempotency                |
| chunk      | ✓ sliding-window overlap, mtime-based cache invalidation    |
| embed      | not started (BGE-M3 → vector)                               |
| storage    | not started (pgvector)                                       |
| retrieval  | not started                                                  |
| API        | not started                                                  |

---

*Work in progress. Portfolio project — architecture and pipeline built stage by stage.*