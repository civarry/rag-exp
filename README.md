# rag-exp

A collection of experiments comparing retrieval-augmented generation (RAG) strategies for a single concrete use case: a new support/Jira ticket comes in, and the system needs to find similar historical tickets and suggest a resolution based on how they were fixed.

Rather than building one RAG pipeline and hoping it works, this repo runs several competing approaches head-to-head against the same mock ticket dataset and scores them, so the tradeoffs are measured instead of assumed.

## What it tests

- **Chunking strategies** (`experiments.py`) — summary-only, summary + description, summary + description + comments, per-field embeddings, LLM-normalized text, contextual retrieval (prepending LLM-generated context to each chunk), and metadata-enriched embeddings.
- **Retrieval architectures** (`final_rag_pipeline.py`) — adaptive blob/chunk splitting by ticket length, HyDE (embedding a hypothetical resolution and searching against it), multi-vector retrieval (separate embeddings for the problem description and the resolution), HyDE + multi-vector combined, cross-encoder reranking, and query rewriting before retrieval.
- **Multi-vector production design** (`multi_vector_rag.py`) — documents the reasoning for why matching "problem language" to "problem language" (rather than problem-to-resolution) improves retrieval accuracy, plus a sketch of an incremental re-embedding sync using per-field content hashes.
- **Long-document handling** (`long_doc_answer_accuracy.py`, `overlap_answer_accuracy.py`) — chunk overlap and long-ticket accuracy experiments.
- **Answer quality** (`answer_accuracy.py`, `metrics.py`) — LLM-as-judge scoring of generated answers on correctness, completeness, and hallucination, plus retrieval accuracy against ground-truth ticket mappings.

Each experiment run writes its results to a corresponding `*_results.json` file for comparison.

## Stack

Python, ChromaDB for vector storage, `sentence-transformers` for embeddings (`all-MiniLM-L6-v2`) and cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`), and the Groq API (Llama 3.1/3.3 models) for generation and LLM-as-judge scoring. Mock ticket data (`mock_jira_tickets.py`, `mock_long_tickets.py`) simulates a realistic e-commerce project ("ShopFlow") with varying ticket quality — clean vs. noisy descriptions, with and without comment threads.

## Setup

```
pip install -r requirements.txt
```

Requires a `GROQ_API_KEY` in a `.env` file (loaded via `python-dotenv`). Run any experiment script directly, e.g. `python final_rag_pipeline.py`.
