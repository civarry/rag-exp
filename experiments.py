"""
RAG Experiment Runner for Jira Ticket Retrieval.

Tests different chunking, embedding, and retrieval strategies to find the best
approach for embedding Jira tickets.

Experiments:
1. summary_only              - Embed only the ticket summary
2. summary_desc              - Embed summary + description
3. summary_desc_comments     - Embed summary + description + comments
4. fields_separated          - Embed each field separately (linked by ticket key)
5. llm_normalized            - LLM-normalize summary+desc before embedding
6. llm_normalized_full       - LLM-normalize summary+desc+comments before embedding
7. contextual_retrieval      - Prepend LLM-generated context to each chunk
8. metadata_enriched         - Prepend ticket metadata (type, priority, labels) to text

Each experiment is also tested with and without cross-encoder reranking.
"""

import os
import json
import time
import hashlib
from typing import Dict, List, Tuple, Optional

import chromadb
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
from tabulate import tabulate

from mock_jira_tickets import MOCK_TICKETS, GROUND_TRUTH_QUERIES
from metrics import evaluate_retrieval, aggregate_metrics

load_dotenv()

# ============================================================
# Configuration
# ============================================================
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
CHROMA_PATH = "./chroma_db"
K_VALUES = [1, 3, 5]
RERANK_RETRIEVE_N = 20  # Retrieve this many, then rerank to top-k

# LLM prompt cache for normalized/contextual text
LLM_CACHE_FILE = "./llm_cache.json"


# ============================================================
# Model Loading
# ============================================================
def load_models():
    """Load embedding model and cross-encoder."""
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Loading cross-encoder reranker...")
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
    return embedding_model, cross_encoder


def get_groq_client():
    """Get Groq API client."""
    api_key = os.getenv("GROQ_API_TOKEN")
    if not api_key:
        raise ValueError("GROQ_API_TOKEN not found in environment")
    return Groq(api_key=api_key)


# ============================================================
# LLM Cache (avoid redundant API calls)
# ============================================================
def load_llm_cache() -> dict:
    if os.path.exists(LLM_CACHE_FILE):
        with open(LLM_CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_llm_cache(cache: dict):
    with open(LLM_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def cache_key(prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()


def llm_call(client: Groq, prompt: str, cache: dict) -> str:
    """Make an LLM call with caching."""
    key = cache_key(prompt)
    if key in cache:
        return cache[key]

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024,
    )
    result = response.choices[0].message.content.strip()
    cache[key] = result
    save_llm_cache(cache)
    # Small delay to respect rate limits
    time.sleep(0.5)
    return result


# ============================================================
# Chunking Strategies
# ============================================================
def chunk_summary_only(ticket: dict) -> List[Dict]:
    """Strategy 1: Embed only the summary."""
    return [{
        "text": ticket["summary"],
        "ticket_key": ticket["key"],
        "chunk_type": "summary",
    }]


def chunk_summary_desc(ticket: dict) -> List[Dict]:
    """Strategy 2: Summary + Description concatenated."""
    text = f"{ticket['summary']}\n\n{ticket['description'].strip()}"
    return [{
        "text": text,
        "ticket_key": ticket["key"],
        "chunk_type": "summary_desc",
    }]


def chunk_summary_desc_comments(ticket: dict) -> List[Dict]:
    """Strategy 3: Summary + Description + All Comments."""
    parts = [ticket["summary"], ticket["description"].strip()]
    if ticket.get("comments"):
        parts.append("Comments:\n" + "\n---\n".join(ticket["comments"]))
    text = "\n\n".join(parts)
    return [{
        "text": text,
        "ticket_key": ticket["key"],
        "chunk_type": "summary_desc_comments",
    }]


def chunk_fields_separated(ticket: dict) -> List[Dict]:
    """Strategy 4: Each field as a separate chunk, linked by key."""
    chunks = []
    # Summary chunk
    chunks.append({
        "text": f"[{ticket['key']}] {ticket['summary']}",
        "ticket_key": ticket["key"],
        "chunk_type": "summary",
    })
    # Description chunk
    if ticket.get("description", "").strip():
        chunks.append({
            "text": f"[{ticket['key']}] {ticket['description'].strip()}",
            "ticket_key": ticket["key"],
            "chunk_type": "description",
        })
    # Each comment as separate chunk
    for i, comment in enumerate(ticket.get("comments", [])):
        chunks.append({
            "text": f"[{ticket['key']}] Comment: {comment}",
            "ticket_key": ticket["key"],
            "chunk_type": f"comment_{i}",
        })
    return chunks


def chunk_metadata_enriched(ticket: dict) -> List[Dict]:
    """Strategy 8: Prepend metadata to summary+description."""
    metadata_prefix = (
        f"[{ticket['type']}] [{ticket['priority']}] "
        f"[{', '.join(ticket.get('labels', []))}] "
        f"[{', '.join(ticket.get('components', []))}]"
    )
    text = f"{metadata_prefix}\n{ticket['summary']}\n\n{ticket['description'].strip()}"
    return [{
        "text": text,
        "ticket_key": ticket["key"],
        "chunk_type": "metadata_enriched",
    }]


def chunk_llm_normalized(ticket: dict, client: Groq, cache: dict) -> List[Dict]:
    """Strategy 5: LLM-normalize summary+description before embedding."""
    raw_text = f"Summary: {ticket['summary']}\n\nDescription: {ticket['description'].strip()}"
    prompt = f"""Rewrite the following Jira ticket into a clear, concise technical summary.
Remove any formatting artifacts, redundancy, or boilerplate. Keep all technical details,
error codes, and specific information. Write in plain prose, 2-4 sentences max.

Ticket ({ticket['key']}, {ticket['type']}, {ticket['priority']}):
{raw_text}

Concise technical summary:"""

    normalized = llm_call(client, prompt, cache)
    return [{
        "text": normalized,
        "ticket_key": ticket["key"],
        "chunk_type": "llm_normalized",
    }]


def chunk_llm_normalized_full(ticket: dict, client: Groq, cache: dict) -> List[Dict]:
    """Strategy 6: LLM-normalize summary+description+comments."""
    parts = [f"Summary: {ticket['summary']}", f"Description: {ticket['description'].strip()}"]
    if ticket.get("comments"):
        parts.append("Comments:\n" + "\n".join(f"- {c}" for c in ticket["comments"]))
    raw_text = "\n\n".join(parts)

    prompt = f"""Rewrite the following Jira ticket (including comments) into a clear, concise
technical summary. Include the resolution or current status if mentioned in comments.
Remove formatting artifacts and boilerplate. Keep all technical details. Write in plain
prose, 3-5 sentences max.

Ticket ({ticket['key']}, {ticket['type']}, {ticket['priority']}):
{raw_text}

Concise technical summary:"""

    normalized = llm_call(client, prompt, cache)
    return [{
        "text": normalized,
        "ticket_key": ticket["key"],
        "chunk_type": "llm_normalized_full",
    }]


def chunk_contextual_retrieval(ticket: dict, client: Groq, cache: dict) -> List[Dict]:
    """Strategy 7: Prepend LLM-generated context to the chunk (Anthropic-style)."""
    raw_text = f"{ticket['summary']}\n\n{ticket['description'].strip()}"
    if ticket.get("comments"):
        raw_text += "\n\nComments:\n" + "\n---\n".join(ticket["comments"])

    prompt = f"""Given this Jira ticket, write a brief 1-2 sentence context prefix that
situates this ticket within the broader project. Mention the ticket key, the type of
issue, which system/component it affects, and the core problem or feature. This prefix
will be prepended to the ticket text for embedding in a search system.

Ticket:
Key: {ticket['key']}
Type: {ticket['type']}
Priority: {ticket['priority']}
Labels: {', '.join(ticket.get('labels', []))}
Components: {', '.join(ticket.get('components', []))}

Content:
{raw_text[:1000]}

Context prefix (1-2 sentences only):"""

    context = llm_call(client, prompt, cache)
    full_text = f"{context}\n\n{ticket['summary']}\n\n{ticket['description'].strip()}"
    return [{
        "text": full_text,
        "ticket_key": ticket["key"],
        "chunk_type": "contextual",
    }]


# ============================================================
# Experiment Runner
# ============================================================
class ExperimentRunner:
    def __init__(self):
        self.embedding_model, self.cross_encoder = load_models()
        self.groq_client = get_groq_client()
        self.llm_cache = load_llm_cache()
        self.chroma_client = chromadb.Client()  # In-memory for experiments
        self.results = {}

    def _create_collection(self, name: str) -> chromadb.Collection:
        """Create or get a ChromaDB collection."""
        # Delete if exists
        try:
            self.chroma_client.delete_collection(name)
        except Exception:
            pass
        return self.chroma_client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using the sentence transformer model."""
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def _index_chunks(self, collection: chromadb.Collection, chunks: List[Dict]):
        """Index chunks into a ChromaDB collection."""
        texts = [c["text"] for c in chunks]
        ids = [f"{c['ticket_key']}_{c['chunk_type']}_{i}" for i, c in enumerate(chunks)]
        metadatas = [{"ticket_key": c["ticket_key"], "chunk_type": c["chunk_type"]} for c in chunks]
        embeddings = self._embed_texts(texts)

        # Batch insert (ChromaDB has batch size limits)
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            end = min(i + batch_size, len(texts))
            collection.add(
                ids=ids[i:end],
                documents=texts[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end],
            )

    def _retrieve(self, collection: chromadb.Collection, query: str, n: int) -> List[Tuple[str, float, str]]:
        """Retrieve top-n results. Returns list of (ticket_key, distance, document)."""
        query_embedding = self._embed_texts([query])[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n, collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        output = []
        seen_keys = set()
        for key, dist, doc in zip(
            results["metadatas"][0],
            results["distances"][0],
            results["documents"][0]
        ):
            ticket_key = key["ticket_key"]
            # Deduplicate by ticket key (keep best score for field-separated strategy)
            if ticket_key not in seen_keys:
                seen_keys.add(ticket_key)
                output.append((ticket_key, dist, doc))
        return output

    def _rerank(self, query: str, results: List[Tuple[str, float, str]], top_k: int) -> List[Tuple[str, float, str]]:
        """Rerank results using cross-encoder."""
        if not results:
            return results

        pairs = [(query, doc) for _, _, doc in results]
        scores = self.cross_encoder.predict(pairs)

        # Combine with original results and sort by cross-encoder score (higher is better)
        scored = [(results[i][0], float(scores[i]), results[i][2]) for i in range(len(results))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def run_experiment(self, name: str, chunking_fn, use_llm: bool = False) -> Dict:
        """Run a single experiment configuration."""
        print(f"\n{'='*60}")
        print(f"Experiment: {name}")
        print(f"{'='*60}")

        # Step 1: Generate chunks
        print("  Generating chunks...")
        all_chunks = []
        for ticket in MOCK_TICKETS:
            if use_llm:
                chunks = chunking_fn(ticket, self.groq_client, self.llm_cache)
            else:
                chunks = chunking_fn(ticket)
            all_chunks.extend(chunks)

        print(f"  Total chunks: {len(all_chunks)}")
        avg_len = np.mean([len(c["text"]) for c in all_chunks])
        print(f"  Avg chunk length: {avg_len:.0f} chars")

        # Step 2: Index
        print("  Indexing into ChromaDB...")
        collection = self._create_collection(name.replace(" ", "_"))
        self._index_chunks(collection, all_chunks)

        # Step 3: Evaluate - no reranking
        print("  Evaluating (no reranking)...")
        no_rerank_results = []
        for gt in GROUND_TRUTH_QUERIES:
            retrieved = self._retrieve(collection, gt["query"], RERANK_RETRIEVE_N)
            retrieved_keys = [r[0] for r in retrieved]
            metrics = evaluate_retrieval(
                retrieved_keys, gt["relevant_keys"], gt["primary_key"], K_VALUES
            )
            no_rerank_results.append(metrics)

        no_rerank_agg = aggregate_metrics(no_rerank_results)

        # Step 4: Evaluate - with reranking
        print("  Evaluating (with cross-encoder reranking)...")
        rerank_results = []
        for gt in GROUND_TRUTH_QUERIES:
            retrieved = self._retrieve(collection, gt["query"], RERANK_RETRIEVE_N)
            reranked = self._rerank(gt["query"], retrieved, max(K_VALUES))
            retrieved_keys = [r[0] for r in reranked]
            metrics = evaluate_retrieval(
                retrieved_keys, gt["relevant_keys"], gt["primary_key"], K_VALUES
            )
            rerank_results.append(metrics)

        rerank_agg = aggregate_metrics(rerank_results)

        result = {
            "name": name,
            "num_chunks": len(all_chunks),
            "avg_chunk_len": avg_len,
            "no_rerank": no_rerank_agg,
            "with_rerank": rerank_agg,
        }
        self.results[name] = result
        self._print_result(result)
        return result

    def _print_result(self, result: Dict):
        """Print a single experiment result."""
        print(f"\n  Results for: {result['name']}")
        print(f"  Chunks: {result['num_chunks']}, Avg Length: {result['avg_chunk_len']:.0f}")

        headers = ["Metric", "No Rerank", "With Rerank", "Delta"]
        rows = []
        for key in sorted(result["no_rerank"].keys()):
            nr = result["no_rerank"][key]
            wr = result["with_rerank"][key]
            delta = wr - nr
            rows.append([key, f"{nr:.4f}", f"{wr:.4f}", f"{delta:+.4f}"])
        print(tabulate(rows, headers=headers, tablefmt="grid"))

    def run_all_experiments(self):
        """Run all experiment configurations."""
        start_time = time.time()

        # Non-LLM experiments
        experiments = [
            ("1_summary_only", chunk_summary_only, False),
            ("2_summary_desc", chunk_summary_desc, False),
            ("3_summary_desc_comments", chunk_summary_desc_comments, False),
            ("4_fields_separated", chunk_fields_separated, False),
            ("5_metadata_enriched", chunk_metadata_enriched, False),
        ]

        for name, fn, use_llm in experiments:
            self.run_experiment(name, fn, use_llm)

        # LLM-based experiments
        llm_experiments = [
            ("6_llm_normalized", chunk_llm_normalized, True),
            ("7_llm_normalized_full", chunk_llm_normalized_full, True),
            ("8_contextual_retrieval", chunk_contextual_retrieval, True),
        ]

        for name, fn, use_llm in llm_experiments:
            self.run_experiment(name, fn, use_llm)

        elapsed = time.time() - start_time
        print(f"\n\nAll experiments completed in {elapsed:.1f}s")

        self._print_comparison()
        self._save_results()

    def _print_comparison(self):
        """Print side-by-side comparison of all experiments."""
        print("\n" + "=" * 100)
        print("EXPERIMENT COMPARISON - KEY METRICS")
        print("=" * 100)

        # Without reranking
        print("\n--- Without Reranking ---")
        headers = ["Experiment", "Hit@1", "Hit@3", "Hit@5", "MRR", "nDCG@5", "P@5", "R@5"]
        rows = []
        for name in sorted(self.results.keys()):
            r = self.results[name]["no_rerank"]
            rows.append([
                name,
                f"{r.get('hit_rate@1', 0):.3f}",
                f"{r.get('hit_rate@3', 0):.3f}",
                f"{r.get('hit_rate@5', 0):.3f}",
                f"{r.get('mrr', 0):.3f}",
                f"{r.get('ndcg@5', 0):.3f}",
                f"{r.get('precision@5', 0):.3f}",
                f"{r.get('recall@5', 0):.3f}",
            ])
        print(tabulate(rows, headers=headers, tablefmt="grid"))

        # With reranking
        print("\n--- With Cross-Encoder Reranking ---")
        rows = []
        for name in sorted(self.results.keys()):
            r = self.results[name]["with_rerank"]
            rows.append([
                name,
                f"{r.get('hit_rate@1', 0):.3f}",
                f"{r.get('hit_rate@3', 0):.3f}",
                f"{r.get('hit_rate@5', 0):.3f}",
                f"{r.get('mrr', 0):.3f}",
                f"{r.get('ndcg@5', 0):.3f}",
                f"{r.get('precision@5', 0):.3f}",
                f"{r.get('recall@5', 0):.3f}",
            ])
        print(tabulate(rows, headers=headers, tablefmt="grid"))

        # Best experiment
        print("\n--- Best Configuration ---")
        best_no_rerank = max(self.results.items(), key=lambda x: x[1]["no_rerank"].get("hit_rate@5", 0))
        best_rerank = max(self.results.items(), key=lambda x: x[1]["with_rerank"].get("hit_rate@5", 0))
        best_mrr = max(self.results.items(), key=lambda x: x[1]["with_rerank"].get("mrr", 0))

        print(f"  Best Hit@5 (no rerank):    {best_no_rerank[0]} = {best_no_rerank[1]['no_rerank']['hit_rate@5']:.3f}")
        print(f"  Best Hit@5 (with rerank):  {best_rerank[0]} = {best_rerank[1]['with_rerank']['hit_rate@5']:.3f}")
        print(f"  Best MRR (with rerank):    {best_mrr[0]} = {best_mrr[1]['with_rerank']['mrr']:.3f}")

    def _save_results(self):
        """Save full results to JSON."""
        output_file = "experiment_results.json"
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nFull results saved to {output_file}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.run_all_experiments()
