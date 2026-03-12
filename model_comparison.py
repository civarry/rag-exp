"""
Multi-Model Comparison for Jira Ticket RAG.

Compares different embedding models using the best chunking strategy
(summary + description + comments) identified from the chunking experiments.

Models tested:
1. sentence-transformers/all-MiniLM-L6-v2     (384d, 22M params - baseline)
2. BAAI/bge-small-en-v1.5                     (384d, 33M params)
3. BAAI/bge-base-en-v1.5                      (768d, 109M params)
4. nomic-ai/nomic-embed-text-v1.5             (768d, 137M params)
5. sentence-transformers/all-mpnet-base-v2     (768d, 109M params)

Each model is tested with all chunking strategies, with and without reranking.
"""

import os
import json
import time
import hashlib
from typing import Dict, List, Tuple

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
EMBEDDING_MODELS = [
    {
        "name": "all-MiniLM-L6-v2",
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "dims": 384,
        "max_tokens": 256,
        "prefix_query": "",
        "prefix_doc": "",
    },
    {
        "name": "bge-small-en-v1.5",
        "model_id": "BAAI/bge-small-en-v1.5",
        "dims": 384,
        "max_tokens": 512,
        "prefix_query": "Represent this sentence for searching relevant passages: ",
        "prefix_doc": "",
    },
    {
        "name": "bge-base-en-v1.5",
        "model_id": "BAAI/bge-base-en-v1.5",
        "dims": 768,
        "max_tokens": 512,
        "prefix_query": "Represent this sentence for searching relevant passages: ",
        "prefix_doc": "",
    },
    {
        "name": "nomic-embed-text-v1.5",
        "model_id": "nomic-ai/nomic-embed-text-v1.5",
        "dims": 768,
        "max_tokens": 8192,
        "prefix_query": "search_query: ",
        "prefix_doc": "search_document: ",
    },
    {
        "name": "all-mpnet-base-v2",
        "model_id": "sentence-transformers/all-mpnet-base-v2",
        "dims": 768,
        "max_tokens": 384,
        "prefix_query": "",
        "prefix_doc": "",
    },
]

CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
K_VALUES = [1, 3, 5]
RERANK_RETRIEVE_N = 20

LLM_CACHE_FILE = "./llm_cache.json"


# ============================================================
# LLM Cache
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
    time.sleep(0.5)
    return result


# ============================================================
# Chunking Strategies (same as experiments.py)
# ============================================================
def chunk_summary_only(ticket: dict) -> List[Dict]:
    return [{"text": ticket["summary"], "ticket_key": ticket["key"], "chunk_type": "summary"}]


def chunk_summary_desc(ticket: dict) -> List[Dict]:
    text = f"{ticket['summary']}\n\n{ticket['description'].strip()}"
    return [{"text": text, "ticket_key": ticket["key"], "chunk_type": "summary_desc"}]


def chunk_summary_desc_comments(ticket: dict) -> List[Dict]:
    parts = [ticket["summary"], ticket["description"].strip()]
    if ticket.get("comments"):
        parts.append("Comments:\n" + "\n---\n".join(ticket["comments"]))
    text = "\n\n".join(parts)
    return [{"text": text, "ticket_key": ticket["key"], "chunk_type": "summary_desc_comments"}]


def chunk_metadata_enriched(ticket: dict) -> List[Dict]:
    metadata_prefix = (
        f"[{ticket['type']}] [{ticket['priority']}] "
        f"[{', '.join(ticket.get('labels', []))}] "
        f"[{', '.join(ticket.get('components', []))}]"
    )
    text = f"{metadata_prefix}\n{ticket['summary']}\n\n{ticket['description'].strip()}"
    return [{"text": text, "ticket_key": ticket["key"], "chunk_type": "metadata_enriched"}]


def chunk_llm_normalized_full(ticket: dict, client: Groq, cache: dict) -> List[Dict]:
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
    return [{"text": normalized, "ticket_key": ticket["key"], "chunk_type": "llm_normalized_full"}]


def chunk_contextual_retrieval(ticket: dict, client: Groq, cache: dict) -> List[Dict]:
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
    return [{"text": full_text, "ticket_key": ticket["key"], "chunk_type": "contextual"}]


CHUNKING_STRATEGIES = [
    ("summary_only", chunk_summary_only, False),
    ("summary_desc", chunk_summary_desc, False),
    ("summary_desc_comments", chunk_summary_desc_comments, False),
    ("metadata_enriched", chunk_metadata_enriched, False),
    ("llm_normalized_full", chunk_llm_normalized_full, True),
    ("contextual_retrieval", chunk_contextual_retrieval, True),
]


# ============================================================
# Multi-Model Experiment Runner
# ============================================================
class ModelComparisonRunner:
    def __init__(self):
        self.cross_encoder = None
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_TOKEN"))
        self.llm_cache = load_llm_cache()
        self.chroma_client = chromadb.Client()
        self.all_results = {}  # {model_name: {strategy_name: {no_rerank: ..., with_rerank: ...}}}
        # Pre-generate all chunks (shared across models)
        self._chunk_cache = {}

    def _load_cross_encoder(self):
        if self.cross_encoder is None:
            print("Loading cross-encoder reranker...")
            self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)

    def _generate_chunks(self, strategy_name: str, chunking_fn, use_llm: bool) -> List[Dict]:
        if strategy_name in self._chunk_cache:
            return self._chunk_cache[strategy_name]

        chunks = []
        for ticket in MOCK_TICKETS:
            if use_llm:
                chunks.extend(chunking_fn(ticket, self.groq_client, self.llm_cache))
            else:
                chunks.extend(chunking_fn(ticket))
        self._chunk_cache[strategy_name] = chunks
        return chunks

    def _embed_texts(self, model: SentenceTransformer, texts: List[str]) -> List[List[float]]:
        embeddings = model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def _create_collection(self, name: str):
        safe_name = name.replace(" ", "_").replace("/", "_").replace("-", "_")[:60]
        try:
            self.chroma_client.delete_collection(safe_name)
        except Exception:
            pass
        return self.chroma_client.create_collection(name=safe_name, metadata={"hnsw:space": "cosine"})

    def _retrieve(self, model: SentenceTransformer, collection, query: str, n: int,
                  query_prefix: str = "") -> List[Tuple[str, float, str]]:
        prefixed_query = f"{query_prefix}{query}" if query_prefix else query
        query_embedding = self._embed_texts(model, [prefixed_query])[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n, collection.count()),
            include=["documents", "metadatas", "distances"]
        )
        output = []
        seen_keys = set()
        for meta, dist, doc in zip(results["metadatas"][0], results["distances"][0], results["documents"][0]):
            ticket_key = meta["ticket_key"]
            if ticket_key not in seen_keys:
                seen_keys.add(ticket_key)
                output.append((ticket_key, dist, doc))
        return output

    def _rerank(self, query: str, results: List[Tuple[str, float, str]], top_k: int) -> List[Tuple[str, float, str]]:
        if not results:
            return results
        self._load_cross_encoder()
        pairs = [(query, doc) for _, _, doc in results]
        scores = self.cross_encoder.predict(pairs)
        scored = [(results[i][0], float(scores[i]), results[i][2]) for i in range(len(results))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _evaluate_strategy(self, model: SentenceTransformer, model_config: dict,
                           strategy_name: str, chunks: List[Dict]) -> Dict:
        """Evaluate a single model+strategy combination."""
        collection_name = f"{model_config['name']}_{strategy_name}"
        collection = self._create_collection(collection_name)

        # Apply document prefix if needed
        doc_prefix = model_config["prefix_doc"]
        texts = [f"{doc_prefix}{c['text']}" if doc_prefix else c["text"] for c in chunks]
        ids = [f"{c['ticket_key']}_{c['chunk_type']}_{i}" for i, c in enumerate(chunks)]
        metadatas = [{"ticket_key": c["ticket_key"], "chunk_type": c["chunk_type"]} for c in chunks]
        embeddings = self._embed_texts(model, texts)

        batch_size = 100
        for i in range(0, len(texts), batch_size):
            end = min(i + batch_size, len(texts))
            collection.add(ids=ids[i:end], documents=[c["text"] for c in chunks[i:end]],
                           embeddings=embeddings[i:end], metadatas=metadatas[i:end])

        query_prefix = model_config["prefix_query"]

        # No reranking
        no_rerank_results = []
        for gt in GROUND_TRUTH_QUERIES:
            retrieved = self._retrieve(model, collection, gt["query"], RERANK_RETRIEVE_N, query_prefix)
            retrieved_keys = [r[0] for r in retrieved]
            metrics = evaluate_retrieval(retrieved_keys, gt["relevant_keys"], gt["primary_key"], K_VALUES)
            no_rerank_results.append(metrics)

        # With reranking
        rerank_results = []
        for gt in GROUND_TRUTH_QUERIES:
            retrieved = self._retrieve(model, collection, gt["query"], RERANK_RETRIEVE_N, query_prefix)
            reranked = self._rerank(gt["query"], retrieved, max(K_VALUES))
            retrieved_keys = [r[0] for r in reranked]
            metrics = evaluate_retrieval(retrieved_keys, gt["relevant_keys"], gt["primary_key"], K_VALUES)
            rerank_results.append(metrics)

        return {
            "no_rerank": aggregate_metrics(no_rerank_results),
            "with_rerank": aggregate_metrics(rerank_results),
        }

    def run_all(self):
        """Run all model x strategy combinations."""
        start_time = time.time()

        # Pre-generate all chunks first
        print("Pre-generating chunks for all strategies...")
        for strategy_name, chunking_fn, use_llm in CHUNKING_STRATEGIES:
            chunks = self._generate_chunks(strategy_name, chunking_fn, use_llm)
            print(f"  {strategy_name}: {len(chunks)} chunks, avg {np.mean([len(c['text']) for c in chunks]):.0f} chars")

        # Run each model
        for model_config in EMBEDDING_MODELS:
            model_name = model_config["name"]
            print(f"\n{'='*80}")
            print(f"MODEL: {model_name} ({model_config['dims']}d, max {model_config['max_tokens']} tokens)")
            print(f"{'='*80}")

            print(f"  Loading {model_config['model_id']}...")
            model = SentenceTransformer(model_config["model_id"], trust_remote_code=True)

            self.all_results[model_name] = {}

            for strategy_name, chunking_fn, use_llm in CHUNKING_STRATEGIES:
                print(f"  Strategy: {strategy_name}...", end=" ", flush=True)
                chunks = self._generate_chunks(strategy_name, chunking_fn, use_llm)
                result = self._evaluate_strategy(model, model_config, strategy_name, chunks)
                self.all_results[model_name][strategy_name] = result

                nr = result["no_rerank"]
                wr = result["with_rerank"]
                print(f"Hit@1={nr['hit_rate@1']:.3f} MRR={nr['mrr']:.3f} nDCG@5={nr['ndcg@5']:.3f} | "
                      f"reranked: Hit@1={wr['hit_rate@1']:.3f} MRR={wr['mrr']:.3f} nDCG@5={wr['ndcg@5']:.3f}")

            # Free model memory
            del model

        elapsed = time.time() - start_time
        print(f"\n\nAll model comparisons completed in {elapsed:.1f}s")

        self._print_model_comparison()
        self._print_best_combos()
        self._save_results()

    def _print_model_comparison(self):
        """Print comparison tables."""
        print("\n" + "=" * 120)
        print("MODEL x STRATEGY COMPARISON")
        print("=" * 120)

        for rerank_mode, rerank_label in [("no_rerank", "Without Reranking"), ("with_rerank", "With Cross-Encoder Reranking")]:
            print(f"\n--- {rerank_label} ---")

            # Table: rows = models, columns = strategies, cells = nDCG@5
            strategy_names = [s[0] for s in CHUNKING_STRATEGIES]
            headers = ["Model \\ Strategy"] + strategy_names
            rows = []
            for model_config in EMBEDDING_MODELS:
                model_name = model_config["name"]
                row = [f"{model_name} ({model_config['dims']}d)"]
                for strategy_name in strategy_names:
                    result = self.all_results[model_name][strategy_name][rerank_mode]
                    ndcg = result.get("ndcg@5", 0)
                    row.append(f"{ndcg:.3f}")
                rows.append(row)
            print("\nnDCG@5:")
            print(tabulate(rows, headers=headers, tablefmt="grid"))

            # Same for Hit@1
            rows = []
            for model_config in EMBEDDING_MODELS:
                model_name = model_config["name"]
                row = [f"{model_name} ({model_config['dims']}d)"]
                for strategy_name in strategy_names:
                    result = self.all_results[model_name][strategy_name][rerank_mode]
                    hit1 = result.get("hit_rate@1", 0)
                    row.append(f"{hit1:.3f}")
                rows.append(row)
            print("\nHit Rate@1:")
            print(tabulate(rows, headers=headers, tablefmt="grid"))

            # MRR
            rows = []
            for model_config in EMBEDDING_MODELS:
                model_name = model_config["name"]
                row = [f"{model_name} ({model_config['dims']}d)"]
                for strategy_name in strategy_names:
                    result = self.all_results[model_name][strategy_name][rerank_mode]
                    mrr = result.get("mrr", 0)
                    row.append(f"{mrr:.3f}")
                rows.append(row)
            print("\nMRR:")
            print(tabulate(rows, headers=headers, tablefmt="grid"))

    def _print_best_combos(self):
        """Find and print the best model+strategy combinations."""
        print("\n" + "=" * 80)
        print("TOP 10 BEST COMBINATIONS (by nDCG@5)")
        print("=" * 80)

        combos = []
        for model_name, strategies in self.all_results.items():
            for strategy_name, result in strategies.items():
                for mode in ["no_rerank", "with_rerank"]:
                    combos.append({
                        "model": model_name,
                        "strategy": strategy_name,
                        "rerank": mode == "with_rerank",
                        "ndcg@5": result[mode].get("ndcg@5", 0),
                        "hit@1": result[mode].get("hit_rate@1", 0),
                        "hit@5": result[mode].get("hit_rate@5", 0),
                        "mrr": result[mode].get("mrr", 0),
                        "recall@5": result[mode].get("recall@5", 0),
                    })

        combos.sort(key=lambda x: x["ndcg@5"], reverse=True)

        headers = ["Rank", "Model", "Strategy", "Rerank", "nDCG@5", "Hit@1", "Hit@5", "MRR", "R@5"]
        rows = []
        for i, c in enumerate(combos[:10]):
            rows.append([
                i + 1,
                c["model"],
                c["strategy"],
                "Yes" if c["rerank"] else "No",
                f"{c['ndcg@5']:.4f}",
                f"{c['hit@1']:.3f}",
                f"{c['hit@5']:.3f}",
                f"{c['mrr']:.3f}",
                f"{c['recall@5']:.3f}",
            ])
        print(tabulate(rows, headers=headers, tablefmt="grid"))

    def _save_results(self):
        output_file = "model_comparison_results.json"
        with open(output_file, "w") as f:
            json.dump(self.all_results, f, indent=2, default=str)
        print(f"\nFull results saved to {output_file}")


if __name__ == "__main__":
    runner = ModelComparisonRunner()
    runner.run_all()
