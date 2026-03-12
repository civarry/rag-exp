"""
Comment Chunking Experiments for Jira Ticket RAG.

Tests different ways to chunk comments specifically, compared against
the previous best strategy (summary_desc_comments as one blob).

Strategies:
1. baseline_blob           - Summary + Desc + Comments all in one chunk (previous best)
2. body_plus_comments_sep  - Body as chunk 1, all comments as chunk 2
3. each_comment_ctx        - Each comment as own chunk with ticket summary prepended
4. each_comment_noisy      - Same as #3 but WITHOUT filtering noise comments
5. noise_filtered           - Each comment as own chunk, skip low-value comments
6. sliding_window           - Groups of 2-3 consecutive comments with overlap
7. parent_child             - Children for search, parent for context (simulated)
8. labeled_fields           - Body + comments with field labels ("Title:", "Description:", "Comment:")
"""

import os
import json
import re
import time
import hashlib
from typing import Dict, List, Tuple

import chromadb
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from tabulate import tabulate

from mock_jira_tickets import MOCK_TICKETS, GROUND_TRUTH_QUERIES
from metrics import evaluate_retrieval, aggregate_metrics

load_dotenv()

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
K_VALUES = [1, 3, 5]
RERANK_RETRIEVE_N = 20

# Patterns for low-value comments
NOISE_PATTERNS = [
    r"^\+1$",
    r"^bump$",
    r"^same here$",
    r"^thx$",
    r"^thanks$",
    r"^thank you$",
    r"^any update\??$",
    r"^following$",
    r"^cc @\w+$",
    r"^@\w+ can you (look at|check) this\??$",
    r"^(Done|done)\.? (PR|pr) #\d+ merged\.?$",
]
NOISE_REGEXES = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]


def is_noise_comment(comment: str) -> bool:
    """Check if a comment is low-value noise."""
    stripped = comment.strip()
    # Very short comments are usually noise
    if len(stripped) < 15:
        for regex in NOISE_REGEXES:
            if regex.match(stripped):
                return True
    return False


# ============================================================
# Chunking Strategies
# ============================================================

def strategy_baseline_blob(ticket: dict) -> List[Dict]:
    """Strategy 1: Everything in one chunk (previous best)."""
    parts = [ticket["summary"], ticket["description"].strip()]
    if ticket.get("comments"):
        parts.append("Comments:\n" + "\n---\n".join(ticket["comments"]))
    text = "\n\n".join(parts)
    return [{"text": text, "ticket_key": ticket["key"], "chunk_type": "blob"}]


def strategy_body_plus_comments_sep(ticket: dict) -> List[Dict]:
    """Strategy 2: Body as one chunk, all comments as another chunk."""
    chunks = []
    # Body chunk
    body = f"{ticket['summary']}\n\n{ticket['description'].strip()}"
    chunks.append({"text": body, "ticket_key": ticket["key"], "chunk_type": "body"})
    # Comments chunk (if any)
    if ticket.get("comments"):
        comments_text = (
            f"[{ticket['key']}] {ticket['summary']}\n\n"
            f"Comments:\n" + "\n---\n".join(ticket["comments"])
        )
        chunks.append({"text": comments_text, "ticket_key": ticket["key"], "chunk_type": "comments_all"})
    return chunks


def strategy_each_comment_with_context(ticket: dict) -> List[Dict]:
    """Strategy 3: Each comment as its own chunk with ticket summary prepended."""
    chunks = []
    # Body chunk
    body = f"{ticket['summary']}\n\n{ticket['description'].strip()}"
    chunks.append({"text": body, "ticket_key": ticket["key"], "chunk_type": "body"})
    # Each comment with context
    for i, comment in enumerate(ticket.get("comments", [])):
        text = f"[{ticket['key']}] {ticket['summary']}\n\nComment: {comment}"
        chunks.append({"text": text, "ticket_key": ticket["key"], "chunk_type": f"comment_{i}"})
    return chunks


def strategy_noise_filtered(ticket: dict) -> List[Dict]:
    """Strategy 5: Each comment as own chunk, but skip noise comments."""
    chunks = []
    # Body chunk
    body = f"{ticket['summary']}\n\n{ticket['description'].strip()}"
    chunks.append({"text": body, "ticket_key": ticket["key"], "chunk_type": "body"})
    # Only meaningful comments
    comment_idx = 0
    for comment in ticket.get("comments", []):
        if not is_noise_comment(comment):
            text = f"[{ticket['key']}] {ticket['summary']}\n\nComment: {comment}"
            chunks.append({"text": text, "ticket_key": ticket["key"], "chunk_type": f"comment_{comment_idx}"})
            comment_idx += 1
    return chunks


def strategy_sliding_window(ticket: dict, window_size: int = 3, overlap: int = 1) -> List[Dict]:
    """Strategy 6: Sliding window over comments, groups of 2-3 with overlap."""
    chunks = []
    # Body chunk
    body = f"{ticket['summary']}\n\n{ticket['description'].strip()}"
    chunks.append({"text": body, "ticket_key": ticket["key"], "chunk_type": "body"})

    comments = ticket.get("comments", [])
    if not comments:
        return chunks

    # If few comments, just put them all in one chunk
    if len(comments) <= window_size:
        text = (
            f"[{ticket['key']}] {ticket['summary']}\n\n"
            f"Comments:\n" + "\n---\n".join(comments)
        )
        chunks.append({"text": text, "ticket_key": ticket["key"], "chunk_type": "comments_window_0"})
        return chunks

    # Sliding window
    step = window_size - overlap
    window_idx = 0
    for start in range(0, len(comments), step):
        end = min(start + window_size, len(comments))
        window_comments = comments[start:end]
        text = (
            f"[{ticket['key']}] {ticket['summary']}\n\n"
            f"Comments:\n" + "\n---\n".join(window_comments)
        )
        chunks.append({"text": text, "ticket_key": ticket["key"], "chunk_type": f"comments_window_{window_idx}"})
        window_idx += 1
        if end >= len(comments):
            break

    return chunks


def strategy_parent_child(ticket: dict) -> List[Dict]:
    """Strategy 7: Parent-child. Children are searched, deduplicated to parent ticket.
    Children = summary chunk + description chunk + each significant comment chunk.
    Each child has ticket summary prepended for context."""
    chunks = []

    # Child: summary
    chunks.append({
        "text": f"[{ticket['key']}] {ticket['summary']}",
        "ticket_key": ticket["key"],
        "chunk_type": "child_summary",
    })

    # Child: description (if substantive)
    desc = ticket.get("description", "").strip()
    if desc and len(desc) > 20:
        chunks.append({
            "text": f"[{ticket['key']}] {ticket['summary']}\n\n{desc}",
            "ticket_key": ticket["key"],
            "chunk_type": "child_description",
        })

    # Child: each significant comment
    comment_idx = 0
    for comment in ticket.get("comments", []):
        if not is_noise_comment(comment):
            chunks.append({
                "text": f"[{ticket['key']}] {ticket['summary']}\n\nComment: {comment}",
                "ticket_key": ticket["key"],
                "chunk_type": f"child_comment_{comment_idx}",
            })
            comment_idx += 1

    return chunks


def strategy_labeled_fields(ticket: dict) -> List[Dict]:
    """Strategy 8: All content with explicit field labels."""
    parts = [
        f"Title: {ticket['summary']}",
        f"Type: {ticket['type']} | Priority: {ticket['priority']}",
        f"Labels: {', '.join(ticket.get('labels', []))}",
        f"Components: {', '.join(ticket.get('components', []))}",
        f"Description: {ticket['description'].strip()}",
    ]
    if ticket.get("comments"):
        for i, comment in enumerate(ticket["comments"]):
            parts.append(f"Comment {i+1}: {comment}")
    text = "\n\n".join(parts)
    return [{"text": text, "ticket_key": ticket["key"], "chunk_type": "labeled"}]


def strategy_noise_filtered_blob(ticket: dict) -> List[Dict]:
    """Strategy 9: Single blob but with noise comments removed."""
    parts = [ticket["summary"], ticket["description"].strip()]
    if ticket.get("comments"):
        meaningful = [c for c in ticket["comments"] if not is_noise_comment(c)]
        if meaningful:
            parts.append("Comments:\n" + "\n---\n".join(meaningful))
    text = "\n\n".join(parts)
    return [{"text": text, "ticket_key": ticket["key"], "chunk_type": "filtered_blob"}]


def strategy_sliding_window_filtered(ticket: dict) -> List[Dict]:
    """Strategy 10: Sliding window but with noise filtered first."""
    chunks = []
    body = f"{ticket['summary']}\n\n{ticket['description'].strip()}"
    chunks.append({"text": body, "ticket_key": ticket["key"], "chunk_type": "body"})

    comments = [c for c in ticket.get("comments", []) if not is_noise_comment(c)]
    if not comments:
        return chunks

    window_size = 3
    overlap = 1

    if len(comments) <= window_size:
        text = (
            f"[{ticket['key']}] {ticket['summary']}\n\n"
            f"Comments:\n" + "\n---\n".join(comments)
        )
        chunks.append({"text": text, "ticket_key": ticket["key"], "chunk_type": "comments_window_0"})
        return chunks

    step = window_size - overlap
    window_idx = 0
    for start in range(0, len(comments), step):
        end = min(start + window_size, len(comments))
        window_comments = comments[start:end]
        text = (
            f"[{ticket['key']}] {ticket['summary']}\n\n"
            f"Comments:\n" + "\n---\n".join(window_comments)
        )
        chunks.append({"text": text, "ticket_key": ticket["key"], "chunk_type": f"comments_window_{window_idx}"})
        window_idx += 1
        if end >= len(comments):
            break

    return chunks


# ============================================================
# Experiment Runner
# ============================================================
ALL_STRATEGIES = [
    ("01_baseline_blob", strategy_baseline_blob),
    ("02_body_plus_comments_sep", strategy_body_plus_comments_sep),
    ("03_each_comment_with_ctx", strategy_each_comment_with_context),
    ("04_noise_filtered_comments", strategy_noise_filtered),
    ("05_sliding_window_3", strategy_sliding_window),
    ("06_parent_child", strategy_parent_child),
    ("07_labeled_fields", strategy_labeled_fields),
    ("08_noise_filtered_blob", strategy_noise_filtered_blob),
    ("09_sliding_window_filtered", strategy_sliding_window_filtered),
]


class CommentChunkingRunner:
    def __init__(self):
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Loading cross-encoder reranker...")
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
        self.chroma_client = chromadb.Client()
        self.results = {}

    def _create_collection(self, name: str):
        safe_name = name[:60]
        try:
            self.chroma_client.delete_collection(safe_name)
        except Exception:
            pass
        return self.chroma_client.create_collection(name=safe_name, metadata={"hnsw:space": "cosine"})

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_model.encode(texts, show_progress_bar=False).tolist()

    def _index_chunks(self, collection, chunks: List[Dict]):
        texts = [c["text"] for c in chunks]
        ids = [f"{c['ticket_key']}_{c['chunk_type']}_{i}" for i, c in enumerate(chunks)]
        metadatas = [{"ticket_key": c["ticket_key"], "chunk_type": c["chunk_type"]} for c in chunks]
        embeddings = self._embed_texts(texts)
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            end = min(i + batch_size, len(texts))
            collection.add(ids=ids[i:end], documents=texts[i:end],
                           embeddings=embeddings[i:end], metadatas=metadatas[i:end])

    def _retrieve(self, collection, query: str, n: int) -> List[Tuple[str, float, str]]:
        query_embedding = self._embed_texts([query])[0]
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
        pairs = [(query, doc) for _, _, doc in results]
        scores = self.cross_encoder.predict(pairs)
        scored = [(results[i][0], float(scores[i]), results[i][2]) for i in range(len(results))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def run_experiment(self, name: str, chunking_fn) -> Dict:
        print(f"\n{'='*60}")
        print(f"Experiment: {name}")
        print(f"{'='*60}")

        # Generate chunks
        all_chunks = []
        for ticket in MOCK_TICKETS:
            all_chunks.extend(chunking_fn(ticket))

        num_chunks = len(all_chunks)
        avg_len = np.mean([len(c["text"]) for c in all_chunks])
        print(f"  Chunks: {num_chunks}, Avg length: {avg_len:.0f} chars")

        # Show chunk breakdown
        chunk_types = {}
        for c in all_chunks:
            ct = c["chunk_type"].split("_")[0] if "_" in c["chunk_type"] else c["chunk_type"]
            chunk_types[ct] = chunk_types.get(ct, 0) + 1
        print(f"  Chunk breakdown: {dict(chunk_types)}")

        # Index
        collection = self._create_collection(name)
        self._index_chunks(collection, all_chunks)

        # Evaluate without reranking
        no_rerank_results = []
        for gt in GROUND_TRUTH_QUERIES:
            retrieved = self._retrieve(collection, gt["query"], RERANK_RETRIEVE_N)
            retrieved_keys = [r[0] for r in retrieved]
            metrics = evaluate_retrieval(retrieved_keys, gt["relevant_keys"], gt["primary_key"], K_VALUES)
            no_rerank_results.append(metrics)
        no_rerank_agg = aggregate_metrics(no_rerank_results)

        # Evaluate with reranking
        rerank_results = []
        for gt in GROUND_TRUTH_QUERIES:
            retrieved = self._retrieve(collection, gt["query"], RERANK_RETRIEVE_N)
            reranked = self._rerank(gt["query"], retrieved, max(K_VALUES))
            retrieved_keys = [r[0] for r in reranked]
            metrics = evaluate_retrieval(retrieved_keys, gt["relevant_keys"], gt["primary_key"], K_VALUES)
            rerank_results.append(metrics)
        rerank_agg = aggregate_metrics(rerank_results)

        result = {
            "name": name,
            "num_chunks": num_chunks,
            "avg_chunk_len": avg_len,
            "no_rerank": no_rerank_agg,
            "with_rerank": rerank_agg,
        }
        self.results[name] = result
        return result

    def run_all(self):
        start_time = time.time()

        for name, fn in ALL_STRATEGIES:
            self.run_experiment(name, fn)

        elapsed = time.time() - start_time
        print(f"\n\nAll experiments completed in {elapsed:.1f}s")

        self._print_comparison()
        self._save_results()

    def _print_comparison(self):
        print("\n" + "=" * 120)
        print("COMMENT CHUNKING EXPERIMENT COMPARISON")
        print("=" * 120)

        for mode, label in [("no_rerank", "Without Reranking"), ("with_rerank", "With Cross-Encoder Reranking")]:
            print(f"\n--- {label} ---")
            headers = ["Strategy", "Chunks", "AvgLen", "Hit@1", "Hit@3", "Hit@5", "MRR", "nDCG@5", "P@5", "R@5"]
            rows = []
            for name in sorted(self.results.keys()):
                res = self.results[name]
                r = res[mode]
                rows.append([
                    name,
                    res["num_chunks"],
                    f"{res['avg_chunk_len']:.0f}",
                    f"{r.get('hit_rate@1', 0):.3f}",
                    f"{r.get('hit_rate@3', 0):.3f}",
                    f"{r.get('hit_rate@5', 0):.3f}",
                    f"{r.get('mrr', 0):.3f}",
                    f"{r.get('ndcg@5', 0):.3f}",
                    f"{r.get('precision@5', 0):.3f}",
                    f"{r.get('recall@5', 0):.3f}",
                ])
            print(tabulate(rows, headers=headers, tablefmt="grid"))

        # Best strategies
        print("\n--- Rankings by nDCG@5 (no rerank) ---")
        ranked = sorted(self.results.items(), key=lambda x: x[1]["no_rerank"]["ndcg@5"], reverse=True)
        for i, (name, res) in enumerate(ranked):
            r = res["no_rerank"]
            print(f"  {i+1}. {name}: nDCG@5={r['ndcg@5']:.4f}  Hit@1={r['hit_rate@1']:.3f}  MRR={r['mrr']:.3f}  R@5={r['recall@5']:.3f}")

        print("\n--- Rankings by nDCG@5 (with rerank) ---")
        ranked = sorted(self.results.items(), key=lambda x: x[1]["with_rerank"]["ndcg@5"], reverse=True)
        for i, (name, res) in enumerate(ranked):
            r = res["with_rerank"]
            print(f"  {i+1}. {name}: nDCG@5={r['ndcg@5']:.4f}  Hit@1={r['hit_rate@1']:.3f}  MRR={r['mrr']:.3f}  R@5={r['recall@5']:.3f}")

    def _save_results(self):
        output_file = "comment_chunking_results.json"
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nFull results saved to {output_file}")


if __name__ == "__main__":
    runner = CommentChunkingRunner()
    runner.run_all()
