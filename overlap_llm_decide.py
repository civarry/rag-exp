"""
Overlapping Chunk + LLM-Decides-If-Complete Experiment.

Approach:
1. Split each ticket into overlapping chunks (200 chars, 50 char overlap)
2. Retrieve top chunks via bi-encoder
3. For each retrieved chunk, ask the LLM: "Is this chunk complete or do you need the next part?"
4. If LLM says incomplete, fetch the adjacent chunk and merge
5. Compare retrieval quality vs baseline (single blob)

Runs verbosely so you can see the LLM reasoning in real time.
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
from sentence_transformers import SentenceTransformer
from tabulate import tabulate

from mock_jira_tickets import MOCK_TICKETS, GROUND_TRUTH_QUERIES
from metrics import evaluate_retrieval, aggregate_metrics

load_dotenv()

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
K_VALUES = [1, 3, 5]
CHUNK_SIZE = 200  # characters per chunk
CHUNK_OVERLAP = 50  # overlap between chunks


# ============================================================
# Overlapping Chunker
# ============================================================
def create_overlapping_chunks(ticket: dict) -> List[Dict]:
    """Split ticket content into overlapping character-based chunks."""
    # Build full text
    full_text = f"{ticket['summary']}\n\n{ticket['description'].strip()}"
    if ticket.get("comments"):
        full_text += "\n\nComments:\n" + "\n---\n".join(ticket["comments"])

    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    if len(full_text) <= CHUNK_SIZE:
        # Short enough for one chunk
        chunks.append({
            "text": full_text,
            "ticket_key": ticket["key"],
            "chunk_type": "chunk_0",
            "chunk_index": 0,
            "total_chunks": 1,
            "full_text": full_text,
        })
        return chunks

    idx = 0
    start = 0
    while start < len(full_text):
        end = min(start + CHUNK_SIZE, len(full_text))
        chunk_text = full_text[start:end]

        chunks.append({
            "text": f"[{ticket['key']}] {ticket['summary']}\n\n...{chunk_text}...",
            "ticket_key": ticket["key"],
            "chunk_type": f"chunk_{idx}",
            "chunk_index": idx,
            "total_chunks": -1,  # filled after
            "full_text": full_text,
        })
        idx += 1
        start += step
        if end >= len(full_text):
            break

    # Fill total_chunks
    for c in chunks:
        c["total_chunks"] = len(chunks)

    return chunks


def create_baseline_chunks(ticket: dict) -> List[Dict]:
    """Baseline: everything in one chunk."""
    parts = [ticket["summary"], ticket["description"].strip()]
    if ticket.get("comments"):
        parts.append("Comments:\n" + "\n---\n".join(ticket["comments"]))
    text = "\n\n".join(parts)
    return [{"text": text, "ticket_key": ticket["key"], "chunk_type": "blob"}]


# ============================================================
# LLM Completeness Check
# ============================================================
def llm_check_completeness(client: Groq, query: str, chunk_text: str, chunk_index: int,
                            total_chunks: int) -> Tuple[bool, str]:
    """Ask LLM if the chunk has enough info to answer the query, or needs more context."""
    prompt = f"""You are evaluating whether a retrieved text chunk contains enough information to answer a user's search query.

USER QUERY: "{query}"

RETRIEVED CHUNK (part {chunk_index + 1} of {total_chunks}):
---
{chunk_text}
---

Does this chunk contain enough information to answer the query? Consider:
1. Is the text cut off mid-sentence or mid-thought?
2. Does it reference something that seems to continue beyond the chunk boundary?
3. Is the core answer to the query present, or is it likely in an adjacent chunk?

Respond with EXACTLY this format:
DECISION: COMPLETE or INCOMPLETE
REASON: <one sentence explaining why>"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=150,
    )
    result = response.choices[0].message.content.strip()

    is_complete = "COMPLETE" in result.split("\n")[0] and "INCOMPLETE" not in result.split("\n")[0]
    return is_complete, result


# ============================================================
# Experiment Runner
# ============================================================
class OverlapExperimentRunner:
    def __init__(self):
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_TOKEN"))
        self.chroma_client = chromadb.Client()

    def _embed(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_model.encode(texts, show_progress_bar=False).tolist()

    def _create_collection(self, name: str):
        try:
            self.chroma_client.delete_collection(name)
        except Exception:
            pass
        return self.chroma_client.create_collection(name=name, metadata={"hnsw:space": "cosine"})

    def _build_chunk_index(self, all_chunks: List[Dict]) -> Dict:
        """Build an index for looking up adjacent chunks by ticket_key + chunk_index."""
        index = {}
        for chunk in all_chunks:
            key = (chunk["ticket_key"], chunk.get("chunk_index", 0))
            index[key] = chunk
        return index

    def run_verbose_experiment(self, num_queries: int = 10):
        """Run the overlap experiment with verbose LLM decision output."""
        print("\n" + "=" * 80)
        print("OVERLAPPING CHUNKS + LLM COMPLETENESS CHECK")
        print(f"Chunk size: {CHUNK_SIZE} chars, Overlap: {CHUNK_OVERLAP} chars")
        print("=" * 80)

        # Generate overlapping chunks
        all_chunks = []
        for ticket in MOCK_TICKETS:
            all_chunks.extend(create_overlapping_chunks(ticket))

        print(f"\nTotal overlapping chunks: {len(all_chunks)}")
        print(f"Avg chunk length: {np.mean([len(c['text']) for c in all_chunks]):.0f} chars")

        # Index
        collection = self._create_collection("overlap_experiment")
        texts = [c["text"] for c in all_chunks]
        ids = [f"{c['ticket_key']}_{c['chunk_type']}_{i}" for i, c in enumerate(all_chunks)]
        metadatas = [{
            "ticket_key": c["ticket_key"],
            "chunk_type": c["chunk_type"],
            "chunk_index": c["chunk_index"],
            "total_chunks": c["total_chunks"],
        } for c in all_chunks]
        embeddings = self._embed(texts)

        batch_size = 100
        for i in range(0, len(texts), batch_size):
            end = min(i + batch_size, len(texts))
            collection.add(ids=ids[i:end], documents=texts[i:end],
                           embeddings=embeddings[i:end], metadatas=metadatas[i:end])

        chunk_index = self._build_chunk_index(all_chunks)

        # Run queries verbosely
        queries_to_run = GROUND_TRUTH_QUERIES[:num_queries]
        llm_expanded_results = []
        plain_overlap_results = []
        total_llm_calls = 0
        total_expansions = 0

        for qi, gt in enumerate(queries_to_run):
            query = gt["query"]
            print(f"\n{'─'*80}")
            print(f"QUERY {qi+1}: \"{query}\"")
            print(f"Expected: {gt['relevant_keys']} (primary: {gt['primary_key']})")
            print(f"{'─'*80}")

            # Retrieve top chunks (get more since we deduplicate)
            query_embedding = self._embed([query])[0]
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(15, collection.count()),
                include=["documents", "metadatas", "distances"]
            )

            # Process each retrieved chunk
            seen_keys = set()
            final_keys_plain = []
            final_keys_expanded = []

            for rank, (meta, dist, doc) in enumerate(zip(
                results["metadatas"][0], results["distances"][0], results["documents"][0]
            )):
                ticket_key = meta["ticket_key"]
                if ticket_key in seen_keys:
                    continue
                seen_keys.add(ticket_key)

                chunk_idx = meta["chunk_index"]
                total = meta["total_chunks"]

                # Plain result (no LLM check)
                final_keys_plain.append(ticket_key)

                print(f"\n  📄 Rank {len(final_keys_plain)}: {ticket_key} "
                      f"(chunk {chunk_idx+1}/{total}, dist={dist:.4f})")
                print(f"  Text preview: {doc[:120]}...")

                if total <= 1:
                    print(f"  ℹ️  Single chunk ticket — skipping LLM check")
                    final_keys_expanded.append(ticket_key)
                else:
                    # Ask LLM if chunk is complete
                    total_llm_calls += 1
                    is_complete, llm_response = llm_check_completeness(
                        self.groq_client, query, doc, chunk_idx, total
                    )
                    time.sleep(0.3)  # Rate limit

                    print(f"\n  🤖 LLM Assessment:")
                    for line in llm_response.strip().split("\n"):
                        print(f"     {line}")

                    if is_complete:
                        print(f"  ✅ LLM says COMPLETE — using this chunk as-is")
                        final_keys_expanded.append(ticket_key)
                    else:
                        print(f"  🔄 LLM says INCOMPLETE — fetching adjacent chunks...")
                        total_expansions += 1

                        # Fetch previous and next chunks
                        merged_text = doc
                        fetched = []

                        if chunk_idx > 0:
                            prev_key = (ticket_key, chunk_idx - 1)
                            if prev_key in chunk_index:
                                prev_chunk = chunk_index[prev_key]
                                merged_text = prev_chunk["text"] + "\n" + merged_text
                                fetched.append(f"chunk {chunk_idx}/{total} (previous)")

                        if chunk_idx < total - 1:
                            next_key = (ticket_key, chunk_idx + 1)
                            if next_key in chunk_index:
                                next_chunk = chunk_index[next_key]
                                merged_text = merged_text + "\n" + next_chunk["text"]
                                fetched.append(f"chunk {chunk_idx+2}/{total} (next)")

                        if fetched:
                            print(f"  📎 Merged with: {', '.join(fetched)}")
                            print(f"  📏 Expanded length: {len(merged_text)} chars")
                        else:
                            print(f"  ⚠️  No adjacent chunks available")

                        final_keys_expanded.append(ticket_key)

                if len(final_keys_plain) >= 5:
                    break

            # Evaluate
            plain_metrics = evaluate_retrieval(
                final_keys_plain, gt["relevant_keys"], gt["primary_key"], K_VALUES
            )
            expanded_metrics = evaluate_retrieval(
                final_keys_expanded, gt["relevant_keys"], gt["primary_key"], K_VALUES
            )
            plain_overlap_results.append(plain_metrics)
            llm_expanded_results.append(expanded_metrics)

            print(f"\n  📊 Results — Plain: Hit@1={plain_metrics['hit_rate@1']:.0f} "
                  f"MRR={plain_metrics['mrr']:.3f} | "
                  f"Expanded: Hit@1={expanded_metrics['hit_rate@1']:.0f} "
                  f"MRR={expanded_metrics['mrr']:.3f}")

        # Now run baseline blob for comparison on same queries
        print(f"\n\n{'='*80}")
        print("RUNNING BASELINE (single blob) FOR COMPARISON...")
        print(f"{'='*80}")

        baseline_chunks = []
        for ticket in MOCK_TICKETS:
            baseline_chunks.extend(create_baseline_chunks(ticket))

        baseline_collection = self._create_collection("baseline_blob_compare")
        b_texts = [c["text"] for c in baseline_chunks]
        b_ids = [f"{c['ticket_key']}_{c['chunk_type']}_{i}" for i, c in enumerate(baseline_chunks)]
        b_metas = [{"ticket_key": c["ticket_key"], "chunk_type": c["chunk_type"]} for c in baseline_chunks]
        b_embeddings = self._embed(b_texts)
        baseline_collection.add(ids=b_ids, documents=b_texts, embeddings=b_embeddings, metadatas=b_metas)

        baseline_results = []
        for gt in queries_to_run:
            q_emb = self._embed([gt["query"]])[0]
            res = baseline_collection.query(query_embeddings=[q_emb], n_results=5,
                                            include=["metadatas", "distances"])
            retrieved_keys = [m["ticket_key"] for m in res["metadatas"][0]]
            metrics = evaluate_retrieval(retrieved_keys, gt["relevant_keys"], gt["primary_key"], K_VALUES)
            baseline_results.append(metrics)

        # Final comparison
        baseline_agg = aggregate_metrics(baseline_results)
        plain_agg = aggregate_metrics(plain_overlap_results)
        expanded_agg = aggregate_metrics(llm_expanded_results)

        print(f"\n\n{'='*80}")
        print(f"FINAL COMPARISON ({num_queries} queries)")
        print(f"{'='*80}")

        headers = ["Strategy", "Hit@1", "Hit@3", "Hit@5", "MRR", "nDCG@5", "R@5", "LLM Calls", "Expansions"]
        rows = [
            ["Baseline (single blob)",
             f"{baseline_agg['hit_rate@1']:.3f}", f"{baseline_agg['hit_rate@3']:.3f}",
             f"{baseline_agg['hit_rate@5']:.3f}", f"{baseline_agg['mrr']:.3f}",
             f"{baseline_agg['ndcg@5']:.3f}", f"{baseline_agg['recall@5']:.3f}",
             "0", "0"],
            ["Overlap (no LLM check)",
             f"{plain_agg['hit_rate@1']:.3f}", f"{plain_agg['hit_rate@3']:.3f}",
             f"{plain_agg['hit_rate@5']:.3f}", f"{plain_agg['mrr']:.3f}",
             f"{plain_agg['ndcg@5']:.3f}", f"{plain_agg['recall@5']:.3f}",
             "0", "0"],
            ["Overlap + LLM expansion",
             f"{expanded_agg['hit_rate@1']:.3f}", f"{expanded_agg['hit_rate@3']:.3f}",
             f"{expanded_agg['hit_rate@5']:.3f}", f"{expanded_agg['mrr']:.3f}",
             f"{expanded_agg['ndcg@5']:.3f}", f"{expanded_agg['recall@5']:.3f}",
             str(total_llm_calls), str(total_expansions)],
        ]
        print(tabulate(rows, headers=headers, tablefmt="grid"))

        print(f"\n📈 LLM expansion stats:")
        print(f"   Total LLM calls: {total_llm_calls}")
        print(f"   Chunks marked incomplete: {total_expansions} ({total_expansions/max(total_llm_calls,1)*100:.0f}%)")
        print(f"   Avg cost per query: {total_llm_calls/num_queries:.1f} LLM calls")


if __name__ == "__main__":
    runner = OverlapExperimentRunner()
    runner.run_verbose_experiment(num_queries=10)
