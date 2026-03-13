"""
LLM Answer Quality with Chunk Expansion.

Use case: User asks a question → retrieve chunks → LLM answers using chunks.
If a chunk is cut off, the LLM requests the next part before giving its final answer.

Flow:
1. User asks question
2. Retrieve top chunk(s) via bi-encoder
3. LLM reads the chunk and tries to answer
4. If LLM detects the chunk is incomplete, it requests adjacent chunks
5. LLM gives final answer with the expanded context

Compare answer quality between:
A) Single blob (no chunking needed)
B) Overlapping chunks — LLM answers directly (no expansion)
C) Overlapping chunks — LLM can request more chunks if needed
"""

import os
import json
import time
from typing import Dict, List, Tuple

import chromadb
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from tabulate import tabulate

from mock_jira_tickets import MOCK_TICKETS, GROUND_TRUTH_QUERIES

load_dotenv()

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 75

# Sample questions that need detailed answers (not just retrieval)
ANSWER_QUESTIONS = [
    {
        "question": "What caused the Stripe payment errors and how was it fixed?",
        "expected_ticket": "SHOP-101",
        "expected_answer_keywords": ["redis", "cache", "TTL", "key rotation", "24h", "12h", "v2.4.2"],
    },
    {
        "question": "Why are customers not receiving order status emails?",
        "expected_ticket": "SHOP-503",
        "expected_answer_keywords": ["sendgrid", "api key", "rotated", "staging", "production", "403"],
    },
    {
        "question": "How is the search zero results problem being solved?",
        "expected_ticket": "SHOP-201",
        "expected_answer_keywords": ["fuzzy", "edit distance", "misspell", "elasticsearch", "fuzziness"],
    },
    {
        "question": "What's the root cause of the overselling bug and what's the fix?",
        "expected_ticket": "SHOP-601",
        "expected_answer_keywords": ["race condition", "atomic", "UPDATE", "count", "WHERE", "rowcount"],
    },
    {
        "question": "Why did the database connections run out during Black Friday?",
        "expected_ticket": "SHOP-402",
        "expected_answer_keywords": ["connection pool", "pgbouncer", "inventory", "lock", "SKIP LOCKED"],
    },
    {
        "question": "What's wrong with the analytics conversion rate metric?",
        "expected_ticket": "SHOP-901",
        "expected_answer_keywords": ["formula", "swapped", "numerator", "denominator", "sessions"],
    },
    {
        "question": "How are JWT tokens being handled after password changes?",
        "expected_ticket": "SHOP-302",
        "expected_answer_keywords": ["token_generation", "invalidat", "24h", "option A"],
    },
    {
        "question": "What's causing the product page to load slowly on mobile?",
        "expected_ticket": "SHOP-401",
        "expected_answer_keywords": ["image", "WebP", "2.4MB", "LCP", "reviews", "third-party", "SSR"],
    },
]


def create_overlapping_chunks(ticket: dict) -> List[Dict]:
    full_text = f"{ticket['summary']}\n\n{ticket['description'].strip()}"
    if ticket.get("comments"):
        full_text += "\n\nComments:\n" + "\n---\n".join(ticket["comments"])

    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    if len(full_text) <= CHUNK_SIZE:
        chunks.append({
            "text": full_text, "ticket_key": ticket["key"],
            "chunk_index": 0, "total_chunks": 1, "full_text": full_text,
        })
        return chunks

    idx = 0
    start = 0
    while start < len(full_text):
        end = min(start + CHUNK_SIZE, len(full_text))
        chunk_text = full_text[start:end]
        chunks.append({
            "text": chunk_text, "ticket_key": ticket["key"],
            "chunk_index": idx, "total_chunks": -1, "full_text": full_text,
        })
        idx += 1
        start += step
        if end >= len(full_text):
            break

    for c in chunks:
        c["total_chunks"] = len(chunks)
    return chunks


def create_blob_chunks(ticket: dict) -> List[Dict]:
    parts = [ticket["summary"], ticket["description"].strip()]
    if ticket.get("comments"):
        parts.append("Comments:\n" + "\n---\n".join(ticket["comments"]))
    text = "\n\n".join(parts)
    return [{"text": text, "ticket_key": ticket["key"], "chunk_index": 0, "total_chunks": 1, "full_text": text}]


class AnswerExperiment:
    def __init__(self):
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.groq = Groq(api_key=os.getenv("GROQ_API_TOKEN"))
        self.chroma = chromadb.Client()

    def _embed(self, texts):
        return self.embedding_model.encode(texts, show_progress_bar=False).tolist()

    def _index(self, name, chunks):
        try:
            self.chroma.delete_collection(name)
        except Exception:
            pass
        col = self.chroma.create_collection(name=name, metadata={"hnsw:space": "cosine"})
        texts = [c["text"] for c in chunks]
        ids = [f"{c['ticket_key']}_c{c['chunk_index']}_{i}" for i, c in enumerate(chunks)]
        metas = [{"ticket_key": c["ticket_key"], "chunk_index": c["chunk_index"],
                  "total_chunks": c["total_chunks"]} for c in chunks]
        embs = self._embed(texts)
        col.add(ids=ids, documents=texts, embeddings=embs, metadatas=metas)
        return col

    def _retrieve_top(self, col, query, n=3):
        q_emb = self._embed([query])[0]
        res = col.query(query_embeddings=[q_emb], n_results=min(n, col.count()),
                        include=["documents", "metadatas", "distances"])
        return list(zip(res["metadatas"][0], res["documents"][0], res["distances"][0]))

    def _llm_answer(self, question, context, allow_request_more=False):
        if allow_request_more:
            system = """You are a helpful assistant answering questions about Jira tickets.
You will be given retrieved context chunks. Answer the question based ONLY on the provided context.

IMPORTANT: If the context appears to be cut off mid-sentence or you can see it's missing
critical information that would be in the next part, respond with EXACTLY:
NEED_MORE_CONTEXT: <reason why you need more>

Otherwise, provide your answer directly. Be specific and include technical details."""
        else:
            system = """You are a helpful assistant answering questions about Jira tickets.
Answer the question based ONLY on the provided context. Be specific and include technical details.
If the context doesn't contain enough info, say what you can and note what's missing."""

        prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}"

        resp = self.groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        time.sleep(0.5)
        return resp.choices[0].message.content.strip()

    def _score_answer(self, answer: str, keywords: List[str]) -> Tuple[float, List[str], List[str]]:
        answer_lower = answer.lower()
        found = [kw for kw in keywords if kw.lower() in answer_lower]
        missing = [kw for kw in keywords if kw.lower() not in answer_lower]
        score = len(found) / len(keywords) if keywords else 0
        return score, found, missing

    def _build_chunk_lookup(self, chunks):
        lookup = {}
        for c in chunks:
            lookup[(c["ticket_key"], c["chunk_index"])] = c
        return lookup

    def run(self):
        print("\n" + "=" * 100)
        print("LLM ANSWER QUALITY: BLOB vs OVERLAP vs OVERLAP+EXPANSION")
        print("=" * 100)

        # Build indexes
        blob_chunks = []
        overlap_chunks = []
        for t in MOCK_TICKETS:
            blob_chunks.extend(create_blob_chunks(t))
            overlap_chunks.extend(create_overlapping_chunks(t))

        print(f"\nBlob chunks: {len(blob_chunks)} (avg {np.mean([len(c['text']) for c in blob_chunks]):.0f} chars)")
        print(f"Overlap chunks: {len(overlap_chunks)} (avg {np.mean([len(c['text']) for c in overlap_chunks]):.0f} chars)")

        blob_col = self._index("blob_answer", blob_chunks)
        overlap_col = self._index("overlap_answer", overlap_chunks)
        chunk_lookup = self._build_chunk_lookup(overlap_chunks)

        results_a = []  # Blob
        results_b = []  # Overlap direct
        results_c = []  # Overlap + expansion

        for qi, q in enumerate(ANSWER_QUESTIONS):
            question = q["question"]
            keywords = q["expected_answer_keywords"]

            print(f"\n{'━'*100}")
            print(f"Q{qi+1}: {question}")
            print(f"Expected ticket: {q['expected_ticket']}")
            print(f"Expected keywords: {keywords}")
            print(f"{'━'*100}")

            # ─── Strategy A: Blob ───
            blob_results = self._retrieve_top(blob_col, question, n=1)
            blob_context = blob_results[0][1]
            blob_ticket = blob_results[0][0]["ticket_key"]

            answer_a = self._llm_answer(question, blob_context)
            score_a, found_a, missing_a = self._score_answer(answer_a, keywords)

            print(f"\n  ┌─ STRATEGY A: Single Blob (retrieved: {blob_ticket})")
            print(f"  │  Answer: {answer_a[:300]}{'...' if len(answer_a) > 300 else ''}")
            print(f"  │  Keywords found: {len(found_a)}/{len(keywords)} = {score_a:.0%} {found_a}")
            if missing_a:
                print(f"  │  Missing: {missing_a}")
            print(f"  └─")

            # ─── Strategy B: Overlap, direct answer ───
            overlap_results = self._retrieve_top(overlap_col, question, n=1)
            overlap_chunk = overlap_results[0][1]
            overlap_ticket = overlap_results[0][0]["ticket_key"]
            overlap_idx = overlap_results[0][0]["chunk_index"]
            overlap_total = overlap_results[0][0]["total_chunks"]

            answer_b = self._llm_answer(question, overlap_chunk)
            score_b, found_b, missing_b = self._score_answer(answer_b, keywords)

            print(f"\n  ┌─ STRATEGY B: Overlap chunk, answer directly")
            print(f"  │  Retrieved: {overlap_ticket} chunk {overlap_idx+1}/{overlap_total}")
            print(f"  │  Chunk: {overlap_chunk[:150]}...")
            print(f"  │  Answer: {answer_b[:300]}{'...' if len(answer_b) > 300 else ''}")
            print(f"  │  Keywords found: {len(found_b)}/{len(keywords)} = {score_b:.0%} {found_b}")
            if missing_b:
                print(f"  │  Missing: {missing_b}")
            print(f"  └─")

            # ─── Strategy C: Overlap + LLM decides if it needs more ───
            context_c = overlap_chunk
            total_llm_calls = 1
            expansions = 0
            current_idx = overlap_idx
            ticket_key = overlap_ticket
            expansion_log = []

            # Allow up to 3 expansion rounds
            for round_num in range(3):
                answer_attempt = self._llm_answer(question, context_c, allow_request_more=True)
                total_llm_calls += 1

                if answer_attempt.startswith("NEED_MORE_CONTEXT"):
                    reason = answer_attempt.replace("NEED_MORE_CONTEXT:", "").strip()
                    expansion_log.append(f"Round {round_num+1}: LLM needs more → \"{reason}\"")
                    expansions += 1

                    # Fetch next chunk
                    next_idx = current_idx + 1
                    next_key = (ticket_key, next_idx)
                    if next_key in chunk_lookup:
                        next_chunk = chunk_lookup[next_key]
                        context_c = context_c + "\n\n[CONTINUED]\n" + next_chunk["text"]
                        current_idx = next_idx
                        expansion_log.append(f"  → Fetched chunk {next_idx+1}, context now {len(context_c)} chars")
                    else:
                        # Try previous chunk if next doesn't exist
                        prev_idx = overlap_idx - 1
                        prev_key = (ticket_key, prev_idx)
                        if prev_key in chunk_lookup and prev_idx >= 0:
                            prev_chunk = chunk_lookup[prev_key]
                            context_c = prev_chunk["text"] + "\n\n[CONTINUED]\n" + context_c
                            expansion_log.append(f"  → No next chunk, fetched previous chunk {prev_idx+1}")
                        else:
                            expansion_log.append(f"  → No adjacent chunks available, answering with what we have")
                            break
                else:
                    # LLM is satisfied, use this answer
                    break
            else:
                # Max rounds hit, get final answer without expansion option
                answer_attempt = self._llm_answer(question, context_c, allow_request_more=False)
                total_llm_calls += 1

            answer_c = answer_attempt if not answer_attempt.startswith("NEED_MORE_CONTEXT") else \
                self._llm_answer(question, context_c, allow_request_more=False)

            score_c, found_c, missing_c = self._score_answer(answer_c, keywords)

            print(f"\n  ┌─ STRATEGY C: Overlap + LLM expansion")
            print(f"  │  Initial chunk: {overlap_ticket} chunk {overlap_idx+1}/{overlap_total}")
            for log in expansion_log:
                print(f"  │  🔄 {log}")
            if not expansion_log:
                print(f"  │  ✅ LLM answered on first attempt (no expansion needed)")
            print(f"  │  Final context size: {len(context_c)} chars ({expansions} expansions, {total_llm_calls} LLM calls)")
            print(f"  │  Answer: {answer_c[:300]}{'...' if len(answer_c) > 300 else ''}")
            print(f"  │  Keywords found: {len(found_c)}/{len(keywords)} = {score_c:.0%} {found_c}")
            if missing_c:
                print(f"  │  Missing: {missing_c}")
            print(f"  └─")

            results_a.append({"score": score_a, "found": len(found_a), "total": len(keywords), "llm_calls": 1})
            results_b.append({"score": score_b, "found": len(found_b), "total": len(keywords), "llm_calls": 1})
            results_c.append({"score": score_c, "found": len(found_c), "total": len(keywords),
                              "llm_calls": total_llm_calls, "expansions": expansions})

        # ─── Summary ───
        print(f"\n\n{'='*100}")
        print("ANSWER QUALITY SUMMARY")
        print(f"{'='*100}")

        avg_a = np.mean([r["score"] for r in results_a])
        avg_b = np.mean([r["score"] for r in results_b])
        avg_c = np.mean([r["score"] for r in results_c])
        total_calls_c = sum(r["llm_calls"] for r in results_c)
        total_expansions_c = sum(r["expansions"] for r in results_c)

        headers = ["Strategy", "Avg Keyword Score", "Total Keywords Found",
                    "Total LLM Calls", "Expansions"]
        rows = [
            ["A: Single Blob",
             f"{avg_a:.0%}",
             f"{sum(r['found'] for r in results_a)}/{sum(r['total'] for r in results_a)}",
             str(len(ANSWER_QUESTIONS)),
             "0"],
            ["B: Overlap (direct)",
             f"{avg_b:.0%}",
             f"{sum(r['found'] for r in results_b)}/{sum(r['total'] for r in results_b)}",
             str(len(ANSWER_QUESTIONS)),
             "0"],
            ["C: Overlap + Expansion",
             f"{avg_c:.0%}",
             f"{sum(r['found'] for r in results_c)}/{sum(r['total'] for r in results_c)}",
             str(total_calls_c),
             str(total_expansions_c)],
        ]
        print(tabulate(rows, headers=headers, tablefmt="grid"))

        print(f"\n  Improvement B→C (expansion): {avg_c - avg_b:+.0%}")
        print(f"  Gap A vs C (blob vs expanded): {avg_a - avg_c:+.0%}")

        # Per-question breakdown
        print(f"\n{'─'*80}")
        print("Per-Question Keyword Scores:")
        print(f"{'─'*80}")
        headers2 = ["Question", "Blob", "Overlap", "Expanded", "Expansions"]
        rows2 = []
        for qi, q in enumerate(ANSWER_QUESTIONS):
            rows2.append([
                q["question"][:50] + "...",
                f"{results_a[qi]['score']:.0%}",
                f"{results_b[qi]['score']:.0%}",
                f"{results_c[qi]['score']:.0%}",
                str(results_c[qi]["expansions"]),
            ])
        print(tabulate(rows2, headers=headers2, tablefmt="grid"))


if __name__ == "__main__":
    exp = AnswerExperiment()
    exp.run()
