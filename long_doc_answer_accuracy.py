"""
Long Document Answer Accuracy — Blob vs Overlap vs Overlap+LLM Expansion.

Tests whether overlapping chunks + LLM expansion beats blob on longer documents
(5-15KB per ticket instead of 1-3KB).
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

from mock_long_tickets import LONG_TICKETS, LONG_QA_PAIRS

load_dotenv()

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"

# Test multiple chunk sizes to see what works best for long docs
CHUNK_CONFIGS = [
    {"name": "overlap_200", "size": 200, "overlap": 50},
    {"name": "overlap_500", "size": 500, "overlap": 100},
    {"name": "overlap_1000", "size": 1000, "overlap": 200},
]


# ============================================================
# Chunking
# ============================================================
def create_blob_chunks(ticket):
    parts = [ticket["summary"], ticket["description"].strip()]
    if ticket.get("comments"):
        parts.append("Comments:\n" + "\n---\n".join(ticket["comments"]))
    return [{"text": "\n\n".join(parts), "ticket_key": ticket["key"]}]


def create_overlap_chunks(ticket, chunk_size, chunk_overlap):
    full_text = f"{ticket['summary']}\n\n{ticket['description'].strip()}"
    if ticket.get("comments"):
        full_text += "\n\nComments:\n" + "\n---\n".join(ticket["comments"])

    chunks = []
    step = chunk_size - chunk_overlap

    if len(full_text) <= chunk_size:
        chunks.append({
            "text": full_text,
            "ticket_key": ticket["key"],
            "chunk_index": 0,
            "total_chunks": 1,
            "full_text": full_text,
        })
        return chunks

    idx = 0
    start = 0
    while start < len(full_text):
        end = min(start + chunk_size, len(full_text))
        chunk_text = full_text[start:end]
        chunks.append({
            "text": f"[{ticket['key']}] {ticket['summary']}\n\n...{chunk_text}...",
            "ticket_key": ticket["key"],
            "chunk_index": idx,
            "total_chunks": -1,
            "full_text": full_text,
        })
        idx += 1
        start += step
        if end >= len(full_text):
            break

    for c in chunks:
        c["total_chunks"] = len(chunks)
    return chunks


# ============================================================
# Runner
# ============================================================
class LongDocRunner:
    def __init__(self):
        print("Loading models...")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.groq = Groq(api_key=os.getenv("GROQ_API_TOKEN"))
        self.chroma = chromadb.Client()

    def _embed(self, texts):
        return self.embed_model.encode(texts, show_progress_bar=False).tolist()

    def _index(self, name, chunks):
        try:
            self.chroma.delete_collection(name)
        except Exception:
            pass
        col = self.chroma.create_collection(name=name, metadata={"hnsw:space": "cosine"})
        col.add(
            ids=[f"{c['ticket_key']}_{i}" for i, c in enumerate(chunks)],
            documents=[c["text"] for c in chunks],
            embeddings=self._embed([c["text"] for c in chunks]),
            metadatas=[{
                "ticket_key": c["ticket_key"],
                "chunk_index": c.get("chunk_index", 0),
                "total_chunks": c.get("total_chunks", 1),
            } for c in chunks],
        )
        return col

    def _retrieve_top_deduped(self, col, query, n=15):
        res = col.query(
            query_embeddings=self._embed([query]),
            n_results=min(n, col.count()),
            include=["documents", "metadatas"]
        )
        seen = set()
        for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
            key = meta["ticket_key"]
            if key not in seen:
                seen.add(key)
                yield doc, meta

    def _llm_check_complete(self, query, chunk_text, chunk_index, total_chunks):
        prompt = f"""You are evaluating whether a retrieved text chunk contains enough information to answer a user's question.

USER QUESTION: "{query}"

RETRIEVED CHUNK (part {chunk_index + 1} of {total_chunks}):
---
{chunk_text[:2000]}
---

Does this chunk contain enough information to fully answer the question?

Respond with EXACTLY this format:
DECISION: COMPLETE or INCOMPLETE
REASON: <one sentence>"""

        resp = self.groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150,
        )
        time.sleep(0.5)
        result = resp.choices[0].message.content.strip()
        is_complete = "COMPLETE" in result.split("\n")[0] and "INCOMPLETE" not in result.split("\n")[0]
        return is_complete, result

    def _expand_chunk(self, chunk_doc, chunk_meta, all_chunks_index, expand_radius=2):
        """Fetch up to `expand_radius` adjacent chunks in each direction and merge."""
        ticket_key = chunk_meta["ticket_key"]
        chunk_idx = chunk_meta["chunk_index"]
        total = chunk_meta["total_chunks"]

        parts = []
        expanded_from = []

        # Gather previous chunks
        for offset in range(expand_radius, 0, -1):
            prev_idx = chunk_idx - offset
            if prev_idx >= 0:
                prev_key = (ticket_key, prev_idx)
                if prev_key in all_chunks_index:
                    parts.append(all_chunks_index[prev_key]["text"])
                    expanded_from.append(f"chunk {prev_idx+1}")

        # Current chunk
        parts.append(chunk_doc)

        # Gather next chunks
        for offset in range(1, expand_radius + 1):
            next_idx = chunk_idx + offset
            if next_idx < total:
                next_key = (ticket_key, next_idx)
                if next_key in all_chunks_index:
                    parts.append(all_chunks_index[next_key]["text"])
                    expanded_from.append(f"chunk {next_idx+1}")

        return "\n".join(parts), expanded_from

    def _answer(self, question, context):
        resp = self.groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "Answer the question based ONLY on the provided context. Be specific, include technical details, error codes, and solutions mentioned. If the context doesn't contain certain information, say so explicitly."},
                {"role": "user", "content": f"Context:\n{context[:6000]}\n\nQuestion: {question}"},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        time.sleep(0.5)
        return resp.choices[0].message.content.strip()

    def _judge(self, question, answer, ground_truth, context):
        prompt = f"""You are an expert evaluator. Score the ANSWER against the GROUND TRUTH on three criteria.

QUESTION: {question}

GROUND TRUTH (the complete correct answer):
{ground_truth}

ANSWER BEING EVALUATED:
{answer}

CONTEXT PROVIDED TO ANSWERER:
{context[:2000]}

Score each criterion from 0-5:

CORRECTNESS (0-5): Are the facts stated in the answer actually correct?
  0=completely wrong, 3=some facts correct but key errors, 5=all stated facts are correct

COMPLETENESS (0-5): Does the answer cover the key points from the ground truth?
  0=misses everything, 3=covers about half the key points, 5=covers all key points

HALLUCINATION (0-5): Does the answer avoid making up information not in the context?
  0=heavily fabricated, 3=some unsupported claims, 5=everything is grounded in context

Respond with EXACTLY this format (numbers only after the colon):
CORRECTNESS: <0-5>
COMPLETENESS: <0-5>
HALLUCINATION: <0-5>
REASONING: <1-2 sentences explaining the scores>"""

        resp = self.groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
        )
        time.sleep(0.5)
        result = resp.choices[0].message.content.strip()

        scores = {"correctness": 0, "completeness": 0, "hallucination": 0, "reasoning": ""}
        for line in result.split("\n"):
            line = line.strip()
            if line.startswith("CORRECTNESS:"):
                try: scores["correctness"] = int(line.split(":")[1].strip()[0])
                except: pass
            elif line.startswith("COMPLETENESS:"):
                try: scores["completeness"] = int(line.split(":")[1].strip()[0])
                except: pass
            elif line.startswith("HALLUCINATION:"):
                try: scores["hallucination"] = int(line.split(":")[1].strip()[0])
                except: pass
            elif line.startswith("REASONING:"):
                scores["reasoning"] = line.split(":", 1)[1].strip()
        return scores

    def _eval_strategy(self, name, get_context_fn):
        """Run evaluation for a strategy. get_context_fn(question, qa) -> (context, retrieved_key, extra_info)"""
        print(f"\n{'─'*110}")
        print(f"STRATEGY: {name}")
        print(f"{'─'*110}")

        scores_list = []
        for qi, qa in enumerate(LONG_QA_PAIRS):
            context, retrieved_key, extra = get_context_fn(qa["question"], qa)
            answer = self._answer(qa["question"], context)
            scores = self._judge(qa["question"], answer, qa["ground_truth"], context)

            correct_retr = retrieved_key == qa["target_ticket"]
            print(f"\n  Q{qi+1}: {qa['question'][:65]}...")
            print(f"  Retrieved: {retrieved_key} {'✅' if correct_retr else '❌ (expected ' + qa['target_ticket'] + ')'} {extra}")
            print(f"  Context length: {len(context)} chars")
            print(f"  Scores → C:{scores['correctness']} Cm:{scores['completeness']} H:{scores['hallucination']}")
            print(f"  Judge: {scores['reasoning']}")

            scores_list.append({
                **scores, "retrieved": retrieved_key,
                "correct_retrieval": correct_retr,
                "context_len": len(context),
            })
        return scores_list

    def run(self):
        # Print document sizes
        print(f"\n{'='*110}")
        print("LONG DOCUMENT ANSWER ACCURACY — Blob vs Overlap vs Overlap+LLM Expand")
        print(f"{'='*110}")

        print("\nDocument sizes:")
        for t in LONG_TICKETS:
            full = t["summary"] + "\n\n" + t["description"]
            if t.get("comments"):
                full += "\n\n" + "\n---\n".join(t["comments"])
            print(f"  {t['key']}: {len(full):,} chars ({len(full)/1024:.1f} KB)")

        all_results = {}

        # ─── Blob ───
        blob_chunks = []
        for t in LONG_TICKETS:
            blob_chunks.extend(create_blob_chunks(t))
        blob_col = self._index("long_blob", blob_chunks)
        print(f"\nBlob: {len(blob_chunks)} chunks")

        def blob_context(question, qa):
            doc, meta = next(self._retrieve_top_deduped(blob_col, question))
            return doc, meta["ticket_key"], ""

        all_results["blob_all"] = self._eval_strategy("Blob (entire ticket)", blob_context)

        # ─── Overlap configs ───
        for cfg in CHUNK_CONFIGS:
            chunk_size = cfg["size"]
            chunk_overlap = cfg["overlap"]
            cfg_name = cfg["name"]

            overlap_chunks = []
            for t in LONG_TICKETS:
                overlap_chunks.extend(create_overlap_chunks(t, chunk_size, chunk_overlap))
            overlap_col = self._index(f"long_{cfg_name}", overlap_chunks)
            chunk_lookup = {(c["ticket_key"], c.get("chunk_index", 0)): c for c in overlap_chunks}

            print(f"\n{cfg_name}: {len(overlap_chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")

            # Direct (no expansion)
            def make_direct_fn(col):
                def fn(question, qa):
                    doc, meta = next(self._retrieve_top_deduped(col, question))
                    return doc, meta["ticket_key"], f"(chunk {meta['chunk_index']+1}/{meta['total_chunks']})"
                return fn

            all_results[f"{cfg_name}_direct"] = self._eval_strategy(
                f"{cfg_name} (direct)", make_direct_fn(overlap_col))

            # LLM expansion
            def make_expand_fn(col, lookup, cs):
                def fn(question, qa):
                    doc, meta = next(self._retrieve_top_deduped(col, question))
                    tk = meta["ticket_key"]
                    ci = meta["chunk_index"]
                    total = meta["total_chunks"]
                    extra = f"(chunk {ci+1}/{total})"

                    if total <= 1:
                        return doc, tk, extra + " [single]"

                    is_complete, llm_result = self._llm_check_complete(question, doc, ci, total)

                    if is_complete:
                        return doc, tk, extra + " [LLM: complete]"

                    # Expand with radius=2 for larger chunks, radius=3 for smaller
                    radius = 3 if cs <= 200 else 2
                    merged, expanded_from = self._expand_chunk(doc, meta, lookup, expand_radius=radius)
                    return merged, tk, extra + f" [expanded ±{radius}: {len(merged)} chars]"
                return fn

            all_results[f"{cfg_name}_llm_expand"] = self._eval_strategy(
                f"{cfg_name} + LLM expand", make_expand_fn(overlap_col, chunk_lookup, chunk_size))

        # ─── Final Summary ───
        print(f"\n\n{'='*110}")
        print("FINAL COMPARISON")
        print(f"{'='*110}")

        headers = ["Strategy", "Retrieval", "Correctness", "Completeness", "Hallucination", "Avg Score", "Avg Context"]
        rows = []
        for name, scores in all_results.items():
            retr = sum(1 for s in scores if s["correct_retrieval"]) / len(scores)
            avg_c = np.mean([s["correctness"] for s in scores])
            avg_cm = np.mean([s["completeness"] for s in scores])
            avg_h = np.mean([s["hallucination"] for s in scores])
            avg_total = np.mean([(s["correctness"] + s["completeness"] + s["hallucination"]) / 3 for s in scores])
            avg_ctx = np.mean([s["context_len"] for s in scores])
            rows.append([name, f"{retr:.0%}", f"{avg_c:.1f}/5", f"{avg_cm:.1f}/5",
                         f"{avg_h:.1f}/5", f"{avg_total:.1f}/5", f"{avg_ctx:.0f}"])
        print(tabulate(rows, headers=headers, tablefmt="grid"))

        # Per-question
        print(f"\n{'─'*110}")
        print("Per-Question Breakdown")
        print(f"{'─'*110}")

        strat_names = list(all_results.keys())
        q_headers = ["Question", "Target"] + [f"{n}\nC/Cm/H" for n in strat_names]
        q_rows = []
        for qi, qa in enumerate(LONG_QA_PAIRS):
            row = [qa["question"][:40] + "...", qa["target_ticket"]]
            for name in strat_names:
                s = all_results[name][qi]
                r = "✅" if s["correct_retrieval"] else "❌"
                row.append(f"{r} {s['correctness']}/{s['completeness']}/{s['hallucination']}")
            q_rows.append(row)
        print(tabulate(q_rows, headers=q_headers, tablefmt="grid"))

        with open("long_doc_answer_accuracy_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print("\nResults saved to long_doc_answer_accuracy_results.json")


if __name__ == "__main__":
    runner = LongDocRunner()
    runner.run()
