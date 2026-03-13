"""
Overlapping Chunks + LLM Expansion — Answer Accuracy Evaluation.

Compares three strategies using LLM-as-judge:
  A) Baseline blob (all ticket content in one chunk)
  B) Overlapping chunks — use top retrieved chunk directly
  C) Overlapping chunks — LLM decides if chunk is complete, expands if not

Scores each answer on Correctness, Completeness, Hallucination (0-5).
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

from mock_jira_tickets import MOCK_TICKETS
from answer_accuracy import QA_PAIRS

load_dotenv()

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50


# ============================================================
# Chunking
# ============================================================
def create_overlapping_chunks(ticket: dict) -> List[Dict]:
    full_text = f"{ticket['summary']}\n\n{ticket['description'].strip()}"
    if ticket.get("comments"):
        full_text += "\n\nComments:\n" + "\n---\n".join(ticket["comments"])

    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    if len(full_text) <= CHUNK_SIZE:
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
        end = min(start + CHUNK_SIZE, len(full_text))
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


def create_blob_chunks(ticket: dict) -> List[Dict]:
    parts = [ticket["summary"], ticket["description"].strip()]
    if ticket.get("comments"):
        parts.append("Comments:\n" + "\n---\n".join(ticket["comments"]))
    return [{"text": "\n\n".join(parts), "ticket_key": ticket["key"]}]


# ============================================================
# Runner
# ============================================================
class OverlapAnswerAccuracyRunner:
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

    def _retrieve_top(self, col, query):
        """Retrieve the single best matching chunk."""
        res = col.query(
            query_embeddings=self._embed([query]),
            n_results=min(1, col.count()),
            include=["documents", "metadatas"]
        )
        return res["documents"][0][0], res["metadatas"][0][0]

    def _retrieve_top_deduped(self, col, query, n=10):
        """Retrieve top chunk per unique ticket."""
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
{chunk_text}
---

Does this chunk contain enough information to fully answer the question? Consider:
1. Is the text cut off mid-sentence or mid-thought?
2. Does it reference something that continues beyond the chunk boundary?
3. Is the core answer present, or is it likely in an adjacent chunk?

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

    def _expand_chunk(self, chunk_doc, chunk_meta, all_chunks_index):
        """Fetch adjacent chunks and merge."""
        ticket_key = chunk_meta["ticket_key"]
        chunk_idx = chunk_meta["chunk_index"]
        total = chunk_meta["total_chunks"]

        merged = chunk_doc
        expanded_from = []

        if chunk_idx > 0:
            prev_key = (ticket_key, chunk_idx - 1)
            if prev_key in all_chunks_index:
                merged = all_chunks_index[prev_key]["text"] + "\n" + merged
                expanded_from.append(f"chunk {chunk_idx} (prev)")

        if chunk_idx < total - 1:
            next_key = (ticket_key, chunk_idx + 1)
            if next_key in all_chunks_index:
                merged = merged + "\n" + all_chunks_index[next_key]["text"]
                expanded_from.append(f"chunk {chunk_idx + 2} (next)")

        return merged, expanded_from

    def _answer(self, question, context):
        resp = self.groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "Answer the question based ONLY on the provided context. Be specific, include technical details, error codes, and solutions mentioned. If the context doesn't contain certain information, say so explicitly."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
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
{context[:1500]}

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
                try:
                    scores["correctness"] = int(line.split(":")[1].strip()[0])
                except (ValueError, IndexError):
                    pass
            elif line.startswith("COMPLETENESS:"):
                try:
                    scores["completeness"] = int(line.split(":")[1].strip()[0])
                except (ValueError, IndexError):
                    pass
            elif line.startswith("HALLUCINATION:"):
                try:
                    scores["hallucination"] = int(line.split(":")[1].strip()[0])
                except (ValueError, IndexError):
                    pass
            elif line.startswith("REASONING:"):
                scores["reasoning"] = line.split(":", 1)[1].strip()

        return scores

    def run(self):
        print(f"\n{'='*110}")
        print("OVERLAP + LLM EXPANSION — ANSWER ACCURACY EVALUATION")
        print(f"Chunk size: {CHUNK_SIZE} chars, Overlap: {CHUNK_OVERLAP} chars")
        print(f"{'='*110}")

        # Build overlap chunks + index
        overlap_chunks = []
        for t in MOCK_TICKETS:
            overlap_chunks.extend(create_overlapping_chunks(t))
        chunk_lookup = {}
        for c in overlap_chunks:
            chunk_lookup[(c["ticket_key"], c.get("chunk_index", 0))] = c

        overlap_col = self._index("overlap_acc", overlap_chunks)
        print(f"Overlap chunks indexed: {len(overlap_chunks)}")

        # Build blob chunks
        blob_chunks = []
        for t in MOCK_TICKETS:
            blob_chunks.extend(create_blob_chunks(t))
        blob_col = self._index("blob_acc_baseline", blob_chunks)
        print(f"Blob chunks indexed: {len(blob_chunks)}")

        strategies = {}
        total_llm_calls = 0
        total_expansions = 0

        # ─── Strategy A: Blob baseline ───
        print(f"\n{'─'*110}")
        print("STRATEGY A: Blob (all content in one chunk)")
        print(f"{'─'*110}")

        blob_scores = []
        for qi, qa in enumerate(QA_PAIRS):
            doc, meta = self._retrieve_top(blob_col, qa["question"])
            retrieved_key = meta["ticket_key"]
            answer = self._answer(qa["question"], doc)
            scores = self._judge(qa["question"], answer, qa["ground_truth"], doc)

            correct_retr = retrieved_key == qa["target_ticket"]
            print(f"\n  Q{qi+1}: {qa['question'][:60]}...")
            print(f"  Retrieved: {retrieved_key} {'✅' if correct_retr else '❌ (expected ' + qa['target_ticket'] + ')'}")
            print(f"  Scores → C:{scores['correctness']} Cm:{scores['completeness']} H:{scores['hallucination']}")
            print(f"  Judge: {scores['reasoning']}")

            blob_scores.append({**scores, "retrieved": retrieved_key,
                                "correct_retrieval": correct_retr, "answer": answer[:200]})

        strategies["blob_all"] = blob_scores

        # ─── Strategy B: Overlap — direct (no LLM check) ───
        print(f"\n{'─'*110}")
        print("STRATEGY B: Overlapping chunks (direct, no LLM expansion)")
        print(f"{'─'*110}")

        overlap_direct_scores = []
        for qi, qa in enumerate(QA_PAIRS):
            top_doc, top_meta = next(self._retrieve_top_deduped(overlap_col, qa["question"]))
            retrieved_key = top_meta["ticket_key"]
            answer = self._answer(qa["question"], top_doc)
            scores = self._judge(qa["question"], answer, qa["ground_truth"], top_doc)

            correct_retr = retrieved_key == qa["target_ticket"]
            print(f"\n  Q{qi+1}: {qa['question'][:60]}...")
            print(f"  Retrieved: {retrieved_key} (chunk {top_meta['chunk_index']+1}/{top_meta['total_chunks']}) {'✅' if correct_retr else '❌'}")
            print(f"  Scores → C:{scores['correctness']} Cm:{scores['completeness']} H:{scores['hallucination']}")
            print(f"  Judge: {scores['reasoning']}")

            overlap_direct_scores.append({**scores, "retrieved": retrieved_key,
                                          "correct_retrieval": correct_retr, "answer": answer[:200]})

        strategies["overlap_direct"] = overlap_direct_scores

        # ─── Strategy C: Overlap + LLM expansion ───
        print(f"\n{'─'*110}")
        print("STRATEGY C: Overlapping chunks + LLM decides if complete → expand")
        print(f"{'─'*110}")

        overlap_expanded_scores = []
        for qi, qa in enumerate(QA_PAIRS):
            top_doc, top_meta = next(self._retrieve_top_deduped(overlap_col, qa["question"]))
            retrieved_key = top_meta["ticket_key"]
            total = top_meta["total_chunks"]
            chunk_idx = top_meta["chunk_index"]

            correct_retr = retrieved_key == qa["target_ticket"]
            print(f"\n  Q{qi+1}: {qa['question'][:60]}...")
            print(f"  Retrieved: {retrieved_key} (chunk {chunk_idx+1}/{total}) {'✅' if correct_retr else '❌'}")

            context = top_doc
            if total > 1:
                total_llm_calls += 1
                is_complete, llm_result = self._llm_check_complete(
                    qa["question"], top_doc, chunk_idx, total
                )
                decision_line = llm_result.split("\n")[0] if llm_result else ""
                reason_line = ""
                for line in llm_result.split("\n"):
                    if line.strip().startswith("REASON:"):
                        reason_line = line.strip()
                        break

                print(f"  🤖 {decision_line}")
                print(f"     {reason_line}")

                if not is_complete:
                    total_expansions += 1
                    merged, expanded_from = self._expand_chunk(top_doc, top_meta, chunk_lookup)
                    context = merged
                    print(f"  🔄 Expanded with: {', '.join(expanded_from)} → {len(merged)} chars")
                else:
                    print(f"  ✅ Using chunk as-is")
            else:
                print(f"  ℹ️  Single chunk — no expansion needed")

            answer = self._answer(qa["question"], context)
            scores = self._judge(qa["question"], answer, qa["ground_truth"], context)

            print(f"  Scores → C:{scores['correctness']} Cm:{scores['completeness']} H:{scores['hallucination']}")
            print(f"  Judge: {scores['reasoning']}")

            overlap_expanded_scores.append({**scores, "retrieved": retrieved_key,
                                            "correct_retrieval": correct_retr, "answer": answer[:200]})

        strategies["overlap_llm_expand"] = overlap_expanded_scores

        # ─── Final Summary ───
        print(f"\n\n{'='*110}")
        print("FINAL COMPARISON")
        print(f"{'='*110}")

        headers = ["Strategy", "Retrieval", "Correctness", "Completeness", "Hallucination", "Avg Score"]
        rows = []
        for name, label in [("blob_all", "A: Blob (baseline)"),
                            ("overlap_direct", "B: Overlap (direct)"),
                            ("overlap_llm_expand", "C: Overlap + LLM expand")]:
            scores = strategies[name]
            retr_acc = sum(1 for s in scores if s["correct_retrieval"]) / len(scores)
            avg_c = np.mean([s["correctness"] for s in scores])
            avg_cm = np.mean([s["completeness"] for s in scores])
            avg_h = np.mean([s["hallucination"] for s in scores])
            avg_total = np.mean([(s["correctness"] + s["completeness"] + s["hallucination"]) / 3 for s in scores])
            rows.append([label, f"{retr_acc:.0%}", f"{avg_c:.1f}/5", f"{avg_cm:.1f}/5",
                         f"{avg_h:.1f}/5", f"{avg_total:.1f}/5"])
        print(tabulate(rows, headers=headers, tablefmt="grid"))

        print(f"\nLLM expansion stats:")
        print(f"  Total LLM completeness calls: {total_llm_calls}")
        print(f"  Chunks expanded: {total_expansions} ({total_expansions/max(total_llm_calls,1)*100:.0f}%)")

        # Per-question breakdown
        print(f"\n{'─'*110}")
        print("Per-Question Breakdown")
        print(f"{'─'*110}")

        q_headers = ["Question", "Target", "Blob C/Cm/H", "Overlap C/Cm/H", "Expanded C/Cm/H"]
        q_rows = []
        for qi, qa in enumerate(QA_PAIRS):
            b = strategies["blob_all"][qi]
            o = strategies["overlap_direct"][qi]
            e = strategies["overlap_llm_expand"][qi]
            br = "✅" if b["correct_retrieval"] else "❌"
            orr = "✅" if o["correct_retrieval"] else "❌"
            er = "✅" if e["correct_retrieval"] else "❌"
            q_rows.append([
                qa["question"][:42] + "...",
                qa["target_ticket"],
                f"{br} {b['correctness']}/{b['completeness']}/{b['hallucination']}",
                f"{orr} {o['correctness']}/{o['completeness']}/{o['hallucination']}",
                f"{er} {e['correctness']}/{e['completeness']}/{e['hallucination']}",
            ])
        print(tabulate(q_rows, headers=q_headers, tablefmt="grid"))

        # Save
        with open("overlap_answer_accuracy_results.json", "w") as f:
            json.dump(strategies, f, indent=2, default=str)
        print("\nResults saved to overlap_answer_accuracy_results.json")


if __name__ == "__main__":
    runner = OverlapAnswerAccuracyRunner()
    runner.run()
