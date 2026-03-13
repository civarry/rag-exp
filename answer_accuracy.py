"""
Answer Accuracy Evaluation for RAG on Jira Tickets.

Measures how accurately the LLM answers questions using retrieved chunks.
Uses LLM-as-judge to score answers on:
  - Correctness (0-5): Are the facts in the answer correct?
  - Completeness (0-5): Does the answer cover all key points?
  - Hallucination (0-5): Does the answer fabricate info not in the context? (5 = no hallucination)

Compares chunking strategies for ANSWER quality (not retrieval).
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

load_dotenv()

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"

# ============================================================
# Questions with ground truth answers
# ============================================================
QA_PAIRS = [
    {
        "question": "What caused the Stripe payment errors and how was it fixed?",
        "target_ticket": "SHOP-101",
        "ground_truth": "The Stripe payment gateway was returning HTTP 500 errors for ~15% of transactions. The root cause was that after API key rotation, the old key was being cached in Redis with a 24-hour TTL, but keys were rotated every 12 hours. So ~15% of requests used the stale cached key. The fix was deployed in v2.4.2 — they reduced the Redis TTL and added cache invalidation on key rotation. After the hotfix, Stripe 500 errors dropped to zero.",
    },
    {
        "question": "Why are customers not receiving order status emails?",
        "target_ticket": "SHOP-503",
        "ground_truth": "Customers stopped receiving order status change emails 3 days ago after deploying notification service v3.1. The order status events were being published to RabbitMQ and consumed by the notification service correctly, but SendGrid API calls were failing with 403 Forbidden. The root cause was that the SendGrid API key was rotated but only updated in staging, not production. The fix was updating the production SendGrid key. They also re-sent ~2,400 backlogged emails and planned to move secrets to HashiCorp Vault.",
    },
    {
        "question": "How is the search zero results problem being solved?",
        "target_ticket": "SHOP-201",
        "ground_truth": "12% of searches returned zero results, and 40% of those had obvious misspellings. The Elasticsearch config had no fuzzy matching enabled. The fix involved: 1) enabling fuzziness=AUTO in the match query (edit distance 1 for 3-5 char terms, distance 2 for >5 chars), 2) adding phonetic matching using ES phonetic analysis plugin, 3) adding synonym mapping for common product terms. A 'Did you mean...' suggestion using ES suggest API was planned as a follow-up.",
    },
    {
        "question": "What's the root cause of the overselling bug and what's the fix?",
        "target_ticket": "SHOP-601",
        "ground_truth": "The overselling bug was caused by a race condition in the inventory service. The inventory check and decrement were not atomic — two concurrent requests could both read the same count, both see sufficient stock, and both decrement, resulting in negative inventory. This happened 47 times in one month. The fix was using an atomic UPDATE with a WHERE clause: UPDATE inventory SET count = count - 1 WHERE sku = %s AND count > 0, checking rowcount == 0 to raise OutOfStockError. They also added a database constraint CHECK (count >= 0). For flash sales, Redis-based inventory reservation with TTL expiry was suggested.",
    },
    {
        "question": "Why did the database connections run out during Black Friday?",
        "target_ticket": "SHOP-402",
        "ground_truth": "During Black Friday, traffic spiked to 50K concurrent users. The PostgreSQL connection pool (max 100) was exhausted within 5 minutes. The root cause was long-running inventory check queries averaging 2.3 seconds that held connections while waiting for row-level locks on popular items. Auto-scaling made it worse by adding pods that also couldn't get connections. Fixes included: deploying PgBouncer for transaction-level pooling (handling 1000 connections with only 50 actual DB connections), optimizing inventory queries with SELECT FOR UPDATE SKIP LOCKED (reducing avg from 2.3s to 0.15s), adding Redis-based inventory cache, and implementing circuit breaker pattern.",
    },
    {
        "question": "What's wrong with the analytics conversion rate metric?",
        "target_ticket": "SHOP-901",
        "ground_truth": "The analytics dashboard showed a 12.5% conversion rate, but the actual rate was around 3.2%. The bug was in analytics_service/reports.py line 234 — the formula had the numerator and denominator swapped. It was calculating conversions / unique_sessions_with_purchase (which is ~1 since it divides purchases by purchasers), instead of unique_sessions_with_purchase / total_unique_sessions. Fixed in PR #930 with unit tests added for all metric calculations.",
    },
    {
        "question": "How are JWT tokens being handled after password changes?",
        "target_ticket": "SHOP-302",
        "ground_truth": "JWT tokens were NOT being invalidated when users changed their passwords, which is a security vulnerability. If an account was compromised and the user changed their password, the attacker's JWT remained valid for up to 24 hours (the JWT expiry). The chosen fix was Option A: adding a token_generation column to the users table, incrementing it on password change, including it in the JWT payload, and validating it on each request. This was implemented in migration PR #885. A 'Log out all sessions' button was also planned for the security settings page.",
    },
    {
        "question": "What's causing the product page to load slowly on mobile?",
        "target_ticket": "SHOP-401",
        "ground_truth": "Lighthouse audit showed product detail pages took 4.2s to load on mobile (3G). Core Web Vitals were failing: LCP 4.2s (target <2.5s), FID 180ms (target <100ms), CLS 0.25 (target <0.1). Root causes: 1) Hero product image was 2.4MB unoptimized JPEG (fix: convert to WebP with srcset, saving 70% bandwidth), 2) All product reviews (up to 500) loaded on initial render (fix: virtual scrolling, only render first 10), 3) Third-party scripts (analytics, chat widget) blocked main thread for 800ms (fix: load after onload event using requestIdleCallback), 4) No server-side rendering — entire page was client-rendered React.",
    },
    {
        "question": "What's wrong with the FedEx webhook integration?",
        "target_ticket": "SHOP-501",
        "ground_truth": "FedEx shipping status webhooks were received but the status mapping in the webhook handler was wrong: IT (In Transit) was mapped to Processing instead of Shipped, DL (Delivered) was mapped to Shipped instead of Delivered, and DE (Delivery Exception) was not handled at all (should be Delivery Issue). This caused customers to see stale order statuses, flooding support with 'where is my order' tickets. Additionally, the webhook endpoint had no signature verification, allowing anyone to send fake status updates. Fixed in PR #910 with correct mapping and HMAC signature verification. ~340 orders with incorrect statuses were backfilled.",
    },
    {
        "question": "Why is the CI/CD pipeline so slow and what's the plan to fix it?",
        "target_ticket": "SHOP-1001",
        "ground_truth": "The GitHub Actions CI pipeline takes 45 minutes: dependency install 8 min (no caching), unit tests 12 min (sequential), integration tests 15 min (spinning up Docker each time), build 5 min, staging deploy 5 min. Optimization plan: 1) Cache node_modules and pip packages (-8 min), 2) Parallelize unit tests across 4 workers (-9 min), 3) Use persistent Docker services in CI (-10 min), 4) Use turbo/nx for incremental builds (-5 min). Target: under 15 minutes. Progress so far: caching brought it to 37 min, parallel tests to 28 min, Docker optimization in progress.",
    },
]


# ============================================================
# Chunking strategies
# ============================================================
def chunk_blob(ticket):
    parts = [ticket["summary"], ticket["description"].strip()]
    if ticket.get("comments"):
        parts.append("Comments:\n" + "\n---\n".join(ticket["comments"]))
    return [{"text": "\n\n".join(parts), "ticket_key": ticket["key"]}]


def chunk_summary_desc(ticket):
    return [{"text": f"{ticket['summary']}\n\n{ticket['description'].strip()}", "ticket_key": ticket["key"]}]


def chunk_summary_only(ticket):
    return [{"text": ticket["summary"], "ticket_key": ticket["key"]}]


def chunk_labeled(ticket):
    parts = [
        f"Title: {ticket['summary']}",
        f"Type: {ticket['type']} | Priority: {ticket['priority']}",
        f"Description: {ticket['description'].strip()}",
    ]
    if ticket.get("comments"):
        for i, c in enumerate(ticket["comments"]):
            parts.append(f"Comment {i+1}: {c}")
    return [{"text": "\n\n".join(parts), "ticket_key": ticket["key"]}]


STRATEGIES = [
    ("summary_only", chunk_summary_only),
    ("summary_desc", chunk_summary_desc),
    ("blob_all", chunk_blob),
    ("labeled_fields", chunk_labeled),
]


# ============================================================
# Runner
# ============================================================
class AnswerAccuracyRunner:
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
            metadatas=[{"ticket_key": c["ticket_key"]} for c in chunks],
        )
        return col

    def _retrieve(self, col, query, n=1):
        res = col.query(
            query_embeddings=self._embed([query]),
            n_results=min(n, col.count()),
            include=["documents", "metadatas"]
        )
        return res["documents"][0][0], res["metadatas"][0][0]["ticket_key"]

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

        return scores, result

    def run(self):
        print(f"\n{'='*110}")
        print("ANSWER ACCURACY EVALUATION")
        print(f"{'='*110}")

        all_results = {}

        for strat_name, strat_fn in STRATEGIES:
            print(f"\n{'─'*110}")
            print(f"STRATEGY: {strat_name}")
            print(f"{'─'*110}")

            # Index
            chunks = []
            for t in MOCK_TICKETS:
                chunks.extend(strat_fn(t))
            col = self._index(f"acc_{strat_name}", chunks)

            strat_scores = []

            for qi, qa in enumerate(QA_PAIRS):
                context, retrieved_key = self._retrieve(col, qa["question"])
                answer = self._answer(qa["question"], context)
                scores, raw_judge = self._judge(qa["question"], answer, qa["ground_truth"], context)

                correct_retrieval = retrieved_key == qa["target_ticket"]

                print(f"\n  Q{qi+1}: {qa['question'][:60]}...")
                print(f"  Retrieved: {retrieved_key} {'✅' if correct_retrieval else '❌ (expected ' + qa['target_ticket'] + ')'}")
                print(f"  Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}")
                print(f"  Scores → Correctness: {scores['correctness']}/5  "
                      f"Completeness: {scores['completeness']}/5  "
                      f"Hallucination: {scores['hallucination']}/5")
                print(f"  Judge: {scores['reasoning']}")

                strat_scores.append({
                    "question": qa["question"][:50],
                    "target": qa["target_ticket"],
                    "retrieved": retrieved_key,
                    "correct_retrieval": correct_retrieval,
                    **scores,
                })

            all_results[strat_name] = strat_scores

        # ─── Final Summary ───
        print(f"\n\n{'='*110}")
        print("FINAL SUMMARY")
        print(f"{'='*110}")

        headers = ["Strategy", "Retrieval Acc", "Correctness", "Completeness", "Hallucination", "Avg Score"]
        rows = []
        for strat_name in [s[0] for s in STRATEGIES]:
            scores = all_results[strat_name]
            retrieval_acc = sum(1 for s in scores if s["correct_retrieval"]) / len(scores)
            avg_correct = np.mean([s["correctness"] for s in scores])
            avg_complete = np.mean([s["completeness"] for s in scores])
            avg_halluc = np.mean([s["hallucination"] for s in scores])
            avg_total = np.mean([(s["correctness"] + s["completeness"] + s["hallucination"]) / 3 for s in scores])
            rows.append([
                strat_name,
                f"{retrieval_acc:.0%}",
                f"{avg_correct:.1f}/5",
                f"{avg_complete:.1f}/5",
                f"{avg_halluc:.1f}/5",
                f"{avg_total:.1f}/5",
            ])
        print(tabulate(rows, headers=headers, tablefmt="grid"))

        # Per-question breakdown for best strategy
        print(f"\n{'─'*110}")
        print("Per-Question Breakdown (all strategies)")
        print(f"{'─'*110}")

        q_headers = ["Question", "Target"] + [f"{s[0]}\nC/Cm/H" for s in STRATEGIES]
        q_rows = []
        for qi, qa in enumerate(QA_PAIRS):
            row = [qa["question"][:45] + "...", qa["target_ticket"]]
            for strat_name in [s[0] for s in STRATEGIES]:
                s = all_results[strat_name][qi]
                retr = "✅" if s["correct_retrieval"] else "❌"
                row.append(f"{retr} {s['correctness']}/{s['completeness']}/{s['hallucination']}")
            q_rows.append(row)
        print(tabulate(q_rows, headers=q_headers, tablefmt="grid"))

        # Save
        with open("answer_accuracy_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print("\nResults saved to answer_accuracy_results.json")


if __name__ == "__main__":
    runner = AnswerAccuracyRunner()
    runner.run()
