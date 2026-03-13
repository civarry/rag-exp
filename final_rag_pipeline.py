"""
Final RAG Pipeline Experiment: Head-to-Head Strategy Comparison.

Tests 6 RAG strategies for the use case:
  "New Jira ticket comes in -> find similar historical tickets -> suggest resolution"

Strategies tested:
  1. Adaptive Blob/Chunk — blob for short tickets, overlapping chunks for long ones
  2. HyDE — embed a hypothetical resolution, search against historical blobs
  3. Multi-Vector — separate problem and resolution embeddings
  4. HyDE + Multi-Vector — hypothetical resolution matched against actual resolution vectors
  5. Rerank — blob retrieval top-10, cross-encoder rerank to top-3
  6. Query Rewrite + Adaptive — LLM rewrites ticket into focused query first

Evaluation:
  - LLM-as-judge scores: Correctness (0-5), Completeness (0-5), Hallucination (0-5)
  - Retrieval accuracy: did we find the right historical tickets?
"""

import json
import os
import re
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import CrossEncoder, SentenceTransformer
from tabulate import tabulate

from mock_jira_tickets import MOCK_TICKETS
from mock_long_tickets import LONG_TICKETS

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
BLOB_SIZE_THRESHOLD = 8000  # 8 KB boundary for adaptive strategy
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K_RETRIEVAL = 3
RERANK_INITIAL_K = 10
LLM_SLEEP = 0.5

# ── Incoming test tickets ────────────────────────────────────────────────────
# 10 new tickets that mirror problems from historical tickets but are phrased
# differently, as a real user would file them. Some short/vague, some long.
INCOMING_TICKETS = [
    {
        "id": "NEW-001",
        "summary": "Payments randomly failing in production — around 10-20% of attempts",
        "description": (
            "We started seeing intermittent payment failures about a week ago. "
            "Roughly 10-20% of checkout attempts return a 500 from the payment gateway. "
            "No code changes on our side recently, but infra team did some credential rotation. "
            "Customers are complaining on social media."
        ),
        "ground_truth_tickets": ["SHOP-101"],
        "ground_truth_resolution": (
            "The root cause was a stale API key cached in Redis after credential rotation. "
            "The Redis TTL for the Stripe API key was 24 hours but keys were rotated every 12 hours, "
            "so ~15% of requests used the expired cached key. Fix: reduce Redis TTL to 6 hours "
            "or add cache invalidation on key rotation. Hotfix deployed in v2.4.2."
        ),
    },
    {
        "id": "NEW-002",
        "summary": "Users searching for 'headphones' with typos get zero results",
        "description": (
            "Our search is too strict. If a customer types 'hedphones' or 'headfones' they "
            "get a blank page with no results. This is hurting conversions. We need some kind "
            "of fuzzy matching or 'did you mean' feature."
        ),
        "ground_truth_tickets": ["SHOP-201"],
        "ground_truth_resolution": (
            "Elasticsearch had no fuzzy matching enabled. Fix: enable fuzziness=AUTO in the "
            "match query (edit distance 1 for 3-5 char terms, distance 2 for longer terms), "
            "add phonetic matching via ES phonetic analysis plugin, and add synonym mappings. "
            "A 'Did you mean...' suggestion using the ES suggest API was planned as follow-up."
        ),
    },
    {
        "id": "NEW-003",
        "summary": "Security: old JWT tokens still work after user resets password",
        "description": (
            "Pen test finding: if a user changes their password, their old JWT stays valid "
            "until it naturally expires (24h). This means a compromised token keeps working "
            "even after the user takes corrective action. Need to invalidate tokens on password change."
        ),
        "ground_truth_tickets": ["SHOP-302"],
        "ground_truth_resolution": (
            "Add a token_generation column to the users table. Increment on password change, "
            "include in JWT payload, validate on each request. Implemented in migration PR #885. "
            "Also plan to add a 'Log out all sessions' button in security settings."
        ),
    },
    {
        "id": "NEW-004",
        "summary": "Database connections exhausted under heavy traffic spike",
        "description": (
            "During our last flash sale event we hit 40K+ concurrent users and the PostgreSQL "
            "connection pool was completely consumed within minutes. Every service started timing "
            "out. Auto-scaling added more pods but that made things worse since new pods also "
            "competed for DB connections. We need a strategy to handle connection pool exhaustion "
            "under sudden traffic spikes.\n\n"
            "Environment:\n"
            "- PostgreSQL 14 on RDS\n"
            "- Connection pool max: 100\n"
            "- 8 microservices sharing the pool\n"
            "- Average query time during spike: 2+ seconds (normally <100ms)\n"
        ),
        "ground_truth_tickets": ["SHOP-402"],
        "ground_truth_resolution": (
            "Deploy PgBouncer for transaction-level connection pooling (handles 1000 connections "
            "with only 50 actual DB connections). Optimize slow inventory queries using "
            "SELECT FOR UPDATE SKIP LOCKED (reduced avg from 2.3s to 0.15s). Add Redis-based "
            "inventory cache with write-through invalidation. Implement circuit breaker pattern."
        ),
    },
    {
        "id": "NEW-005",
        "summary": "order status emails stopped going out",
        "description": "customers saying they arent getting emails when order ships. started few days ago. pls fix asap",
        "ground_truth_tickets": ["SHOP-503"],
        "ground_truth_resolution": (
            "SendGrid API key was rotated but only updated in staging, not production. "
            "API calls were failing with 403 Forbidden. Fix: update the production SendGrid key. "
            "Re-send ~2,400 backlogged emails. Plan to move secrets to HashiCorp Vault."
        ),
    },
    {
        "id": "NEW-006",
        "summary": "Race condition: two customers bought the last item simultaneously",
        "description": (
            "We sold the same last-in-stock item to two different customers. Inventory count "
            "went to -1. This has happened multiple times. Looks like inventory check and "
            "decrement is not atomic — concurrent requests both read count=1, both succeed.\n\n"
            "```python\n"
            "count = db.query('SELECT count FROM inventory WHERE sku = %s', sku)\n"
            "if count > 0:\n"
            "    db.execute('UPDATE inventory SET count = count - 1 WHERE sku = %s', sku)\n"
            "```\n"
        ),
        "ground_truth_tickets": ["SHOP-601"],
        "ground_truth_resolution": (
            "Use atomic UPDATE with WHERE clause: "
            "UPDATE inventory SET count = count - 1 WHERE sku = %s AND count > 0. "
            "Check rowcount == 0 to raise OutOfStockError. Also add DB constraint "
            "CHECK (count >= 0). For flash sales, consider Redis-based inventory reservation "
            "with TTL-based expiry."
        ),
    },
    {
        "id": "NEW-007",
        "summary": "API latency doubled after we added request logging middleware",
        "description": (
            "After deploying v3.3 with new audit logging, all API endpoints are ~150ms slower. "
            "p50 went from 45ms to 200ms. The logging middleware does a synchronous database "
            "INSERT on every request, which is blocking the response. We need to make this async "
            "without losing log data."
        ),
        "ground_truth_tickets": ["SHOP-701"],
        "ground_truth_resolution": (
            "Publish audit log events to RabbitMQ instead of synchronous DB INSERT. "
            "A separate consumer writes to the audit_log table asynchronously. "
            "This brought p50 back down to 52ms with only ~1s delay in log visibility."
        ),
    },
    {
        "id": "NEW-008",
        "summary": "Production Postgres failover took 20+ minutes, all APIs down",
        "description": (
            "Our RDS Multi-AZ failover kicked in after disk errors on the primary node, but "
            "the secondary had a stale authentication config. Three microservices couldn't "
            "authenticate to the new primary. PgBouncer cached the failed connections and "
            "wouldn't retry. It took us over 20 minutes to restore service.\n\n"
            "We need:\n"
            "1. Config sync between primary and failover nodes\n"
            "2. PgBouncer to automatically retry failed auth\n"
            "3. Faster SSH access for emergency fixes\n"
            "4. Health checks that verify ALL service accounts, not just one\n\n"
            "Estimated revenue impact: ~$18K for the outage window."
        ),
        "ground_truth_tickets": ["LONG-101"],
        "ground_truth_resolution": (
            "Root cause was pg_hba.conf drift between primary and secondary. Fix: "
            "1) Implement SSM-based pg_hba.conf sync (cron every 5 min pulling from Parameter Store), "
            "2) Set PgBouncer server_login_retry=3 and server_reconnect_timeout=5 so it retries auth, "
            "3) Pre-authorize SSH for on-call engineers, "
            "4) Health checks must verify all service accounts. "
            "After fixes, controlled failover test showed recovery in ~1 minute vs 23 minutes."
        ),
    },
    {
        "id": "NEW-009",
        "summary": "Order processing service OOM-killed every 2 days — memory leak",
        "description": (
            "The order-processing-service pods are being OOM-killed by Kubernetes approximately "
            "every 48 hours. Memory starts at ~256MB and grows at roughly 2MB/hour until hitting "
            "the 512MB limit. Restart fixes it temporarily but customers experience 30-60 second "
            "delays in order confirmation emails during the restart.\n\n"
            "tracemalloc output shows the largest allocation is in order_processor.py line 142, "
            "in the _build_order_context method. The allocation should be GC'd after each order "
            "but it keeps growing.\n\n"
            "Stack:\n"
            "- Python 3.11 / FastAPI / Celery\n"
            "- 3 replicas, 512MB memory limit each\n"
            "- RabbitMQ for queueing"
        ),
        "ground_truth_tickets": ["LONG-201"],
        "ground_truth_resolution": (
            "The leak was a class-level dict _processing_cache on OrderProcessor that was never "
            "cleared — it was intended as per-request cache but being a class variable it persisted "
            "across all requests. Fix: change to instance variable or use a bounded LRU cache. "
            "Also move from class variable to instance variable initialized in __init__, "
            "or use functools.lru_cache with maxsize."
        ),
    },
    {
        "id": "NEW-010",
        "summary": "Elasticsearch bulk indexing silently drops ~3% of product updates",
        "description": (
            "Merchants are reporting that product updates aren't showing up in search. "
            "We traced it to the ES bulk API: our code checks the top-level HTTP 200 status "
            "but never inspects per-document errors in the response body. When ES is under load, "
            "some documents get 429 rejected but we don't notice because the overall request "
            "succeeded.\n\n"
            "About 1,600 out of 50,000 daily updates are lost. Customer support has 40+ open "
            "tickets about missing products. Estimated revenue impact: $45K/month."
        ),
        "ground_truth_tickets": ["LONG-401"],
        "ground_truth_resolution": (
            "Parse individual item errors from bulk response. Retry 429/503 errors with "
            "exponential backoff using Celery's native retry mechanism. Send permanent failures "
            "(400 mapper errors) to a dead letter queue (PostgreSQL table). "
            "Implement size-based batching (50MB max instead of fixed 500 docs) to avoid "
            "413 circuit breaker exceptions. After fix: 0% documents lost, 98.2% retry success rate."
        ),
    },
]


# ── Utility functions ────────────────────────────────────────────────────────

def ticket_full_text(ticket: Dict) -> str:
    """Combine all ticket fields into a single blob."""
    parts = [ticket["summary"]]
    desc = ticket.get("description", "").strip()
    if desc:
        parts.append(desc)
    comments = ticket.get("comments", [])
    if comments:
        parts.append("Comments:\n" + "\n---\n".join(comments))
    return "\n\n".join(parts)


def ticket_problem_text(ticket: Dict) -> str:
    """Extract only the problem description (summary + description)."""
    parts = [ticket["summary"]]
    desc = ticket.get("description", "").strip()
    if desc:
        parts.append(desc)
    return "\n\n".join(parts)


def ticket_resolution_text(ticket: Dict) -> str:
    """Extract resolution info (comments where fixes are discussed)."""
    comments = ticket.get("comments", [])
    if comments:
        return "Resolution from comments:\n" + "\n---\n".join(comments)
    return ticket["summary"]


def overlapping_chunks(text: str, chunk_size: int = CHUNK_SIZE,
                       overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def incoming_ticket_text(ticket: Dict) -> str:
    """Combine incoming ticket summary + description."""
    return f"{ticket['summary']}\n\n{ticket['description']}"


# ── Main Pipeline Class ─────────────────────────────────────────────────────

class RAGPipelineExperiment:
    """Runs all 6 strategies head-to-head and evaluates with LLM-as-judge."""

    STRATEGY_NAMES = [
        "adaptive_blob_chunk",
        "hyde",
        "multi_vector",
        "hyde_multi_vector",
        "rerank",
        "query_rewrite_adaptive",
    ]

    def __init__(self):
        print("=" * 110)
        print("FINAL RAG PIPELINE EXPERIMENT")
        print("Comparing 6 retrieval strategies for Jira ticket resolution suggestion")
        print("=" * 110)

        print("\nLoading embedding model...")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)

        print("Loading cross-encoder reranker...")
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

        self.groq = Groq(api_key=os.getenv("GROQ_API_TOKEN"))
        self.chroma = chromadb.Client()

        # Combine MOCK_TICKETS and LONG_TICKETS as historical knowledge base
        self.historical_tickets = list(MOCK_TICKETS) + list(LONG_TICKETS)
        print(f"Historical knowledge base: {len(self.historical_tickets)} tickets "
              f"({len(MOCK_TICKETS)} short + {len(LONG_TICKETS)} long)")
        print(f"Incoming test tickets: {len(INCOMING_TICKETS)}")

        # Map ticket key -> full ticket for easy lookup
        self.ticket_map = {t["key"]: t for t in self.historical_tickets}

        self.results: Dict[str, List[Dict]] = {}

    # ── Embedding helpers ────────────────────────────────────────────────

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts."""
        return self.embed_model.encode(texts, show_progress_bar=False).tolist()

    def _embed_single(self, text: str) -> List[float]:
        """Embed a single text."""
        return self._embed([text])[0]

    # ── ChromaDB helpers ─────────────────────────────────────────────────

    def _create_collection(self, name: str) -> chromadb.Collection:
        """Create or recreate a ChromaDB collection."""
        try:
            self.chroma.delete_collection(name)
        except Exception:
            pass
        return self.chroma.create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

    def _query_collection(self, collection: chromadb.Collection,
                          query_embedding: List[float],
                          n_results: int = TOP_K_RETRIEVAL
                          ) -> Tuple[List[str], List[Dict], List[float]]:
        """Query a collection and return (documents, metadatas, distances)."""
        count = collection.count()
        if count == 0:
            return [], [], []
        n = min(n_results, count)
        res = collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        return (
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
        )

    # ── LLM helpers ──────────────────────────────────────────────────────

    def _llm_call(self, system: str, user: str, max_tokens: int = 512,
                  temperature: float = 0.1) -> str:
        """Make a Groq LLM call with rate-limit sleep."""
        try:
            resp = self.groq.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            time.sleep(LLM_SLEEP)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"    [LLM ERROR] {e}")
            time.sleep(2)
            return f"[LLM call failed: {e}]"

    def _generate_hypothetical_resolution(self, ticket_text: str) -> str:
        """HyDE: generate a hypothetical resolution for the incoming ticket."""
        return self._llm_call(
            system=(
                "You are a senior engineer. Given a bug report or ticket, write a short "
                "hypothetical resolution describing what the root cause likely is and how "
                "to fix it. Be specific and technical. Write 2-4 sentences."
            ),
            user=f"Ticket:\n{ticket_text}",
            max_tokens=256,
        )

    def _rewrite_query(self, ticket_text: str) -> str:
        """Rewrite the incoming ticket into a focused search query."""
        return self._llm_call(
            system=(
                "You are a search query optimizer. Given a Jira ticket, rewrite it as a "
                "concise, focused search query that would find similar historical tickets. "
                "Focus on the core technical problem, key error messages, and affected components. "
                "Output ONLY the rewritten query, 1-3 sentences max."
            ),
            user=f"Ticket:\n{ticket_text}",
            max_tokens=128,
        )

    def _synthesize_resolution(self, incoming_text: str,
                               retrieved_contexts: List[str],
                               retrieved_keys: List[str]) -> str:
        """LLM synthesizes a resolution from retrieved historical tickets."""
        context_block = ""
        for i, (ctx, key) in enumerate(zip(retrieved_contexts, retrieved_keys)):
            # Truncate very long contexts to keep prompt manageable
            truncated = ctx[:3000] if len(ctx) > 3000 else ctx
            context_block += f"\n--- Historical Ticket {key} ---\n{truncated}\n"

        return self._llm_call(
            system=(
                "You are a senior support engineer. Based on the similar historical tickets "
                "provided, suggest a resolution for the new incoming ticket. Be specific: "
                "mention root causes, exact fixes, code changes, and configuration updates "
                "from the historical tickets that apply. If the historical tickets don't "
                "contain relevant resolution info, say so."
            ),
            user=(
                f"NEW INCOMING TICKET:\n{incoming_text}\n\n"
                f"SIMILAR HISTORICAL TICKETS:\n{context_block}"
            ),
            max_tokens=512,
        )

    def _judge_resolution(self, incoming_text: str, suggested_resolution: str,
                          ground_truth: str) -> Dict[str, Any]:
        """LLM-as-judge scores the suggested resolution."""
        prompt = f"""You are an expert evaluator. Score the SUGGESTED RESOLUTION against the GROUND TRUTH.

INCOMING TICKET:
{incoming_text[:500]}

GROUND TRUTH RESOLUTION (the correct answer):
{ground_truth}

SUGGESTED RESOLUTION BEING EVALUATED:
{suggested_resolution}

Score each criterion from 0-5:

CORRECTNESS (0-5): Are the facts and technical details in the suggested resolution correct?
  0=completely wrong, 3=some facts correct but key errors, 5=all stated facts are correct

COMPLETENESS (0-5): Does the resolution cover the key fix steps from the ground truth?
  0=misses everything, 3=covers about half the key points, 5=covers all key points

HALLUCINATION (0-5): Does the resolution avoid fabricating information not supported by the historical tickets?
  0=heavily fabricated, 3=some unsupported claims, 5=everything is grounded

Respond with EXACTLY this format (numbers only after the colon):
CORRECTNESS: <0-5>
COMPLETENESS: <0-5>
HALLUCINATION: <0-5>
REASONING: <1-2 sentences>"""

        raw = self._llm_call(
            system="You are a strict evaluator. Follow the scoring format exactly.",
            user=prompt,
            max_tokens=300,
        )

        scores = {"correctness": 0, "completeness": 0, "hallucination": 0, "reasoning": ""}
        for line in raw.split("\n"):
            line = line.strip()
            for key in ["CORRECTNESS", "COMPLETENESS", "HALLUCINATION"]:
                if line.upper().startswith(key + ":"):
                    try:
                        val_str = line.split(":", 1)[1].strip()
                        # Extract first digit
                        match = re.search(r"\d", val_str)
                        if match:
                            scores[key.lower()] = min(int(match.group()), 5)
                    except (ValueError, IndexError):
                        pass
            if line.upper().startswith("REASONING:"):
                scores["reasoning"] = line.split(":", 1)[1].strip()

        return scores

    # ── Strategy 1: Adaptive Blob/Chunk ──────────────────────────────────

    def _index_adaptive(self) -> chromadb.Collection:
        """Index: blob if <8KB, overlapping chunks if >=8KB."""
        col = self._create_collection("strat_adaptive")
        ids, docs, embeddings, metas = [], [], [], []

        for ticket in self.historical_tickets:
            full = ticket_full_text(ticket)
            if len(full) < BLOB_SIZE_THRESHOLD:
                chunk_id = f"{ticket['key']}_blob"
                ids.append(chunk_id)
                docs.append(full)
                metas.append({"ticket_key": ticket["key"], "chunk_type": "blob"})
            else:
                chunks = overlapping_chunks(full)
                for ci, chunk in enumerate(chunks):
                    chunk_id = f"{ticket['key']}_chunk_{ci}"
                    ids.append(chunk_id)
                    docs.append(chunk)
                    metas.append({"ticket_key": ticket["key"], "chunk_type": "chunk"})

        embeddings = self._embed(docs)
        col.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
        return col

    def _retrieve_adaptive(self, col: chromadb.Collection,
                           query_text: str) -> Tuple[List[str], List[str]]:
        """Retrieve top-K unique tickets from adaptive index."""
        query_emb = self._embed_single(query_text)
        # Retrieve more than needed to deduplicate by ticket key
        docs, metas, dists = self._query_collection(col, query_emb, n_results=RERANK_INITIAL_K)

        seen_keys = set()
        result_docs, result_keys = [], []
        for doc, meta in zip(docs, metas):
            key = meta["ticket_key"]
            if key not in seen_keys:
                seen_keys.add(key)
                result_docs.append(doc)
                result_keys.append(key)
            if len(result_keys) >= TOP_K_RETRIEVAL:
                break
        return result_docs, result_keys

    def run_strategy_adaptive(self, incoming: Dict) -> Dict:
        """Strategy 1: Adaptive Blob/Chunk."""
        query = incoming_ticket_text(incoming)
        col = self._get_or_build_index("adaptive", self._index_adaptive)
        docs, keys = self._retrieve_adaptive(col, query)
        # For synthesis, get full ticket text for each retrieved key
        contexts = [ticket_full_text(self.ticket_map[k]) for k in keys if k in self.ticket_map]
        resolution = self._synthesize_resolution(query, contexts, keys)
        return {"retrieved_keys": keys, "resolution": resolution}

    # ── Strategy 2: HyDE ────────────────────────────────────────────────

    def _index_hyde_blobs(self) -> chromadb.Collection:
        """Index all historical tickets as blobs for HyDE search."""
        col = self._create_collection("strat_hyde")
        ids, docs, metas = [], [], []
        for ticket in self.historical_tickets:
            full = ticket_full_text(ticket)
            ids.append(ticket["key"])
            docs.append(full)
            metas.append({"ticket_key": ticket["key"]})
        embeddings = self._embed(docs)
        col.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
        return col

    def run_strategy_hyde(self, incoming: Dict) -> Dict:
        """Strategy 2: HyDE."""
        query = incoming_ticket_text(incoming)
        # Generate hypothetical resolution
        hypo = self._generate_hypothetical_resolution(query)
        # Embed the hypothetical resolution and search
        col = self._get_or_build_index("hyde", self._index_hyde_blobs)
        hypo_emb = self._embed_single(hypo)
        docs, metas, dists = self._query_collection(col, hypo_emb, n_results=TOP_K_RETRIEVAL)
        keys = [m["ticket_key"] for m in metas]
        contexts = [ticket_full_text(self.ticket_map[k]) for k in keys if k in self.ticket_map]
        resolution = self._synthesize_resolution(query, contexts, keys)
        return {"retrieved_keys": keys, "resolution": resolution, "hypothetical": hypo}

    # ── Strategy 3: Multi-Vector ─────────────────────────────────────────

    def _index_multi_vector(self) -> Tuple[chromadb.Collection, chromadb.Collection]:
        """Index problem vectors and resolution vectors separately."""
        prob_col = self._create_collection("strat_mv_problem")
        res_col = self._create_collection("strat_mv_resolution")

        # Problem vectors
        prob_ids, prob_docs, prob_metas = [], [], []
        res_ids, res_docs, res_metas = [], [], []

        for ticket in self.historical_tickets:
            prob_text = ticket_problem_text(ticket)
            prob_ids.append(f"{ticket['key']}_prob")
            prob_docs.append(prob_text)
            prob_metas.append({"ticket_key": ticket["key"]})

            res_text = ticket_resolution_text(ticket)
            res_ids.append(f"{ticket['key']}_res")
            res_docs.append(res_text)
            res_metas.append({"ticket_key": ticket["key"]})

        prob_embs = self._embed(prob_docs)
        res_embs = self._embed(res_docs)

        prob_col.add(ids=prob_ids, documents=prob_docs,
                     embeddings=prob_embs, metadatas=prob_metas)
        res_col.add(ids=res_ids, documents=res_docs,
                    embeddings=res_embs, metadatas=res_metas)

        return prob_col, res_col

    def run_strategy_multi_vector(self, incoming: Dict) -> Dict:
        """Strategy 3: Multi-Vector — match against problem, return resolution."""
        query = incoming_ticket_text(incoming)
        prob_col, res_col = self._get_or_build_index(
            "multi_vector", self._index_multi_vector
        )
        # Search against PROBLEM vectors
        query_emb = self._embed_single(query)
        docs, metas, dists = self._query_collection(prob_col, query_emb, n_results=TOP_K_RETRIEVAL)
        keys = [m["ticket_key"] for m in metas]
        # Return RESOLUTION text from matched tickets
        contexts = [ticket_resolution_text(self.ticket_map[k]) for k in keys if k in self.ticket_map]
        resolution = self._synthesize_resolution(query, contexts, keys)
        return {"retrieved_keys": keys, "resolution": resolution}

    # ── Strategy 4: HyDE + Multi-Vector ──────────────────────────────────

    def run_strategy_hyde_multi_vector(self, incoming: Dict) -> Dict:
        """Strategy 4: HyDE + Multi-Vector — hypothetical fix vs actual fix vectors."""
        query = incoming_ticket_text(incoming)
        hypo = self._generate_hypothetical_resolution(query)
        # Search against RESOLUTION vectors with hypothetical resolution embedding
        _, res_col = self._get_or_build_index(
            "multi_vector", self._index_multi_vector
        )
        hypo_emb = self._embed_single(hypo)
        docs, metas, dists = self._query_collection(res_col, hypo_emb, n_results=TOP_K_RETRIEVAL)
        keys = [m["ticket_key"] for m in metas]
        contexts = [ticket_resolution_text(self.ticket_map[k]) for k in keys if k in self.ticket_map]
        resolution = self._synthesize_resolution(query, contexts, keys)
        return {"retrieved_keys": keys, "resolution": resolution, "hypothetical": hypo}

    # ── Strategy 5: Rerank ───────────────────────────────────────────────

    def _index_rerank_blobs(self) -> chromadb.Collection:
        """Index all historical tickets as blobs for reranking."""
        col = self._create_collection("strat_rerank")
        ids, docs, metas = [], [], []
        for ticket in self.historical_tickets:
            full = ticket_full_text(ticket)
            ids.append(ticket["key"])
            docs.append(full)
            metas.append({"ticket_key": ticket["key"]})
        embeddings = self._embed(docs)
        col.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
        return col

    def run_strategy_rerank(self, incoming: Dict) -> Dict:
        """Strategy 5: Retrieve top-10 blobs, cross-encoder rerank to top-3."""
        query = incoming_ticket_text(incoming)
        col = self._get_or_build_index("rerank", self._index_rerank_blobs)
        query_emb = self._embed_single(query)
        docs, metas, dists = self._query_collection(col, query_emb, n_results=RERANK_INITIAL_K)

        if not docs:
            return {"retrieved_keys": [], "resolution": "[No results]"}

        # Cross-encoder rerank
        pairs = [(query, doc[:2000]) for doc in docs]  # Truncate for cross-encoder
        ce_scores = self.cross_encoder.predict(pairs)

        # Sort by cross-encoder score descending
        ranked = sorted(
            zip(docs, metas, ce_scores.tolist()),
            key=lambda x: x[2],
            reverse=True,
        )

        top_docs = [r[0] for r in ranked[:TOP_K_RETRIEVAL]]
        top_keys = [r[1]["ticket_key"] for r in ranked[:TOP_K_RETRIEVAL]]

        contexts = [ticket_full_text(self.ticket_map[k]) for k in top_keys if k in self.ticket_map]
        resolution = self._synthesize_resolution(query, contexts, top_keys)
        return {"retrieved_keys": top_keys, "resolution": resolution}

    # ── Strategy 6: Query Rewrite + Adaptive ─────────────────────────────

    def run_strategy_query_rewrite(self, incoming: Dict) -> Dict:
        """Strategy 6: LLM rewrites query, then adaptive retrieval."""
        original_query = incoming_ticket_text(incoming)
        rewritten = self._rewrite_query(original_query)

        col = self._get_or_build_index("adaptive", self._index_adaptive)
        docs, keys = self._retrieve_adaptive(col, rewritten)

        contexts = [ticket_full_text(self.ticket_map[k]) for k in keys if k in self.ticket_map]
        resolution = self._synthesize_resolution(original_query, contexts, keys)
        return {"retrieved_keys": keys, "resolution": resolution, "rewritten_query": rewritten}

    # ── Index caching ────────────────────────────────────────────────────

    _index_cache: Dict[str, Any] = {}

    def _get_or_build_index(self, name: str, builder):
        """Cache built indices so we don't rebuild for every incoming ticket."""
        if name not in self._index_cache:
            print(f"  Building index: {name}...")
            self._index_cache[name] = builder()
        return self._index_cache[name]

    # ── Run all strategies on one incoming ticket ────────────────────────

    def _run_one_ticket(self, incoming: Dict) -> Dict[str, Dict]:
        """Run all 6 strategies on a single incoming ticket."""
        ticket_results = {}
        strategy_runners = {
            "adaptive_blob_chunk": self.run_strategy_adaptive,
            "hyde": self.run_strategy_hyde,
            "multi_vector": self.run_strategy_multi_vector,
            "hyde_multi_vector": self.run_strategy_hyde_multi_vector,
            "rerank": self.run_strategy_rerank,
            "query_rewrite_adaptive": self.run_strategy_query_rewrite,
        }

        for strat_name, runner in strategy_runners.items():
            try:
                result = runner(incoming)
                # Judge the resolution
                scores = self._judge_resolution(
                    incoming_ticket_text(incoming),
                    result["resolution"],
                    incoming["ground_truth_resolution"],
                )
                # Check retrieval accuracy
                gt_keys = set(incoming["ground_truth_tickets"])
                retrieved_keys = result.get("retrieved_keys", [])
                hit = bool(gt_keys & set(retrieved_keys))

                ticket_results[strat_name] = {
                    "retrieved_keys": retrieved_keys,
                    "resolution": result["resolution"],
                    "scores": scores,
                    "retrieval_hit": hit,
                    **{k: v for k, v in result.items()
                       if k not in ("retrieved_keys", "resolution")},
                }
            except Exception as e:
                print(f"    [ERROR in {strat_name}] {e}")
                traceback.print_exc()
                ticket_results[strat_name] = {
                    "retrieved_keys": [],
                    "resolution": f"[Error: {e}]",
                    "scores": {"correctness": 0, "completeness": 0,
                               "hallucination": 0, "reasoning": f"Error: {e}"},
                    "retrieval_hit": False,
                }

        return ticket_results

    # ── Main run ─────────────────────────────────────────────────────────

    def run(self):
        """Execute the full experiment."""
        all_results = {}

        print(f"\n{'=' * 110}")
        print("PHASE 1: INDEXING HISTORICAL TICKETS")
        print(f"{'=' * 110}")

        # Pre-build all indices
        self._get_or_build_index("adaptive", self._index_adaptive)
        self._get_or_build_index("hyde", self._index_hyde_blobs)
        self._get_or_build_index("multi_vector", self._index_multi_vector)
        self._get_or_build_index("rerank", self._index_rerank_blobs)
        print("  All indices built.\n")

        print(f"{'=' * 110}")
        print("PHASE 2: PROCESSING INCOMING TICKETS THROUGH ALL STRATEGIES")
        print(f"{'=' * 110}")

        for ti, incoming in enumerate(INCOMING_TICKETS):
            print(f"\n{'─' * 110}")
            print(f"INCOMING TICKET {ti + 1}/{len(INCOMING_TICKETS)}: "
                  f"{incoming['id']} — {incoming['summary']}")
            print(f"Ground truth: {incoming['ground_truth_tickets']}")
            print(f"{'─' * 110}")

            ticket_results = self._run_one_ticket(incoming)
            all_results[incoming["id"]] = ticket_results

            # Print per-strategy results for this ticket
            for strat_name in self.STRATEGY_NAMES:
                r = ticket_results[strat_name]
                hit_marker = "HIT" if r["retrieval_hit"] else "MISS"
                s = r["scores"]
                print(f"\n  [{strat_name}] Retrieved: {r['retrieved_keys']} [{hit_marker}]")
                print(f"    Scores: C={s['correctness']}/5  Cm={s['completeness']}/5  "
                      f"H={s['hallucination']}/5")
                print(f"    Resolution: {r['resolution'][:200]}...")
                if s.get("reasoning"):
                    print(f"    Judge: {s['reasoning']}")

        # ── Final comparison ─────────────────────────────────────────────
        self._print_final_comparison(all_results)
        self._save_results(all_results)

        return all_results

    def _print_final_comparison(self, all_results: Dict):
        """Print the final comparison tables."""
        print(f"\n\n{'=' * 110}")
        print("FINAL COMPARISON: ALL STRATEGIES")
        print(f"{'=' * 110}\n")

        # Aggregate scores per strategy
        headers = [
            "Strategy", "Retrieval Hit%",
            "Correctness", "Completeness", "Hallucination", "Avg Score",
        ]
        rows = []

        for strat in self.STRATEGY_NAMES:
            hits, correct, complete, halluc, total = [], [], [], [], []
            for tid, ticket_results in all_results.items():
                r = ticket_results[strat]
                hits.append(1 if r["retrieval_hit"] else 0)
                correct.append(r["scores"]["correctness"])
                complete.append(r["scores"]["completeness"])
                halluc.append(r["scores"]["hallucination"])
                avg = (r["scores"]["correctness"] + r["scores"]["completeness"]
                       + r["scores"]["hallucination"]) / 3
                total.append(avg)

            rows.append([
                strat,
                f"{np.mean(hits):.0%}",
                f"{np.mean(correct):.2f}/5",
                f"{np.mean(complete):.2f}/5",
                f"{np.mean(halluc):.2f}/5",
                f"{np.mean(total):.2f}/5",
            ])

        print(tabulate(rows, headers=headers, tablefmt="grid"))

        # ── Per-ticket breakdown ─────────────────────────────────────────
        print(f"\n{'─' * 110}")
        print("PER-TICKET BREAKDOWN")
        print(f"{'─' * 110}\n")

        q_headers = ["Ticket", "Ground Truth"] + [
            f"{s}\nHit|C|Cm|H" for s in self.STRATEGY_NAMES
        ]
        q_rows = []

        for incoming in INCOMING_TICKETS:
            tid = incoming["id"]
            row = [
                f"{tid}: {incoming['summary'][:35]}...",
                ", ".join(incoming["ground_truth_tickets"]),
            ]
            for strat in self.STRATEGY_NAMES:
                r = all_results[tid][strat]
                hit = "Y" if r["retrieval_hit"] else "N"
                s = r["scores"]
                row.append(f"{hit}|{s['correctness']}|{s['completeness']}|{s['hallucination']}")
            q_rows.append(row)

        print(tabulate(q_rows, headers=q_headers, tablefmt="grid"))

        # ── Strategy ranking ─────────────────────────────────────────────
        print(f"\n{'─' * 110}")
        print("STRATEGY RANKING (by average score)")
        print(f"{'─' * 110}\n")

        strategy_avgs = []
        for strat in self.STRATEGY_NAMES:
            scores = []
            for tid, ticket_results in all_results.items():
                r = ticket_results[strat]
                avg = (r["scores"]["correctness"] + r["scores"]["completeness"]
                       + r["scores"]["hallucination"]) / 3
                scores.append(avg)
            strategy_avgs.append((strat, np.mean(scores)))

        strategy_avgs.sort(key=lambda x: x[1], reverse=True)
        for rank, (strat, avg) in enumerate(strategy_avgs, 1):
            print(f"  {rank}. {strat:30s}  avg={avg:.2f}/5")

    def _save_results(self, all_results: Dict):
        """Save results to JSON."""
        # Convert to serializable format
        output = {
            "experiment": "final_rag_pipeline",
            "embedding_model": EMBEDDING_MODEL,
            "cross_encoder_model": CROSS_ENCODER_MODEL,
            "llm_model": GROQ_MODEL,
            "blob_size_threshold": BLOB_SIZE_THRESHOLD,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "num_historical_tickets": len(self.historical_tickets),
            "num_incoming_tickets": len(INCOMING_TICKETS),
            "strategies": self.STRATEGY_NAMES,
            "results": {},
            "summary": {},
        }

        for tid, ticket_results in all_results.items():
            output["results"][tid] = {}
            for strat, r in ticket_results.items():
                output["results"][tid][strat] = {
                    "retrieved_keys": r["retrieved_keys"],
                    "resolution": r["resolution"],
                    "scores": r["scores"],
                    "retrieval_hit": r["retrieval_hit"],
                }

        # Summary per strategy
        for strat in self.STRATEGY_NAMES:
            hits, correct, complete, halluc = [], [], [], []
            for tid, ticket_results in all_results.items():
                r = ticket_results[strat]
                hits.append(1 if r["retrieval_hit"] else 0)
                correct.append(r["scores"]["correctness"])
                complete.append(r["scores"]["completeness"])
                halluc.append(r["scores"]["hallucination"])

            output["summary"][strat] = {
                "retrieval_hit_rate": float(np.mean(hits)),
                "avg_correctness": float(np.mean(correct)),
                "avg_completeness": float(np.mean(complete)),
                "avg_hallucination": float(np.mean(halluc)),
                "avg_overall": float(np.mean([
                    (c + cm + h) / 3
                    for c, cm, h in zip(correct, complete, halluc)
                ])),
            }

        out_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "final_rag_pipeline_results.json",
        )
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {out_path}")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    experiment = RAGPipelineExperiment()
    experiment.run()
