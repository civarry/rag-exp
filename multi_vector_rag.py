"""
Multi-Vector RAG Pipeline for Jira Ticket Resolution Suggestion.

Architecture:
  1. INDEX: For each historical ticket, create two separate embeddings:
     - Problem vector: summary + description (the symptom/bug report)
     - Resolution vector: comments (where the fix discussion lives)
  2. RETRIEVE: Embed incoming ticket (summary + description), search against
     problem vectors to find similar historical tickets.
  3. SYNTHESIZE: Feed the resolution text from top-K matched tickets to an LLM,
     which synthesizes a suggested resolution for the new ticket.

Why this works:
  - New tickets describe problems. Historical tickets also describe problems
    (in summary + description) and solutions (in comments).
  - Matching problem-to-problem uses the same vocabulary (error messages,
    symptoms, affected components), so retrieval accuracy is high.
  - Returning the resolution (not the problem) gives the LLM the actual fix
    details to work with.

Usage:
  pipeline = MultiVectorRAG()
  pipeline.index_tickets(historical_tickets)
  result = pipeline.resolve(incoming_ticket)
  print(result["resolution"])
  print(result["similar_tickets"])
"""

import logging
import os
import re
import time
from typing import Dict, List

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

log = logging.getLogger("multi_vector_rag")

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
TOP_K = 3
LLM_SLEEP = 0.5  # Rate limit buffer for Groq free tier


# ── Text Extraction ──────────────────────────────────────────────────────────

def extract_problem_text(ticket: Dict) -> str:
    """Extract problem description: summary + description.
    This is what gets embedded into the problem vector space."""
    parts = [ticket["summary"]]
    desc = ticket.get("description", "").strip()
    if desc:
        parts.append(desc)
    return "\n\n".join(parts)


def extract_resolution_text(ticket: Dict) -> str:
    """Extract resolution info: comments where fixes are discussed.
    This is what gets embedded into the resolution vector space,
    and what gets returned to the LLM for synthesis."""
    comments = ticket.get("comments", [])
    if comments:
        return "Resolution from comments:\n" + "\n---\n".join(comments)
    # Fallback if no comments exist yet
    return ticket["summary"]


def format_incoming_ticket(ticket: Dict) -> str:
    """Format an incoming ticket for embedding and LLM context."""
    return f"{ticket['summary']}\n\n{ticket.get('description', '')}"


# ── Multi-Vector RAG Pipeline ────────────────────────────────────────────────

class MultiVectorRAG:
    """
    Multi-Vector RAG pipeline that maintains separate problem and resolution
    embedding spaces for accurate Jira ticket similarity search and
    resolution suggestion.

    Ticket schema expected:
        {
            "key": "PROJ-123",
            "summary": "Brief title of the issue",
            "description": "Detailed description of the problem...",
            "comments": ["comment 1", "comment 2", ...]  # optional
        }
    """

    def __init__(self, embedding_model: str = EMBEDDING_MODEL,
                 groq_model: str = GROQ_MODEL,
                 groq_api_key: str = None,
                 top_k: int = TOP_K,
                 chroma_path: str = None):
        """
        Args:
            embedding_model: HuggingFace model name for sentence embeddings.
            groq_model: Groq LLM model for resolution synthesis.
            groq_api_key: Groq API key. Falls back to GROQ_API_TOKEN env var.
            top_k: Number of similar tickets to retrieve.
            chroma_path: Path for persistent ChromaDB storage. None = in-memory.
        """
        log.info("Loading embedding model: %s", embedding_model)
        self.embed_model = SentenceTransformer(embedding_model)
        log.info("Embedding model loaded")

        self.groq = Groq(api_key=groq_api_key or os.getenv("GROQ_API_TOKEN"))
        self.groq_model = groq_model
        self.top_k = top_k
        log.info("LLM: %s | top_k: %d", groq_model, top_k)

        # ChromaDB: in-memory for experiments, persistent for production
        if chroma_path:
            self.chroma = chromadb.PersistentClient(path=chroma_path)
            log.info("ChromaDB persistent client at: %s", chroma_path)
        else:
            self.chroma = chromadb.Client()
            log.info("ChromaDB in-memory client")

        self.problem_col = None
        self.resolution_col = None
        self.ticket_map: Dict[str, Dict] = {}

    # ── Embedding ─────────────────────────────────────────────────────────

    def _embed(self, texts: List[str]) -> List[List[float]]:
        log.debug("Embedding %d texts", len(texts))
        embeddings = self.embed_model.encode(texts, show_progress_bar=False).tolist()
        log.debug("Embedding complete — dim=%d", len(embeddings[0]) if embeddings else 0)
        return embeddings

    def _embed_single(self, text: str) -> List[float]:
        log.debug("Embedding single text (%d chars): %.80s...", len(text), text)
        return self._embed([text])[0]

    # ── Indexing ──────────────────────────────────────────────────────────

    def index_tickets(self, tickets: List[Dict]):
        """
        Index historical tickets into two separate vector collections:
        one for problems, one for resolutions.

        Each ticket needs at minimum: key, summary, description.
        Comments are used for the resolution vector (falls back to summary).

        Call this once with your full historical ticket corpus.
        """
        log.info("=" * 70)
        log.info("INDEXING: %d historical tickets", len(tickets))
        log.info("=" * 70)

        # Store tickets for later lookup
        for t in tickets:
            self.ticket_map[t["key"]] = t

        # Create/recreate collections
        for name in ["mv_problems", "mv_resolutions"]:
            try:
                self.chroma.delete_collection(name)
            except Exception:
                pass

        self.problem_col = self.chroma.create_collection(
            name="mv_problems", metadata={"hnsw:space": "cosine"}
        )
        self.resolution_col = self.chroma.create_collection(
            name="mv_resolutions", metadata={"hnsw:space": "cosine"}
        )
        log.info("Created collections: mv_problems, mv_resolutions (cosine)")

        # Build parallel lists for batch insertion
        prob_ids, prob_docs, prob_metas = [], [], []
        res_ids, res_docs, res_metas = [], [], []

        for ticket in tickets:
            key = ticket["key"]

            # Problem vector: summary + description
            prob_text = extract_problem_text(ticket)
            prob_ids.append(f"{key}_prob")
            prob_docs.append(prob_text)
            prob_metas.append({"ticket_key": key})

            # Resolution vector: comments
            res_text = extract_resolution_text(ticket)
            res_ids.append(f"{key}_res")
            res_docs.append(res_text)
            res_metas.append({"ticket_key": key})

            log.debug(
                "  [%s] problem=%d chars | resolution=%d chars | has_comments=%s",
                key, len(prob_text), len(res_text), bool(ticket.get("comments")),
            )

        # Embed and insert
        log.info("Embedding %d problem vectors...", len(prob_docs))
        prob_embeddings = self._embed(prob_docs)
        self.problem_col.add(
            ids=prob_ids,
            documents=prob_docs,
            embeddings=prob_embeddings,
            metadatas=prob_metas,
        )
        log.info("Problem collection: %d vectors inserted", self.problem_col.count())

        log.info("Embedding %d resolution vectors...", len(res_docs))
        res_embeddings = self._embed(res_docs)
        self.resolution_col.add(
            ids=res_ids,
            documents=res_docs,
            embeddings=res_embeddings,
            metadatas=res_metas,
        )
        log.info("Resolution collection: %d vectors inserted", self.resolution_col.count())

        log.info("Indexing complete. %d tickets -> %d problem + %d resolution vectors",
                 len(tickets), self.problem_col.count(), self.resolution_col.count())

    # ── Retrieval ─────────────────────────────────────────────────────────

    def find_similar(self, incoming_ticket: Dict,
                     top_k: int = None) -> List[Dict]:
        """
        Find the most similar historical tickets by matching the incoming
        ticket's problem against historical problem vectors.

        Returns a list of dicts:
            [
                {
                    "ticket_key": "PROJ-123",
                    "ticket": { ... full ticket dict ... },
                    "problem_text": "...",
                    "resolution_text": "...",
                    "similarity": 0.87,
                },
                ...
            ]
        """
        k = top_k or self.top_k
        query_text = format_incoming_ticket(incoming_ticket)

        log.info("-" * 70)
        log.info("RETRIEVAL: finding top-%d similar tickets", k)
        log.info("-" * 70)
        log.info("Query text (%d chars):\n%s", len(query_text), query_text[:300])

        log.info("Embedding query...")
        query_embedding = self._embed_single(query_text)
        log.info("Query embedding: dim=%d, first 5 values=%s",
                 len(query_embedding), query_embedding[:5])

        # Search against PROBLEM vectors
        count = self.problem_col.count()
        n = min(k, count)
        log.info("Searching problem collection (%d vectors) for top-%d...", count, n)
        results = self.problem_col.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        similar = []
        for rank, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ), 1):
            key = meta["ticket_key"]
            ticket = self.ticket_map.get(key, {})
            similarity = 1 - dist
            res_text = extract_resolution_text(ticket)
            similar.append({
                "ticket_key": key,
                "ticket": ticket,
                "problem_text": doc,
                "resolution_text": res_text,
                "similarity": similarity,
            })
            log.info(
                "  #%d  %s  (similarity=%.4f, distance=%.4f)\n"
                "       Problem:    %.100s...\n"
                "       Resolution: %.100s...",
                rank, key, similarity, dist,
                doc.replace("\n", " "),
                res_text.replace("\n", " "),
            )

        return similar

    # ── LLM Resolution Synthesis ──────────────────────────────────────────

    def _llm_call(self, system: str, user: str,
                  max_tokens: int = 512) -> str:
        log.debug("LLM call — model=%s, max_tokens=%d", self.groq_model, max_tokens)
        log.debug("LLM system prompt (%d chars): %.120s...", len(system), system)
        log.debug("LLM user prompt (%d chars): %.200s...", len(user), user)
        try:
            resp = self.groq.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.1,
                max_tokens=max_tokens,
            )
            time.sleep(LLM_SLEEP)
            content = resp.choices[0].message.content.strip()
            usage = resp.usage
            log.info("LLM response: %d chars | tokens: prompt=%d, completion=%d, total=%d",
                     len(content), usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)
            log.debug("LLM output:\n%s", content)
            return content
        except Exception as e:
            log.error("LLM call failed: %s", e)
            time.sleep(2)
            return f"[LLM call failed: {e}]"

    def synthesize_resolution(self, incoming_ticket: Dict,
                              similar_tickets: List[Dict]) -> str:
        """
        Use the LLM to synthesize a resolution from retrieved similar tickets.
        Feeds the resolution text (comments) from each similar ticket as context.
        """
        log.info("-" * 70)
        log.info("SYNTHESIS: building resolution from %d similar tickets", len(similar_tickets))
        log.info("-" * 70)

        incoming_text = format_incoming_ticket(incoming_ticket)

        context_block = ""
        for match in similar_tickets:
            key = match["ticket_key"]
            res_text = match["resolution_text"]
            # Truncate very long resolution text to keep prompt manageable
            truncated = res_text[:3000] if len(res_text) > 3000 else res_text
            context_block += f"\n--- Historical Ticket {key} ---\n{truncated}\n"
            log.info("  Context from %s: %d chars (truncated=%s)",
                     key, len(res_text), len(res_text) > 3000)

        log.info("Total context block: %d chars", len(context_block))

        return self._llm_call(
            system=(
                "You are a senior support engineer. Based on the similar historical "
                "tickets provided, suggest a resolution for the new incoming ticket. "
                "Be specific: mention root causes, exact fixes, code changes, and "
                "configuration updates from the historical tickets that apply. "
                "If the historical tickets don't contain relevant resolution info, say so."
            ),
            user=(
                f"NEW INCOMING TICKET:\n{incoming_text}\n\n"
                f"SIMILAR HISTORICAL TICKETS:\n{context_block}"
            ),
        )

    # ── Main Interface ────────────────────────────────────────────────────

    def resolve(self, incoming_ticket: Dict,
                top_k: int = None) -> Dict:
        """
        Full pipeline: find similar tickets and synthesize a resolution.

        Args:
            incoming_ticket: Dict with at minimum "summary" and "description".
            top_k: Override number of similar tickets to retrieve.

        Returns:
            {
                "resolution": "Suggested resolution text from LLM...",
                "similar_tickets": [
                    {"ticket_key": "PROJ-123", "similarity": 0.87, ...},
                    ...
                ],
            }
        """
        log.info("=" * 70)
        log.info("RESOLVE: %s", incoming_ticket.get("summary", "")[:80])
        log.info("=" * 70)

        # Step 1: Find similar historical tickets via problem vectors
        similar = self.find_similar(incoming_ticket, top_k=top_k)

        # Step 2: Synthesize resolution from matched tickets' comments
        resolution = self.synthesize_resolution(incoming_ticket, similar)

        log.info("=" * 70)
        log.info("RESULT: retrieved=%s | resolution=%d chars",
                 [s["ticket_key"] for s in similar], len(resolution))
        log.info("=" * 70)

        return {
            "resolution": resolution,
            "similar_tickets": [
                {
                    "ticket_key": s["ticket_key"],
                    "similarity": round(s["similarity"], 4),
                    "resolution_text": s["resolution_text"][:500],
                }
                for s in similar
            ],
        }

    # ── Batch Evaluation ──────────────────────────────────────────────────

    def evaluate(self, test_tickets: List[Dict]) -> Dict:
        """
        Evaluate the pipeline on test tickets with ground truth.

        Each test ticket should have:
            {
                "id": "NEW-001",
                "summary": "...",
                "description": "...",
                "ground_truth_tickets": ["PROJ-123"],  # expected matches
                "ground_truth_resolution": "...",       # correct resolution
            }
        """
        print(f"\nEvaluating on {len(test_tickets)} test tickets...")
        results = []

        for ti, ticket in enumerate(test_tickets):
            print(f"\n{'─'*80}")
            print(f"  [{ti+1}/{len(test_tickets)}] {ticket.get('id', '?')}: "
                  f"{ticket['summary'][:60]}")

            # Run pipeline
            output = self.resolve(ticket)

            # Check retrieval accuracy
            gt_keys = set(ticket.get("ground_truth_tickets", []))
            retrieved_keys = [s["ticket_key"] for s in output["similar_tickets"]]
            hit = bool(gt_keys & set(retrieved_keys))

            # Judge with LLM
            scores = self._judge(
                format_incoming_ticket(ticket),
                output["resolution"],
                ticket.get("ground_truth_resolution", ""),
            )

            print(f"  Retrieved: {retrieved_keys} {'HIT' if hit else 'MISS'}")
            print(f"  Scores: C={scores['correctness']}/5  "
                  f"Cm={scores['completeness']}/5  H={scores['hallucination']}/5")
            print(f"  Judge: {scores['reasoning']}")

            results.append({
                "ticket_id": ticket.get("id", f"test_{ti}"),
                "retrieved_keys": retrieved_keys,
                "retrieval_hit": hit,
                "scores": scores,
                "resolution": output["resolution"],
            })

        # Summary
        hits = sum(1 for r in results if r["retrieval_hit"])
        avg_c = sum(r["scores"]["correctness"] for r in results) / len(results)
        avg_cm = sum(r["scores"]["completeness"] for r in results) / len(results)
        avg_h = sum(r["scores"]["hallucination"] for r in results) / len(results)
        avg_total = (avg_c + avg_cm + avg_h) / 3

        print(f"\n{'='*80}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"  Retrieval Hit Rate: {hits}/{len(results)} ({hits/len(results):.0%})")
        print(f"  Avg Correctness:    {avg_c:.2f}/5")
        print(f"  Avg Completeness:   {avg_cm:.2f}/5")
        print(f"  Avg Hallucination:  {avg_h:.2f}/5")
        print(f"  Avg Overall:        {avg_total:.2f}/5")

        return {
            "results": results,
            "summary": {
                "retrieval_hit_rate": hits / len(results),
                "avg_correctness": avg_c,
                "avg_completeness": avg_cm,
                "avg_hallucination": avg_h,
                "avg_overall": avg_total,
            },
        }

    def _judge(self, incoming_text: str, suggested: str,
               ground_truth: str) -> Dict:
        """LLM-as-judge scores the suggested resolution against ground truth."""
        raw = self._llm_call(
            system="You are a strict evaluator. Follow the scoring format exactly.",
            user=f"""Score the SUGGESTED RESOLUTION against the GROUND TRUTH.

INCOMING TICKET:
{incoming_text[:500]}

GROUND TRUTH RESOLUTION:
{ground_truth}

SUGGESTED RESOLUTION:
{suggested}

Score each from 0-5:
CORRECTNESS: Are the facts correct? (0=wrong, 5=all correct)
COMPLETENESS: Does it cover key fix steps? (0=misses all, 5=covers all)
HALLUCINATION: Does it avoid fabrication? (0=fabricated, 5=fully grounded)

Respond EXACTLY:
CORRECTNESS: <0-5>
COMPLETENESS: <0-5>
HALLUCINATION: <0-5>
REASONING: <1-2 sentences>""",
            max_tokens=300,
        )

        scores = {"correctness": 0, "completeness": 0,
                  "hallucination": 0, "reasoning": ""}
        for line in raw.split("\n"):
            line = line.strip()
            for key in ["CORRECTNESS", "COMPLETENESS", "HALLUCINATION"]:
                if line.upper().startswith(key + ":"):
                    match = re.search(r"\d", line.split(":", 1)[1])
                    if match:
                        scores[key.lower()] = min(int(match.group()), 5)
            if line.upper().startswith("REASONING:"):
                scores["reasoning"] = line.split(":", 1)[1].strip()
        return scores


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from mock_jira_tickets import MOCK_TICKETS
    from mock_long_tickets import LONG_TICKETS

    # Configure logging — INFO shows each pipeline step, DEBUG shows embeddings/prompts
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    # 1. Initialize pipeline
    pipeline = MultiVectorRAG()

    # 2. Index all historical tickets
    historical = list(MOCK_TICKETS) + list(LONG_TICKETS)
    pipeline.index_tickets(historical)

    # 3. Demo: resolve a new incoming ticket
    incoming = {
        "summary": "Payments randomly failing in production — around 10-20% of attempts",
        "description": (
            "We started seeing intermittent payment failures about a week ago. "
            "Roughly 10-20% of checkout attempts return a 500 from the payment gateway. "
            "No code changes on our side recently, but infra team did some credential rotation. "
            "Customers are complaining on social media."
        ),
    }

    print(f"\n{'='*80}")
    print("INCOMING TICKET")
    print(f"{'='*80}")
    print(f"  Summary: {incoming['summary']}")
    print(f"  Description: {incoming['description'][:100]}...")

    result = pipeline.resolve(incoming)

    print(f"\n{'='*80}")
    print("SIMILAR HISTORICAL TICKETS")
    print(f"{'='*80}")
    for s in result["similar_tickets"]:
        print(f"  {s['ticket_key']} (similarity: {s['similarity']})")
        print(f"    {s['resolution_text'][:120]}...")
        print()

    print(f"{'='*80}")
    print("SUGGESTED RESOLUTION")
    print(f"{'='*80}")
    print(result["resolution"])
