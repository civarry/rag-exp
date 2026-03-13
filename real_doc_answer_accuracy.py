"""
Real Document Answer Accuracy — 142KB CRS Report on Central Valley Project.

Tests blob vs overlapping chunks vs overlap+LLM expansion on a genuine
long-form document (142KB / ~35K tokens) where blob can't possibly fit
in a single embedding.
"""

import os
import json
import time
from typing import Dict, List

import chromadb
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from tabulate import tabulate

load_dotenv()

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"

# Load the document
with open("longdoc.md", "r") as f:
    FULL_DOC = f.read()

# ============================================================
# QA Pairs with ground truth from the document
# ============================================================
QA_PAIRS = [
    {
        "question": "What is Executive Order 14181 and what is Action 5?",
        "ground_truth": "On January 24, 2025, the Trump Administration issued E.O. 14181 directing multiple agencies to ensure adequate water supplies for California and to 'immediately take actions to override existing activities that unduly burden efforts to maximize water deliveries.' It also directs considering actions consistent with the 2020 Trump Administration ROD and expediting ESA exemption processes. In December 2025, Reclamation issued a new ROD (Action 5) modifying the 2024 ROD. Action 5 includes adjustments to Delta export operations, removal of the Delta Summer and Fall Habitat Action (Fall X2), removal of early export-reduction concepts from California's Healthy Rivers and Landscapes Program, and revised governance structure. Expected to increase CVP deliveries by 130,000-180,000 AF and SWP by 120,000-220,000 AF. In March 2026, the Center for Biological Diversity sued Reclamation alleging ESA violations.",
    },
    {
        "question": "What happened with the 2019 Biological Opinions and the associated litigation?",
        "ground_truth": "FWS and NMFS issued BiOps on October 21, 2019, concluding Reclamation's proposed operations would NOT jeopardize threatened or endangered species — a different conclusion from the 2008/2009 BiOps which found jeopardy. Reclamation finalized changes in a 2020 ROD on February 20, 2020. California and environmental organizations sued, alleging ESA, NEPA, and APA violations. Court granted a temporary stay from May 11-31, 2020. After Biden Administration took office, E.O. 13990 required reconsideration of the 2019 BiOps. Reclamation reinitiated consultation September 30, 2021. Court allowed an Interim Operations Plan (IOP) to govern CVP operations for water years 2021-2024 while agencies reconsidered. New BiOps and ROD finalized in late 2024 by the Biden Administration. In August 2025, Trump Administration moved to dismiss the litigation as moot. In December 2025, plaintiffs requested voluntary dismissal without prejudice, which the court granted on December 18, 2025.",
    },
    {
        "question": "What is the Bay-Delta Water Quality Control Plan update and how does it affect water supplies?",
        "ground_truth": "The State Water Board's Bay-Delta Plan was first issued in 1978, with substantive updates in 1991, 1995, and 2006, currently implemented through D-1641 (issued 1999). The Bay-Delta Plan Update is being done in two separate processes: one for the San Joaquin River/Southern Delta and one for the Sacramento River/tributaries north of the Delta. In December 2018, the State Water Board adopted amendments requiring approximately 40% unimpaired flows from the San Joaquin River (range 30-50%). This would reduce water available for human use by 7-23% on average, up to 38% in critically dry years. Sacramento River updates were proposed in July 2025 and updated December 2025. The plan allows voluntary agreements as alternatives to flow-only measures — estimated cost $5.2 billion over 15 years ($740M federal, $2.2B state, $2.3B water users). Reclamation opposed the San Joaquin standards in a July 2018 letter arguing they'd reduce New Melones storage by ~315,000 AF/year.",
    },
    {
        "question": "What are Sacramento River Settlement Contractors and San Joaquin River Exchange Contractors?",
        "ground_truth": "Sacramento River Settlement Contractors are contractors (both individuals and districts) that diverted natural flows from the Sacramento River prior to the CVP's construction and executed settlement agreements with Reclamation for negotiated water rights allocation. They are the largest contract holders of CVP water by percentage of total contracted amounts. San Joaquin River Exchange Contractors are irrigation districts that agreed to 'exchange' exercising their water rights to divert water on the San Joaquin and Kings Rivers for guaranteed CVP water deliveries (typically from the Delta-Mendota Canal and waters north of the Delta). Both are water rights contractors who receive 100% of their contracted amounts in most water-year types. During shortage (critical years based on Shasta inflows), their maximum entitlement may be reduced. Their priority is higher than water service contractors like SOD agricultural users.",
    },
    {
        "question": "What is the Delta Conveyance Project and what is its current status?",
        "ground_truth": "Introduced by Governor Newsom in 2019, the Delta Conveyance Project involves construction of a single tunnel to convey water from two new intakes on the Sacramento River to existing pumps in the Bay-Delta. DWR's reasons: protect water supplies from climate change, sea-level rise, saltwater intrusion, and earthquakes. Estimated cost: $20.1 billion, paid largely by public water agencies and ratepayers. The Delta Conveyance Design and Construction Authority oversees design and construction. State released final environmental impact report in December 2023 and approved the project. In January 2024, lawsuits filed alleging CEQA violations. June 2024: trial court granted preliminary injunction blocking DWR's geotechnical investigations pending Delta Reform Act certification. October 2025: California appeals court reversed, finding CEQA's whole-of-the-action requirement doesn't apply to the Delta Reform Act, and remanded to trial court to reconsider.",
    },
    {
        "question": "What is the San Joaquin River Restoration Program and why was it created?",
        "ground_truth": "Historically the San Joaquin River supported large Chinook salmon populations. After Reclamation completed Friant Dam in the late 1940s, much of the river's water was diverted for agricultural uses and approximately 60 miles of the river became dry in most years. In 1988, environmental groups sued Reclamation. A court ruled Friant Dam operation was violating state law because of its destruction of downstream fisheries. Rather than proceed to trial on a remedy, parties negotiated a settlement in 2006. Implementing legislation was enacted by Congress in 2010 (Title X of P.L. 111-11). SJRRP requires new releases from Friant Dam to restore fisheries (including salmon) over 60 miles to the Merced River confluence. Also requires mitigating water supply losses. Goals achieved through channel/structural modifications and reintroduction of Chinook salmon. Funded by federal appropriations and surcharges from CVP Friant water users. Controversial because restoration flows reduce diversions for irrigation, hydropower, and M&I uses.",
    },
    {
        "question": "How have CVP water allocations varied for south-of-Delta agricultural contractors in recent years?",
        "ground_truth": "SOD agricultural contractors have received full contract allocations only five times since 1990: 1995, 1998, 2006, 2017, and 2023. Recent allocations: 0% in 2015, 5% in 2016, 100% in 2017, 50% in 2018, 75% in 2019, 20% in 2020, 0% in 2021, 0% in 2022, 100% in 2023, 50% in 2024, 55% in 2025, and 15% (est.) in 2026. The largest SOD contractor is Westlands Water District. In 2026, Reclamation designated the water year as noncritical based on Shasta inflows but initially allocated SOD ag contractors only 15%. Since 2024, allocations have also included a drought reserve pool which sets aside water in San Luis Reservoir for future droughts. During the extremely dry water years of 2012-2015, CVP annual deliveries averaged approximately 3.45 million AF.",
    },
    {
        "question": "How does the Coordinated Operations Agreement work between CVP and SWP?",
        "ground_truth": "The CVP and SWP operate together under the 1986 Coordinated Operations Agreement (COA), executed pursuant to P.L. 99-546. COA defines rights and responsibilities of CVP and SWP for in-basin water needs. The original 1986 agreement had a fixed 75% CVP/25% SWP ratio for sharing regulatory requirements for storage withdrawals. In December 2018, an addendum adjusted the sharing ratios by water-year type: Wet and Above Normal 80/20 CVP/SWP, Below Normal 75/25, Dry 65/35, Critically Dry 60/40. The 2018 addendum also changed export capacity sharing from 50/50 to 60/40 CVP/SWP during excess conditions and 65/35 during balanced conditions. The state agreed to transport up to 195,000 AF of CVP water through the SWP's California Aqueduct. Recent disagreements about operational changes, particularly for ESA requirements, have called into question the future of coordinated operations under COA.",
    },
    {
        "question": "What are the WIIN Act's CVP operational provisions and what happened with them?",
        "ground_truth": "Title II of the WIIN Act (P.L. 114-322, enacted 2016) directed pumping to maximize CVP water supplies in accordance with applicable BiOps, allowed increased pumping during storm events generating high flows, authorized water transfer facilitation, and established a new standard for measuring effects on species. Most operational provisions expired at end of 2021 (5 years after enactment). Reclamation reported limited implementation — hydrology during 2017-2018 and disagreements with the state affected implementation. Results included improved agency communication, and in 2018 relaxed inflow-to-export ratios enabled additional exports of 50,000-60,000 AF. Many WIIN Act principles were incorporated into both the Trump 2020 BiOps and the Biden 2024 BiOps (e.g., real-time monitoring). In the 119th Congress, H.R. 6639 (WATER Act) has been proposed to codify parts of the 2025 Trump E.O. Congress also authorized WIIN Act Section 4007 funding for new water storage projects.",
    },
    {
        "question": "What species are affected by CVP operations and what is the status of Delta smelt?",
        "ground_truth": "Several ESA-listed species are affected by CVP and SWP operations. Delta smelt, a small pelagic fish listed as threatened under ESA in 1993, can be trapped and killed (entrained) in CVP/SWP pumps in the Delta. No Delta smelt were found in the annual September midwater trawl survey from 2018 to 2024 — six consecutive years with no smelt found. This raises concerns because low populations could result in greater restrictions on water flowing to users. Multiple anadromous salmonid species are also listed: endangered Sacramento River winter-run Chinook salmon, threatened Central Valley spring-run Chinook salmon, threatened Central Valley steelhead, threatened Southern Oregon/Northern California Coast coho salmon, and threatened Central California Coast steelhead. Winter-run Chinook population estimates were as high as 120,000 in the 1960s, plummeted to fewer than 200 in the 1990s, and have increased in recent years but with significant variation in returning spawners.",
    },
]


# ============================================================
# Chunking
# ============================================================
def create_overlap_chunks(text, chunk_size, chunk_overlap, doc_id="DOC"):
    chunks = []
    step = chunk_size - chunk_overlap
    idx = 0
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        chunks.append({
            "text": chunk_text,
            "doc_id": doc_id,
            "chunk_index": idx,
            "total_chunks": -1,
            "char_start": start,
            "char_end": end,
        })
        idx += 1
        start += step
        if end >= len(text):
            break
    for c in chunks:
        c["total_chunks"] = len(chunks)
    return chunks


# ============================================================
# Runner
# ============================================================
class RealDocRunner:
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

        # Batch add (ChromaDB limit)
        batch = 100
        for i in range(0, len(chunks), batch):
            end = min(i + batch, len(chunks))
            batch_chunks = chunks[i:end]
            col.add(
                ids=[f"chunk_{c['chunk_index']}" for c in batch_chunks],
                documents=[c["text"] for c in batch_chunks],
                embeddings=self._embed([c["text"] for c in batch_chunks]),
                metadatas=[{
                    "chunk_index": c["chunk_index"],
                    "total_chunks": c["total_chunks"],
                    "char_start": c["char_start"],
                    "char_end": c["char_end"],
                } for c in batch_chunks],
            )
        return col

    def _retrieve_top(self, col, query, n=1):
        res = col.query(
            query_embeddings=self._embed([query]),
            n_results=min(n, col.count()),
            include=["documents", "metadatas"]
        )
        return res["documents"][0][0], res["metadatas"][0][0]

    def _llm_check_complete(self, query, chunk_text, chunk_index, total_chunks):
        prompt = f"""You are evaluating whether a retrieved text chunk contains enough information to answer a user's question.

USER QUESTION: "{query}"

RETRIEVED CHUNK (part {chunk_index + 1} of {total_chunks}):
---
{chunk_text[:3000]}
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

    def _expand_chunk(self, chunk_meta, all_chunks, radius=2):
        """Merge adjacent chunks within radius."""
        idx = chunk_meta["chunk_index"]
        total = chunk_meta["total_chunks"]

        start_idx = max(0, idx - radius)
        end_idx = min(total - 1, idx + radius)

        parts = []
        expanded_from = []
        for i in range(start_idx, end_idx + 1):
            parts.append(all_chunks[i]["text"])
            if i != idx:
                expanded_from.append(f"chunk {i+1}")

        return "\n".join(parts), expanded_from

    def _answer(self, question, context):
        resp = self.groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "Answer the question based ONLY on the provided context. Be specific and include all relevant details, numbers, dates, and names mentioned. If the context doesn't contain certain information, say so explicitly."},
                {"role": "user", "content": f"Context:\n{context[:6000]}\n\nQuestion: {question}"},
            ],
            temperature=0.1,
            max_tokens=600,
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
{context[:2500]}

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

    def run(self):
        print(f"\n{'='*110}")
        print("REAL DOCUMENT (142KB CRS Report) — Answer Accuracy Evaluation")
        print(f"Document: Central Valley Project: Issues and Legislation")
        print(f"Size: {len(FULL_DOC):,} chars ({len(FULL_DOC)/1024:.1f} KB)")
        print(f"{'='*110}")

        # Chunk configurations to test
        configs = [
            {"name": "chunk_500", "size": 500, "overlap": 100, "expand_radius": 3},
            {"name": "chunk_1000", "size": 1000, "overlap": 200, "expand_radius": 2},
            {"name": "chunk_2000", "size": 2000, "overlap": 400, "expand_radius": 2},
            {"name": "chunk_4000", "size": 4000, "overlap": 800, "expand_radius": 1},
        ]

        all_results = {}

        for cfg in configs:
            chunk_size = cfg["size"]
            overlap = cfg["overlap"]
            radius = cfg["expand_radius"]
            name = cfg["name"]

            chunks = create_overlap_chunks(FULL_DOC, chunk_size, overlap)
            col = self._index(f"real_{name}", chunks)
            print(f"\n{name}: {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")

            # ─── Direct (no expansion) ───
            direct_scores = []
            print(f"\n{'─'*110}")
            print(f"STRATEGY: {name} (direct)")
            print(f"{'─'*110}")

            for qi, qa in enumerate(QA_PAIRS):
                doc, meta = self._retrieve_top(col, qa["question"])
                answer = self._answer(qa["question"], doc)
                scores = self._judge(qa["question"], answer, qa["ground_truth"], doc)

                print(f"\n  Q{qi+1}: {qa['question'][:65]}...")
                print(f"  Chunk {meta['chunk_index']+1}/{meta['total_chunks']} | Context: {len(doc)} chars")
                print(f"  Scores → C:{scores['correctness']} Cm:{scores['completeness']} H:{scores['hallucination']}")
                print(f"  Judge: {scores['reasoning']}")

                direct_scores.append({**scores, "context_len": len(doc)})

            all_results[f"{name}_direct"] = direct_scores

            # ─── LLM expansion ───
            expand_scores = []
            total_llm_calls = 0
            total_expansions = 0

            print(f"\n{'─'*110}")
            print(f"STRATEGY: {name} + LLM expand (±{radius})")
            print(f"{'─'*110}")

            for qi, qa in enumerate(QA_PAIRS):
                doc, meta = self._retrieve_top(col, qa["question"])
                total_llm_calls += 1
                is_complete, llm_result = self._llm_check_complete(
                    qa["question"], doc, meta["chunk_index"], meta["total_chunks"]
                )

                context = doc
                extra = ""
                if not is_complete:
                    total_expansions += 1
                    merged, expanded_from = self._expand_chunk(meta, chunks, radius=radius)
                    context = merged
                    extra = f" [expanded ±{radius}: {len(merged)} chars]"
                else:
                    extra = " [LLM: complete]"

                answer = self._answer(qa["question"], context)
                scores = self._judge(qa["question"], answer, qa["ground_truth"], context)

                print(f"\n  Q{qi+1}: {qa['question'][:65]}...")
                print(f"  Chunk {meta['chunk_index']+1}/{meta['total_chunks']}{extra}")
                print(f"  Scores → C:{scores['correctness']} Cm:{scores['completeness']} H:{scores['hallucination']}")
                print(f"  Judge: {scores['reasoning']}")

                expand_scores.append({**scores, "context_len": len(context)})

            all_results[f"{name}_expand"] = expand_scores
            print(f"\n  LLM calls: {total_llm_calls}, Expansions: {total_expansions} ({total_expansions/max(total_llm_calls,1)*100:.0f}%)")

        # ─── Final Summary ───
        print(f"\n\n{'='*110}")
        print("FINAL COMPARISON")
        print(f"{'='*110}")

        headers = ["Strategy", "Correctness", "Completeness", "Hallucination", "Avg Score", "Avg Context"]
        rows = []
        for name, scores in all_results.items():
            avg_c = np.mean([s["correctness"] for s in scores])
            avg_cm = np.mean([s["completeness"] for s in scores])
            avg_h = np.mean([s["hallucination"] for s in scores])
            avg_total = np.mean([(s["correctness"] + s["completeness"] + s["hallucination"]) / 3 for s in scores])
            avg_ctx = np.mean([s["context_len"] for s in scores])
            rows.append([name, f"{avg_c:.1f}/5", f"{avg_cm:.1f}/5",
                         f"{avg_h:.1f}/5", f"{avg_total:.1f}/5", f"{avg_ctx:.0f}"])
        print(tabulate(rows, headers=headers, tablefmt="grid"))

        # Per-question breakdown
        print(f"\n{'─'*110}")
        print("Per-Question Breakdown")
        print(f"{'─'*110}")

        strat_names = list(all_results.keys())
        q_headers = ["Question"] + [f"{n}\nC/Cm/H" for n in strat_names]
        q_rows = []
        for qi, qa in enumerate(QA_PAIRS):
            row = [qa["question"][:45] + "..."]
            for name in strat_names:
                s = all_results[name][qi]
                row.append(f"{s['correctness']}/{s['completeness']}/{s['hallucination']}")
            q_rows.append(row)
        print(tabulate(q_rows, headers=q_headers, tablefmt="grid"))

        with open("real_doc_answer_accuracy_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print("\nResults saved to real_doc_answer_accuracy_results.json")


if __name__ == "__main__":
    runner = RealDocRunner()
    runner.run()
