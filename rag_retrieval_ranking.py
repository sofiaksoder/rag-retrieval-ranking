# Copyright (c) 2026 Sofia Soderberg. All rights reserved. View-only: see LICENSE.
"""
rag_retrieval_ranking.py - ranking and evaluation toolkit for retrieval-augmented
question answering. Pure standard library, no dependencies.

Two halves that are meant to be used together:

1. RANKING. A two-stage retriever: an approximate-nearest-neighbour search
   (any vector store, passed in as a callable) produces a candidate pool,
   then candidates are re-ranked by a *confidence score* that blends three
   signals:

       similarity   how close the stored question is to the asked one
       recency      newer documents are trusted more (clamped linear decay)
       metadata     does the document's context (industry, product line, ...)
                    match what the asker is working on?

   The blend is a weighted geometric mean rather than a product. Three
   signals of 0.8 multiply to 0.51, which reads as a weak match even though
   every signal agrees; the geometric mean returns 0.8. And when a signal
   does not exist (the asker gave no context to match against) it is
   *dropped and the remaining weights renormalised*, instead of defaulting to
   a neutral 1.0 that would silently inflate every score with a "match" that
   was never computed.

2. EVALUATION. Metrics for judging a retriever honestly on a small
   hand-labelled set: hit@k, MRR, 95% Wilson intervals (so "75%" on 36
   questions is reported as the 59-86% it actually is), and a calibration
   check - does the top-1 score separate right answers from wrong or
   unanswerable ones (AUROC, plus how often a low-confidence flag fires)?
   Retrieval variants are run side by side as an ablation so you can see
   which stage helps and which one is hurting.

Run `python rag_retrieval_ranking.py` for a demo on synthetic data. The demo shows
the machinery working; its numbers say nothing about any real system.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Callable, Iterable, Optional, Sequence, Union

DateLike = Union[date, str, None]

# Similarity matters most; recency second (stale answers can be actively
# wrong, not just less relevant); metadata is a light adjustment.
DEFAULT_WEIGHTS = (0.6, 0.25, 0.15)


# =============================================================================
# 1. Ranking
# =============================================================================

def recency_weight(doc_date: DateLike, today: Optional[date] = None, *,
                   full_weight_years: float = 1.0, max_age_years: float = 5.0,
                   floor: float = 0.75, unknown: float = 0.88) -> float:
    """1.0 for documents up to `full_weight_years` old, decaying linearly to
    `floor` at `max_age_years`. Undated or unparseable documents get
    `unknown`: below fresh, above the floor, since not knowing a date is
    weaker evidence of staleness than knowing it is old.
    """
    if doc_date is None or doc_date == "":
        return unknown
    if isinstance(doc_date, str):
        try:
            doc_date = datetime.strptime(doc_date.strip(), "%Y-%m-%d").date()
        except ValueError:
            return unknown
    age = max(0.0, ((today or date.today()) - doc_date).days / 365.25)
    if age <= full_weight_years:
        return 1.0
    if age >= max_age_years:
        return floor
    t = (age - full_weight_years) / (max_age_years - full_weight_years)
    return 1.0 - t * (1.0 - floor)


def _contains_either_way(a: str, b: str) -> bool:
    a, b = a.strip().lower(), b.strip().lower()
    return a in b or b in a


def metadata_match_weight(attrs: dict, wanted: Optional[dict], *,
                          penalty: float = 0.12, floor: float = 0.7) -> Optional[float]:
    """Multiplier in [floor, 1.0] from comparing a document's attributes with
    what the asker is working on.

    - Returns None when there is nothing to match against (`wanted` empty).
      None is not the same as 1.0; see `confidence`.
    - A key the document has no value for is treated as unknown, not as a
      mismatch - untagged documents should not be penalised for being untagged.
    - Matching is case-insensitive containment in either direction, because
      metadata is free text ("Healthcare" vs "Healthcare Provider"), not a
      closed taxonomy.
    """
    wanted = {k: v for k, v in (wanted or {}).items() if v}
    if not wanted:
        return None
    weight = 1.0
    for key, want in wanted.items():
        have = attrs.get(key)
        if have and not _contains_either_way(str(have), str(want)):
            weight -= penalty
    return max(weight, floor)


def confidence(similarity: float, recency: float, metadata: Optional[float] = None,
               weights: Sequence[float] = DEFAULT_WEIGHTS) -> float:
    """Weighted geometric mean of the available signals, in [0, 1].

    `metadata=None` drops that signal and renormalises the other two weights
    so the result stays on the same scale. Similarity is floored at 0 first:
    a fractional power of a negative base is complex, and a negative cosine
    should not contribute positively anyway.
    """
    w_sim, w_rec, w_meta = weights
    sim = max(similarity, 0.0)
    if metadata is None:
        total = w_sim + w_rec
        return sim ** (w_sim / total) * recency ** (w_rec / total)
    return sim ** w_sim * recency ** w_rec * metadata ** w_meta


@dataclass(frozen=True)
class Candidate:
    doc_id: str
    similarity: float                       # cosine similarity from the vector store
    doc_date: DateLike = None
    attrs: dict = field(default_factory=dict)
    payload: object = None                  # carry whatever you need (the stored answer, ...)


@dataclass(frozen=True)
class Ranked:
    candidate: Candidate
    score: float
    recency: float
    metadata: Optional[float]


def rerank(pool: Iterable[Candidate], wanted: Optional[dict] = None, *, top_k: int = 5,
           today: Optional[date] = None,
           weights: Sequence[float] = DEFAULT_WEIGHTS) -> list[Ranked]:
    scored = []
    for c in pool:
        rec = recency_weight(c.doc_date, today)
        meta = metadata_match_weight(c.attrs, wanted)
        scored.append(Ranked(c, confidence(c.similarity, rec, meta, weights), rec, meta))
    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:top_k]


def retrieve(search: Callable[[str, int], list[Candidate]], query: str, *,
             wanted: Optional[dict] = None, top_k: int = 5, pool_factor: int = 4,
             today: Optional[date] = None) -> list[Ranked]:
    """Stage 1: over-fetch a pool by similarity. Stage 2: re-rank it.

    Over-fetching matters: re-ranking can only promote what stage 1 returned,
    so a pool of exactly `top_k` makes stage 2 a no-op.
    """
    pool = search(query, max(top_k * pool_factor, top_k + 5))
    return rerank(pool, wanted, top_k=top_k, today=today)


# =============================================================================
# 2. Evaluation
# =============================================================================

def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion. Preferred over the normal
    approximation at small n or near 0/1, where the latter can leave [0, 1].
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def auroc(positives: Sequence[float], negatives: Sequence[float]) -> Optional[float]:
    """P(random positive scores above random negative); ties count half.
    0.5 is chance, 1.0 is perfect separation. None if either side is empty.
    """
    if not positives or not negatives:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0
               for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def first_relevant_rank(ranked_ids: Sequence[str],
                        is_relevant: Callable[[str], bool]) -> Optional[int]:
    """1-based rank of the first relevant result, or None if there isn't one."""
    for i, doc_id in enumerate(ranked_ids, 1):
        if is_relevant(doc_id):
            return i
    return None


def rank_metrics(ranks: Sequence[Optional[int]], ks: Sequence[int] = (1, 3, 5)) -> dict:
    n = len(ranks)
    out: dict = {"n": n}
    for k in ks:
        hits = sum(1 for r in ranks if r is not None and r <= k)
        out[f"hit@{k}"] = {"value": hits / n, "hits": hits, "ci95": wilson_interval(hits, n)}
    out["mrr"] = sum(1.0 / r for r in ranks if r) / n
    return out


@dataclass(frozen=True)
class EvalQuery:
    text: str
    # None => the corpus has no answer (used to test whether confidence
    # drops when it should). Otherwise: is this doc_id an acceptable answer?
    is_relevant: Optional[Callable[[str], bool]]
    wanted: Optional[dict] = None           # metadata context for the asker, if any


# A variant maps an EvalQuery to a ranked list of (doc_id, score).
Variant = Callable[[EvalQuery], list]


def evaluate(variants: dict[str, Variant], queries: Sequence[EvalQuery], *,
             flag_below: float = 0.5, ks: Sequence[int] = (1, 3, 5)) -> dict:
    """Run every variant over every query: ranking metrics on the answerable
    queries, and a calibration read on the top-1 score.
    """
    answerable = [q for q in queries if q.is_relevant is not None]
    report: dict = {}
    for name, variant in variants.items():
        ranks, correct, wrong, unanswerable = [], [], [], []
        for q in queries:
            results = variant(q)
            top_score = results[0][1] if results else None
            if q.is_relevant is None:
                if top_score is not None:
                    unanswerable.append(top_score)
                continue
            rank = first_relevant_rank([doc_id for doc_id, _ in results], q.is_relevant)
            ranks.append(rank)
            if top_score is not None:
                (correct if rank == 1 else wrong).append(top_score)

        mean = lambda xs: sum(xs) / len(xs) if xs else None
        flagged = lambda xs: sum(s < flag_below for s in xs) / len(xs) if xs else None
        report[name] = {
            "ranking": rank_metrics(ranks, ks),
            "calibration": {
                "n": {"correct": len(correct), "wrong": len(wrong), "unanswerable": len(unanswerable)},
                "mean_score": {"correct": mean(correct), "wrong": mean(wrong),
                               "unanswerable": mean(unanswerable)},
                f"flagged_below_{flag_below}": {"correct": flagged(correct), "wrong": flagged(wrong),
                                                "unanswerable": flagged(unanswerable)},
                "auroc": auroc(correct, wrong + unanswerable),
            },
        }
    report["_meta"] = {"answerable": len(answerable), "unanswerable": len(queries) - len(answerable)}
    return report


def format_report(report: dict) -> str:
    meta = report["_meta"]
    names = [n for n in report if n != "_meta"]
    lines = [f"{meta['answerable']} answerable + {meta['unanswerable']} unanswerable queries", "",
             f"{'variant':26s}{'hit@1':>8s}{'hit@3':>8s}{'hit@5':>8s}{'MRR':>8s}   hit@1 95% CI"]
    for n in names:
        r = report[n]["ranking"]
        lo, hi = r["hit@1"]["ci95"]
        lines.append(f"{n:26s}{r['hit@1']['value']:8.1%}{r['hit@3']['value']:8.1%}"
                     f"{r['hit@5']['value']:8.1%}{r['mrr']:8.3f}   {lo:.0%}-{hi:.0%}")
    lines += ["", "Does the top-1 score separate right from wrong?"]
    for n in names:
        c = report[n]["calibration"]
        f = lambda v: "  n/a" if v is None else f"{v:5.2f}"
        au = c["auroc"]
        lines.append(f"  {n:24s} mean score  correct {f(c['mean_score']['correct'])}  "
                     f"wrong {f(c['mean_score']['wrong'])}  unanswerable {f(c['mean_score']['unanswerable'])}"
                     f"   AUROC {'n/a' if au is None else f'{au:.3f}'}")
    return "\n".join(lines)


# =============================================================================
# Demo on synthetic data (shows the machinery; the numbers mean nothing)
# =============================================================================

def _unit(v: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))        # inputs are unit vectors


def _demo() -> None:
    rng = random.Random(7)
    dim, today = 24, date(2026, 1, 1)
    domains = ["finance", "health", "retail", "energy"]

    # Each "answer" is stored under several paraphrases (noisy copies of one
    # centroid) with different dates and domains - the same shape as a real
    # Q&A archive where one answer is reused across many documents.
    centroids = {f"t{i:02d}": _unit([rng.gauss(0, 1) for _ in range(dim)]) for i in range(12)}
    corpus = []
    for topic, c in centroids.items():
        for j in range(rng.randint(3, 6)):
            corpus.append({
                "id": f"{topic}:{j}", "topic": topic,
                "vec": _unit([x + rng.gauss(0, 0.30) for x in c]),
                "date": date(2026 - rng.randint(0, 5), rng.randint(1, 12), 15),
                "attrs": {"domain": rng.choice(domains)},
            })
    topic_of = {d["id"]: d["topic"] for d in corpus}

    def search(query_vec, n):                       # stage 1: brute-force "ANN"
        scored = sorted(((_cosine(query_vec, d["vec"]), d) for d in corpus),
                        key=lambda t: t[0], reverse=True)[:n]
        return [Candidate(d["id"], sim, d["date"], d["attrs"]) for sim, d in scored]

    query_vecs: dict[str, list[float]] = {}
    queries = []
    for topic, c in centroids.items():
        for k in range(3):
            text = f"{topic}-query-{k}"
            query_vecs[text] = _unit([x + rng.gauss(0, 0.55) for x in c])   # noisier than the docs
            queries.append(EvalQuery(text, lambda i, t=topic: topic_of[i] == t,
                                     wanted={"domain": rng.choice(domains)} if k == 2 else None))
    for k in range(4):                                # nothing in the corpus answers these
        text = f"unanswerable-{k}"
        query_vecs[text] = _unit([rng.gauss(0, 1) for _ in range(dim)])
        queries.append(EvalQuery(text, None))

    def similarity_only(q: EvalQuery):
        return [(c.doc_id, c.similarity) for c in search(query_vecs[q.text], 5)]

    def full_pipeline(q: EvalQuery):
        ranked = retrieve(lambda _, n: search(query_vecs[q.text], n), q.text,
                          wanted=q.wanted, today=today)
        return [(r.candidate.doc_id, r.score) for r in ranked]

    print(format_report(evaluate(
        {"A  similarity only": similarity_only, "B  full re-ranking": full_pipeline},
        queries)))
    print("\n(synthetic data: demonstrates the harness, not the method)")


if __name__ == "__main__":
    _demo()
