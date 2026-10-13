# rag-retrieval-ranking

Re-ranking and honest evaluation for retrieval-augmented question answering
in pure Python standard library.

This project is a snippet of a retrieval system I professionally built for 
an internal software prototype. It includes the scoring and evaluation core
that supports answering new RFP/RFI questions using an archive of past 
responses.

The internal application (ingestion, vector store, LLM query analysis, approval 
workflow, and UI) is private and not part of this repo.

> **View-only.** This code is published to be read, not reused. See
> [LICENSE](LICENSE).

## The core in two parts

### 1. Trust-forward confidence scoring

Stage 1 over-fetches candidates by vector similarity. Stage 2 re-ranks them
by a **weighted geometric mean** of three signals:

| signal | meaning | default weight |
|---|---|---|
| similarity | how close the stored question is to the one asked | 0.60 |
| recency | newer documents are trusted more (clamped linear decay) | 0.25 |
| metadata | the degree of context similarity among available metadata values | 0.15 |

Intentional design decisions:

- **Geometric mean instead of a standard product.** Three signals of 0.8 multiply to 0.51 and
  read as a weak match even though every signal agrees. The geometric mean instead
  returns 0.8.
- **A missing signal is dropped, not defaulted.** If the asker gave no context,
  metadata is `None`: the factor is removed and the other weights are renormalised.
  Defaulting to 1.0 would silently credit every result with a "match" that was
  never computed.
- **Missing values don't hurt the score.** A document with no value for an attribute
  is not penalised for it.
- **Over-fetch the candidate pool.** Re-ranking can’t surface documents stage 1 didn’t
  retrieve. Fetch more than `top_k` so stage 2 has extra candidates to choose from.

### 2. Evaluation with confidence transparency

- hit@k and MRR on a hand-labelled set, with **95% Wilson intervals**, so
  "75%" on 36 questions is reported as the 59-86% it actually is.
- **Calibration:** does the top-1 score separate right answers from wrong or
  unanswerable ones? Reported as mean score per group, how often a
  low-confidence flag fires, and AUROC.
- **Ablation:** retrieval variants run side by side, so you can see which stage
  helps and which one hurts.

## Usage

```python
from rag_retrieval_ranking import Candidate, retrieve

def search(query, n):                      # any vector store
    return [Candidate(doc_id, cosine, "2024-03-15", {"industry": "retail"}), ...]

top = retrieve(search, "What are your support SLAs?",
               wanted={"industry": "retail"}, top_k=5)
for r in top:
    print(r.candidate.doc_id, round(r.score, 2), r.recency, r.metadata)
```

`python rag_retrieval_ranking.py` runs a self-contained demo on synthetic data and
`python -m unittest` runs the tests.

```text
36 answerable + 4 unanswerable queries

variant                      hit@1   hit@3   hit@5     MRR   hit@1 95% CI
A  similarity only           58.3%   75.0%   80.6%   0.671   42%-73%
B  full re-ranking           61.1%   75.0%   80.6%   0.681   45%-75%

Does the top-1 score separate right from wrong?
  A  similarity only       mean score  correct  0.50  wrong  0.44  unanswerable  0.50   AUROC 0.634
  B  full re-ranking       mean score  correct  0.60  wrong  0.56  unanswerable  0.60   AUROC 0.629

(synthetic data: demonstrates the harness, not the method)
```

## A Use Case of the real prototype

A hand-labelled set of 36 answerable and 6 unanswerable questions, run against
a synthetic archive (20 documents, 148 Q&A rows, 31 distinct answers,
`all-MiniLM-L6-v2` embeddings). Three variants:

| Variant | hit@1 | 95% CI | hit@5 | MRR |
|---|---|---|---|---|
| A: cosine similarity only | 75.0% | 59-86% | 83.3% | 0.78 |
| B: + recency | 75.0% | 59-86% | 83.3% | 0.78 |
| C: + recency + metadata (full pipeline) | 61.1% | 45-75% | 69.4% | 0.63 |

- On this set the metadata stage **reduced** top-1 accuracy, and recency left
  the ranking unchanged.
- The score's real value was **calibration**: AUROC of about 0.91 for
  separating correct top-1 results from wrong or unanswerable ones, which is
  what lets the application flag low-confidence answers for human review -
  directed towards a subject-matter expert.

## Not included

Prompts, the ingestion pipeline, the vector-store integration, the LLM
relevance gate, expert routing, the approval workflow, the UI and all data.
