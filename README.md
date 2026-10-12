# rag-retrieval-ranking

Re-ranking and honest evaluation for retrieval-augmented question answering.
Pure Python standard library, one file, no dependencies.

This is the scoring and evaluation core extracted from a retrieval system I
built that answers new RFP/RFI questions from an archive of past responses.
The application around it (ingestion, vector store, LLM query analysis,
approval workflow, UI) is private and not part of this repo.

> **View-only.** This code is published to be read, not reused. See
> [LICENSE](LICENSE).

## Two ideas

### 1. A confidence score you can trust

Stage 1 over-fetches candidates by vector similarity. Stage 2 re-ranks them
by a **weighted geometric mean** of three signals:

| signal | meaning | default weight |
|---|---|---|
| similarity | how close the stored question is to the one asked | 0.60 |
| recency | newer documents are trusted more (clamped linear decay) | 0.25 |
| metadata | does the document's context match what the asker is working on | 0.15 |

Design decisions that matter:

- **Geometric mean, not a product.** Three signals of 0.8 multiply to 0.51 and
  read as a weak match even though every signal agrees. The geometric mean
  returns 0.8.
- **A missing signal is dropped, not defaulted.** If the asker gave no context,
  metadata is `None`: it is removed and the other weights are renormalised.
  Defaulting to 1.0 would silently credit every result with a "match" that was
  never computed.
- **Untagged is not mismatched.** A document with no value for an attribute is
  not penalised for it.
- **Over-fetch the pool.** Re-ranking can only promote what stage 1 returned;
  a pool of exactly `top_k` makes stage 2 a no-op.

### 2. Evaluation that admits its uncertainty

- hit@k and MRR on a hand-labelled set, with **95% Wilson intervals**, so
  "75%" on 36 questions is reported as the 59-86% it actually is.
- **Calibration:** does the top-1 score separate right answers from wrong or
  unanswerable ones? Reported as mean score per group, how often a
  low-confidence flag fires, and AUROC.
- **Ablation:** retrieval variants run side by side, so you can see which stage
  helps and which one hurts.

## Usage

```python
from rag_ranking_eval import Candidate, retrieve

def search(query, n):                      # any vector store
    return [Candidate(doc_id, cosine, "2024-03-15", {"industry": "retail"}), ...]

top = retrieve(search, "What are your support SLAs?",
               wanted={"industry": "retail"}, top_k=5)
for r in top:
    print(r.candidate.doc_id, round(r.score, 2), r.recency, r.metadata)
```

`python rag_ranking_eval.py` runs a self-contained demo on synthetic data and
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

The demo's numbers only show the harness working; they say nothing about any
real system.

## What I measured on the real system

A hand-labelled set of 36 answerable and 6 unanswerable questions, run against
a small synthetic archive (20 documents, 148 Q&A rows, 31 distinct answers,
`all-MiniLM-L6-v2` embeddings). Three variants:

| variant | hit@1 | 95% CI | hit@5 | MRR |
|---|---|---|---|---|
| A: cosine similarity only | 75.0% | 59-86% | 83.3% | 0.78 |
| B: + recency | 75.0% | 59-86% | 83.3% | 0.78 |
| C: + recency + metadata (full pipeline) | 61.1% | 45-75% | 69.4% | 0.63 |

- On this set the metadata stage **reduced** top-1 accuracy, and recency left
  the ranking unchanged. I report that rather than hide it: the set is small,
  the intervals overlap, and the archive has few distinct answers.
- The score's real value was **calibration**: AUROC of about 0.91 for
  separating correct top-1 results from wrong or unanswerable ones, which is
  what lets the application flag low-confidence answers for human review.

## Not included, on purpose

Prompts, the ingestion pipeline, the vector-store integration, the LLM
relevance gate, expert routing, the approval workflow, the UI and all data.
