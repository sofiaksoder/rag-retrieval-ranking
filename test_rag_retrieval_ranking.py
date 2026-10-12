# Copyright (c) 2026 Sofia Soderberg. All rights reserved. View-only: see LICENSE.
import unittest
from datetime import date

import rag_retrieval_ranking as r

TODAY = date(2026, 1, 1)


class Recency(unittest.TestCase):
    def test_fresh_is_full_weight(self):
        self.assertEqual(r.recency_weight("2025-06-01", TODAY), 1.0)

    def test_decays_linearly_to_floor(self):
        self.assertAlmostEqual(r.recency_weight("2023-01-01", TODAY), 0.875, places=2)
        self.assertEqual(r.recency_weight("2015-01-01", TODAY), 0.75)

    def test_unknown_sits_between_fresh_and_floor(self):
        for bad in (None, "", "not-a-date"):
            self.assertEqual(r.recency_weight(bad, TODAY), 0.88)


class Metadata(unittest.TestCase):
    def test_nothing_to_match_is_none_not_one(self):
        self.assertIsNone(r.metadata_match_weight({"domain": "finance"}, None))
        self.assertIsNone(r.metadata_match_weight({"domain": "finance"}, {"domain": ""}))

    def test_untagged_document_is_not_penalised(self):
        self.assertEqual(r.metadata_match_weight({}, {"domain": "finance"}), 1.0)

    def test_containment_matches_either_way(self):
        self.assertEqual(r.metadata_match_weight({"domain": "Healthcare Provider"},
                                                 {"domain": "healthcare"}), 1.0)

    def test_mismatch_is_penalised_but_floored(self):
        w = r.metadata_match_weight({"a": "x", "b": "x", "c": "x", "d": "x", "e": "x", "f": "x"},
                                    {k: "y" for k in "abcdef"})
        self.assertEqual(w, 0.7)


class Confidence(unittest.TestCase):
    def test_agreeing_signals_are_not_diluted(self):
        self.assertAlmostEqual(r.confidence(0.8, 0.8, 0.8), 0.8)

    def test_missing_metadata_renormalises_instead_of_inflating(self):
        self.assertAlmostEqual(r.confidence(0.8, 0.8, None), 0.8)

    def test_negative_similarity_cannot_go_complex(self):
        self.assertEqual(r.confidence(-0.3, 1.0, 1.0), 0.0)


class Evaluation(unittest.TestCase):
    def test_wilson_interval_stays_in_bounds(self):
        lo, hi = r.wilson_interval(0, 5)
        self.assertEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)
        lo, hi = r.wilson_interval(27, 36)
        self.assertAlmostEqual(lo, 0.589, places=2)
        self.assertAlmostEqual(hi, 0.862, places=2)

    def test_auroc_extremes_and_ties(self):
        self.assertEqual(r.auroc([0.9], [0.1]), 1.0)
        self.assertEqual(r.auroc([0.5], [0.5]), 0.5)
        self.assertIsNone(r.auroc([], [0.1]))

    def test_rerank_can_promote_a_fresher_candidate(self):
        old = r.Candidate("old", 0.80, "2019-01-01")
        new = r.Candidate("new", 0.78, "2025-09-01")
        top = r.rerank([old, new], top_k=1, today=TODAY)[0]
        self.assertEqual(top.candidate.doc_id, "new")


if __name__ == "__main__":
    unittest.main()
