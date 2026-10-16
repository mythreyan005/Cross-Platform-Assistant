"""
Unit tests for evaluation metrics engine:
Canonical artifact identifier normalization, Recall@k, Precision@k,
structured graph relationship recall, and evidence-groundedness assessment.
"""

from django.test import TestCase
from evaluation.evaluate_rag import (
    normalize_artifact_id,
    artifact_ids_match,
    calculate_retrieval_metrics,
    evaluate_relationship_recall,
    evaluate_groundedness
)


class EvaluationMetricsTests(TestCase):
    def test_normalize_artifact_id(self):
        # Commit normalization
        t, v = normalize_artifact_id("de74a89")
        self.assertEqual(t, "commit")
        self.assertEqual(v, "de74a89")

        t, v = normalize_artifact_id("commit:de74a89012345678901234567890123456789012")
        self.assertEqual(t, "commit")
        self.assertEqual(v, "de74a89")

        # Jira issue normalization
        t, v = normalize_artifact_id("trnsqrtpce-980")
        self.assertEqual(t, "issue")
        self.assertEqual(v, "TRNSQRTPCE-980")

        t, v = normalize_artifact_id("issue:DHIS2-22296")
        self.assertEqual(t, "issue")
        self.assertEqual(v, "DHIS2-22296")

        # Endpoint normalization
        t, v = normalize_artifact_id("GET /api/dataValueSets")
        self.assertEqual(t, "endpoint")
        self.assertEqual(v, "GET /api/dataValueSets")

        t, v = normalize_artifact_id("endpoint:POST:/api/organisationUnits/merge")
        self.assertEqual(t, "endpoint")
        self.assertEqual(v, "POST /api/organisationUnits/merge")

        # Component normalization
        t, v = normalize_artifact_id("component:tapi")
        self.assertEqual(t, "component")
        self.assertEqual(v, "tapi")

    def test_artifact_ids_match(self):
        # Full SHA vs 7-char short SHA match
        self.assertTrue(artifact_ids_match("de74a89", "commit:de74a89012345678901234567890123456789012"))
        self.assertFalse(artifact_ids_match("de74a89", "e10ac2a"))

        # Issue match
        self.assertTrue(artifact_ids_match("TRNSPRTPCE-980", "issue:trnsprtpce-980"))
        self.assertFalse(artifact_ids_match("TRNSPRTPCE-980", "TRNSPRTPCE-1000"))

        # Endpoints must match both method and path
        self.assertTrue(artifact_ids_match("GET /api/me", "endpoint:GET:/api/me"))
        self.assertFalse(artifact_ids_match("GET /api/me", "POST /api/me"))

        # Arbitrary substring should NOT match (e.g. 'me' should not match '/api/me/authorization')
        self.assertFalse(artifact_ids_match("GET /api/me", "GET /api/me/authorization"))

    def test_calculate_retrieval_metrics_empty_retrieved(self):
        expected = ["TRNSPRTPCE-980", "de74a89"]
        retrieved = []
        recall, precision, matched = calculate_retrieval_metrics(expected, retrieved, cutoff_k=5)
        self.assertEqual(recall, 0.0)
        self.assertEqual(precision, 0.0)
        self.assertEqual(matched, [])

    def test_calculate_retrieval_metrics_empty_expected(self):
        expected = []
        retrieved = ["TRNSPRTPCE-980"]
        recall, precision, matched = calculate_retrieval_metrics(expected, retrieved, cutoff_k=5)
        self.assertEqual(recall, 1.0)
        self.assertEqual(precision, 1.0)

    def test_calculate_retrieval_metrics_mismatched_ids(self):
        expected = ["TRNSPRTPCE-980", "de74a89"]
        retrieved = ["DHIS2-100", "GET /api/analytics"]
        recall, precision, matched = calculate_retrieval_metrics(expected, retrieved, cutoff_k=5)
        self.assertEqual(recall, 0.0)
        self.assertEqual(precision, 0.0)
        self.assertEqual(len(matched), 0)

    def test_calculate_retrieval_metrics_duplicate_sources(self):
        expected = ["TRNSPRTPCE-980", "de74a89"]
        # Retrieved has duplicates of the same canonical item
        retrieved = ["TRNSPRTPCE-980", "issue:TRNSPRTPCE-980", "trnsprtpce-980", "de74a89"]
        recall, precision, matched = calculate_retrieval_metrics(expected, retrieved, cutoff_k=5)
        self.assertEqual(recall, 1.0)
        self.assertEqual(precision, 1.0)  # Deduplicated into 2 items matching 2 expected
        self.assertEqual(len(matched), 2)

    def test_calculate_retrieval_metrics_partial_match(self):
        expected = ["TRNSPRTPCE-980", "de74a89", "e10ac2a"]
        retrieved = ["TRNSPRTPCE-980", "other_doc_1", "other_doc_2"]
        recall, precision, matched = calculate_retrieval_metrics(expected, retrieved, cutoff_k=5)
        self.assertEqual(recall, round(1 / 3, 4))
        self.assertEqual(precision, round(1 / 3, 4))
        self.assertEqual(matched, ["TRNSPRTPCE-980"])

    def test_evaluate_relationship_recall_known_relationships(self):
        expected_edges = [
            {"source": "commit:de74a89", "target": "issue:TRNSPRTPCE-980", "relationship": "REFERENCES_ISSUE"},
            {"source": "commit:de74a89", "target": "component:tapi", "relationship": "MODIFIES_COMPONENT"}
        ]
        retrieved_edges = [
            {"source": "commit:de74a89", "target": "issue:TRNSPRTPCE-980", "relationship": "REFERENCES_ISSUE"},
            {"source": "commit:de74a89", "target": "component:tapi", "relationship": "MODIFIES_COMPONENT"},
            {"source": "issue:TRNSPRTPCE-980", "target": "project:transportpce", "relationship": "BELONGS_TO"}
        ]
        recall, matched = evaluate_relationship_recall(expected_edges, retrieved_edges)
        self.assertEqual(recall, 1.0)
        self.assertEqual(len(matched), 2)

    def test_evaluate_relationship_recall_wrong_type(self):
        expected_edges = [
            {"source": "commit:de74a89", "target": "issue:TRNSPRTPCE-980", "relationship": "REFERENCES_ISSUE"}
        ]
        # Same source and target, but wrong relationship type
        retrieved_edges = [
            {"source": "commit:de74a89", "target": "issue:TRNSPRTPCE-980", "relationship": "RELATED_TO"}
        ]
        recall, matched = evaluate_relationship_recall(expected_edges, retrieved_edges)
        self.assertEqual(recall, 0.0)
        self.assertEqual(len(matched), 0)

    def test_evaluate_relationship_recall_inverted_direction(self):
        expected_edges = [
            {"source": "commit:de74a89", "target": "issue:TRNSPRTPCE-980", "relationship": "REFERENCES_ISSUE"}
        ]
        # Inverted direction: issue -> commit
        retrieved_edges = [
            {"source": "issue:TRNSPRTPCE-980", "target": "commit:de74a89", "relationship": "REFERENCES_ISSUE"}
        ]
        recall, matched = evaluate_relationship_recall(expected_edges, retrieved_edges)
        self.assertEqual(recall, 0.0)
        self.assertEqual(len(matched), 0)

    def test_evaluate_groundedness(self):
        retrieved_sources = ["TRNSPRTPCE-980", "de74a89"]
        graph_edges = [
            {"source": "commit:de74a89", "target": "issue:TRNSPRTPCE-980", "relationship": "REFERENCES_ISSUE"}
        ]

        # Fully grounded answer citing supported entities
        answer_grounded = "Commit `de74a89` directly references Jira ticket `TRNSPRTPCE-980` for the path computation fix."
        score = evaluate_groundedness(answer_grounded, retrieved_sources, graph_edges, "SUCCESS")
        self.assertEqual(score, 1.0)

        # Partially grounded / hallucinated answer citing ungrounded ticket
        answer_hallucinated = "Commit `de74a89` resolved `TRNSPRTPCE-980` and also fixed `FAKE-9999`."
        score_hallucinated = evaluate_groundedness(answer_hallucinated, retrieved_sources, graph_edges, "SUCCESS")
        self.assertEqual(score_hallucinated, round(2 / 3, 4))

        # Truthful refusal on empty retrieval
        score_refusal = evaluate_groundedness("No relevant documents found.", [], [], "EMPTY_RETRIEVAL")
        self.assertEqual(score_refusal, 1.0)
