"""
Unit tests for evaluation metrics engine:
1. Identifier normalization and matching (full and short commit SHAs, Jira keys, REST endpoints).
2. Precision@k, Recall@k, and F1@k with fair retrieval cutoffs.
3. Duplicate chunk handling (source record deduplication).
4. Empty retrieval and empty ground-truth safety.
5. Structured graph-edge recall with edge budget limit.
6. Execution status handling (SUCCESS, EXTRACTIVE_FALLBACK, EMPTY_RETRIEVAL, API_ERROR).
7. Evidence-based groundedness scoring without hardcoded values.
"""

from django.test import TestCase
from evaluation.evaluate_rag import (
    normalize_artifact_id,
    artifact_ids_match,
    calculate_retrieval_metrics,
    evaluate_relationship_recall,
    evaluate_groundedness,
    validate_hybrid_execution,
    ClaimVerificationProtocol
)


class EvaluationMetricsTests(TestCase):

    def test_normalize_artifact_id(self):
        # Full 40-char commit SHA
        full_sha = "de74a89012345678901234567890123456789012"
        t, v = normalize_artifact_id(full_sha)
        self.assertEqual(t, "commit")
        self.assertEqual(v, "de74a89")

        # Prefixed commit SHA
        t, v = normalize_artifact_id(f"commit:{full_sha}")
        self.assertEqual(t, "commit")
        self.assertEqual(v, "de74a89")

        # Short 7-char SHA
        t, v = normalize_artifact_id("de74a89")
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
        self.assertTrue(artifact_ids_match("commit:de74a89", "de74a89012345678901234567890123456789012"))
        self.assertFalse(artifact_ids_match("de74a89", "e10ac2a"))

        # Issue match
        self.assertTrue(artifact_ids_match("TRNSPRTPCE-980", "issue:trnsprtpce-980"))
        self.assertFalse(artifact_ids_match("TRNSPRTPCE-980", "TRNSPRTPCE-1000"))

        # Endpoints must match both method and path
        self.assertTrue(artifact_ids_match("GET /api/me", "endpoint:GET:/api/me"))
        self.assertFalse(artifact_ids_match("GET /api/me", "POST /api/me"))

        # Arbitrary substring should NOT match
        self.assertFalse(artifact_ids_match("GET /api/me", "GET /api/me/authorization"))

    def test_calculate_retrieval_metrics_empty_retrieved(self):
        expected = ["TRNSPRTPCE-980", "de74a89"]
        retrieved = []
        recall, precision, f1, matched = calculate_retrieval_metrics(expected, retrieved, cutoff_k=5)
        self.assertEqual(recall, 0.0)
        self.assertEqual(precision, 0.0)
        self.assertEqual(f1, 0.0)
        self.assertEqual(matched, [])

    def test_calculate_retrieval_metrics_empty_expected(self):
        expected = []
        retrieved = ["TRNSPRTPCE-980"]
        recall, precision, f1, matched = calculate_retrieval_metrics(expected, retrieved, cutoff_k=5)
        self.assertEqual(recall, 1.0)
        self.assertEqual(precision, 1.0)
        self.assertEqual(f1, 1.0)
        self.assertEqual(matched, [])

    def test_calculate_retrieval_metrics_mismatched_ids(self):
        expected = ["TRNSPRTPCE-980", "de74a89"]
        retrieved = ["DHIS2-100", "GET /api/analytics"]
        recall, precision, f1, matched = calculate_retrieval_metrics(expected, retrieved, cutoff_k=5)
        self.assertEqual(recall, 0.0)
        self.assertEqual(precision, 0.0)
        self.assertEqual(f1, 0.0)
        self.assertEqual(len(matched), 0)

    def test_calculate_retrieval_metrics_duplicate_sources(self):
        expected = ["TRNSPRTPCE-980", "de74a89"]
        # Retrieved has duplicates of the same canonical source record (e.g. multiple chunks)
        retrieved = ["TRNSPRTPCE-980", "issue:TRNSPRTPCE-980", "trnsprtpce-980", "de74a89"]
        recall, precision, f1, matched = calculate_retrieval_metrics(expected, retrieved, cutoff_k=5)
        self.assertEqual(recall, 1.0)
        self.assertEqual(precision, 1.0)  # Deduplicated to 2 items
        self.assertEqual(f1, 1.0)
        self.assertEqual(len(matched), 2)

    def test_calculate_retrieval_metrics_fair_cutoff_k(self):
        expected = ["item-1", "item-2", "item-3"]
        retrieved = ["item-1", "other-a", "other-b", "other-c", "other-d", "item-2", "item-3"]
        
        # At cutoff_k = 5, only items 0..4 are considered. item-1 is matched, item-2 and 3 are outside cutoff
        recall_5, prec_5, f1_5, matched_5 = calculate_retrieval_metrics(expected, retrieved, cutoff_k=5)
        self.assertEqual(recall_5, round(1 / 3, 4))
        self.assertEqual(prec_5, round(1 / 5, 4))
        self.assertEqual(matched_5, ["item-1"])

        # Without cutoff, all 3 are matched
        recall_all, prec_all, f1_all, matched_all = calculate_retrieval_metrics(expected, retrieved, cutoff_k=None)
        self.assertEqual(recall_all, 1.0)
        self.assertEqual(prec_all, round(3 / 7, 4))

    def test_evaluate_relationship_recall_with_budget(self):
        expected_edges = [
            {"source": "commit:de74a89", "target": "issue:TRNSPRTPCE-980", "relationship": "REFERENCES_ISSUE"},
            {"source": "commit:de74a89", "target": "component:tapi", "relationship": "MODIFIES_COMPONENT"}
        ]
        retrieved_edges = [
            {"source": "commit:de74a89", "target": "issue:TRNSPRTPCE-980", "relationship": "REFERENCES_ISSUE"},
            {"source": "commit:de74a89", "target": "component:tapi", "relationship": "MODIFIES_COMPONENT"},
            {"source": "issue:TRNSPRTPCE-980", "target": "project:transportpce", "relationship": "BELONGS_TO"}
        ]
        recall, matched = evaluate_relationship_recall(expected_edges, retrieved_edges, edge_budget=30)
        self.assertEqual(recall, 1.0)
        self.assertEqual(len(matched), 2)

    def test_evaluate_relationship_recall_empty_cases(self):
        # Empty expected edges -> 1.0
        recall, matched = evaluate_relationship_recall([], [{"source": "a", "target": "b", "relationship": "R"}])
        self.assertEqual(recall, 1.0)

        # Empty retrieved edges -> 0.0
        recall, matched = evaluate_relationship_recall([{"source": "a", "target": "b", "relationship": "R"}], [])
        self.assertEqual(recall, 0.0)

    def test_evaluate_groundedness_statuses(self):
        # EMPTY_RETRIEVAL must receive 0.0 groundedness
        res_empty = evaluate_groundedness(
            "No relevant documents found.",
            [], [], evidence_texts=[], status="EMPTY_RETRIEVAL"
        )
        self.assertEqual(res_empty["score"], 0.0)

        # API_ERROR must receive 0.0 groundedness
        res_error = evaluate_groundedness(
            "Service error occurred.",
            [], [], evidence_texts=[], status="API_ERROR"
        )
        self.assertEqual(res_error["score"], 0.0)

    def test_evaluate_groundedness_supported_claims(self):
        retrieved_sources = ["TRNSPRTPCE-980", "de74a89"]
        graph_edges = [
            {"source": "commit:de74a89", "target": "issue:TRNSPRTPCE-980", "relationship": "REFERENCES_ISSUE"}
        ]
        evidence_texts = [
            "Commit de74a89 was merged to fix bug TRNSPRTPCE-980 regarding tapi transport service path computation."
        ]

        # Fully grounded answer citing supported entities
        answer_grounded = "Commit `de74a89` directly references Jira ticket `TRNSPRTPCE-980` for the path computation fix."
        res = evaluate_groundedness(answer_grounded, retrieved_sources, graph_edges, evidence_texts, status="SUCCESS")
        self.assertEqual(res["score"], 1.0)
        self.assertEqual(res["supported_claims"], 1)
        self.assertEqual(res["unsupported_claims"], 0)

    def test_evaluate_groundedness_hallucinated_entity(self):
        retrieved_sources = ["TRNSPRTPCE-980", "de74a89"]
        graph_edges = [
            {"source": "commit:de74a89", "target": "issue:TRNSPRTPCE-980", "relationship": "REFERENCES_ISSUE"}
        ]
        evidence_texts = ["Commit de74a89 references TRNSPRTPCE-980."]

        # Cites unsupported Jira ticket FAKE-9999
        answer_hallucinated = "Commit `de74a89` resolved `TRNSPRTPCE-980` and also fixed `FAKE-9999`."
        res = evaluate_groundedness(answer_hallucinated, retrieved_sources, graph_edges, evidence_texts, status="SUCCESS")
        self.assertEqual(res["score"], 0.0)
        self.assertEqual(res["supported_claims"], 0)
        self.assertEqual(res["unsupported_claims"], 1)
        self.assertNotIn("contradicted_claims", res)
        self.assertEqual(res["method"], "lexical_and_entity_overlap_heuristic")

    def test_evaluate_groundedness_empty_evidence_corpus(self):
        # Empty evidence must result in 0.0 groundedness and all claims unsupported
        answer = "The authentication service verifies user credentials against database."
        res = evaluate_groundedness(answer, [], [], evidence_texts=[], status="SUCCESS")
        self.assertEqual(res["score"], 0.0)
        self.assertEqual(res["total_claims"], 1)
        self.assertEqual(res["supported_claims"], 0)
        self.assertEqual(res["unsupported_claims"], 1)

    def test_evaluate_groundedness_partial_claim_support(self):
        retrieved_sources = ["TRNSPRTPCE-980"]
        evidence_texts = ["Jira ticket TRNSPRTPCE-980 resolves topology link calculations."]

        # Sentence 1: Supported entity TRNSPRTPCE-980
        # Sentence 2: Unsupported entity FAKE-1234
        answer = (
            "TRNSPRTPCE-980 resolves topology link calculation issues in TransportPCE. "
            "However ticket FAKE-1234 is also modified in this release."
        )
        res = evaluate_groundedness(answer, retrieved_sources, [], evidence_texts, status="SUCCESS")
        self.assertEqual(res["total_claims"], 2)
        self.assertEqual(res["supported_claims"], 1)
        self.assertEqual(res["unsupported_claims"], 1)
        self.assertEqual(res["score"], 0.5)

    def test_evaluate_groundedness_no_technical_identifiers(self):
        evidence_texts = [
            "The authentication module validates user permissions against the central access control database."
        ]

        # Claim with strong token overlap against evidence
        supported_conceptual = "The authentication service verifies user permissions against access control databases."
        res_supp = evaluate_groundedness(supported_conceptual, [], [], evidence_texts, status="SUCCESS")
        self.assertGreater(res_supp["score"], 0.0)

        # Claim with unrelated tokens (hallucination)
        unsupported_conceptual = "The quantum encryption algorithm generates polymorphic cryptographic keys."
        res_unsupp = evaluate_groundedness(unsupported_conceptual, [], [], evidence_texts, status="SUCCESS")
        self.assertEqual(res_unsupp["score"], 0.0)

    def test_validate_hybrid_execution_genuine_hybrid_success(self):
        """A genuine hybrid run with retrieved entities and matching counts passes validation."""
        valid_res = {
            "retrieval_mode": "hybrid_graph_rag",
            "status": "SUCCESS",
            "graph_entities": [{"id": "issue:TRNSPRTPCE-980"}],
            "graph_edges": [{"source": "commit:de74a89", "target": "issue:TRNSPRTPCE-980", "relationship": "REFERENCES_ISSUE"}],
            "graph_node_count": 1,
            "graph_edge_count": 1
        }
        status, mode, err = validate_hybrid_execution(valid_res)
        self.assertEqual(status, "SUCCESS")
        self.assertEqual(mode, "hybrid_graph_rag")
        self.assertIsNone(err)

    def test_validate_hybrid_execution_missing_retrieval_mode(self):
        """Missing or unknown retrieval mode fails validation and produces INVALID_RETRIEVAL_MODE."""
        bad_res = {
            "status": "SUCCESS",
            "graph_entities": [{"id": "issue:1"}]
        }
        status, mode, err = validate_hybrid_execution(bad_res)
        self.assertEqual(status, "INVALID_RETRIEVAL_MODE")
        self.assertIn("Invalid or missing retrieval_mode", err)

    def test_validate_hybrid_execution_vector_fallback_classified_explicitly(self):
        """Legitimate vector fallback is classified as HYBRID_FALLBACK_VECTOR, not full hybrid SUCCESS."""
        fallback_res = {
            "retrieval_mode": "vector_only_fallback",
            "status": "HYBRID_FALLBACK_VECTOR",
            "fallback_reason": "NO_CONNECTING_RELATIONSHIPS",
            "graph_entities": [],
            "graph_edges": [],
            "graph_node_count": 0,
            "graph_edge_count": 0
        }
        status, mode, reason = validate_hybrid_execution(fallback_res)
        self.assertEqual(status, "HYBRID_FALLBACK_VECTOR")
        self.assertEqual(mode, "vector_only_fallback")
        self.assertEqual(reason, "NO_CONNECTING_RELATIONSHIPS")

    def test_validate_hybrid_execution_zero_entities_claiming_hybrid_flagged(self):
        """A run claiming hybrid_graph_rag but returning zero entities is flagged as INVALID_EXECUTION_MODE."""
        invalid_res = {
            "retrieval_mode": "hybrid_graph_rag",
            "status": "SUCCESS",
            "graph_entities": [],
            "graph_edges": [],
            "graph_node_count": 0,
            "graph_edge_count": 0
        }
        status, mode, err = validate_hybrid_execution(invalid_res)
        self.assertEqual(status, "INVALID_EXECUTION_MODE")
        self.assertIn("0 graph entities retrieved", err)

    def test_validate_hybrid_execution_count_mismatch_flagged(self):
        """Mismatch between reported counts and actual list lengths is flagged as INVALID_METADATA."""
        mismatched_res = {
            "retrieval_mode": "hybrid_graph_rag",
            "status": "SUCCESS",
            "graph_entities": [{"id": "issue:1"}],
            "graph_edges": [],
            "graph_node_count": 10,  # Reported 10, actual 1
            "graph_edge_count": 0
        }
        status, mode, err = validate_hybrid_execution(mismatched_res)
        self.assertEqual(status, "INVALID_METADATA")
        self.assertIn("graph_node_count mismatch", err)

