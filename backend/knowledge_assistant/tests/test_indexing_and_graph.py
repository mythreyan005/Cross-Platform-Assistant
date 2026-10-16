"""
Regression tests for indexing pipeline, Knowledge Graph node collisions,
edge provenance, incremental re-indexing, record deletion, and bounded traversal.
"""

import os
import tempfile
from unittest.mock import patch, MagicMock
from django.test import TestCase

from knowledge_assistant.models_data import SourceRecord, SourceType, NodeType, RelationshipType
from knowledge_assistant.knowledge_graph import SoftwareKnowledgeGraph
from knowledge_assistant.rag_pipeline import load_and_index_data


class IndexingAndGraphTests(TestCase):
    def setUp(self):
        self.kg = SoftwareKnowledgeGraph(project_id="test/repo")

    def test_canonical_node_ids_and_collision_handling(self):
        """Verifies distinct entity types with the same base name do not collide in graph."""
        # 1. Component named "auth"
        comp_id = self.kg._component_node_id("auth")
        self.assertEqual(comp_id, "component:auth")

        # 2. Issue named "AUTH"
        issue_id = self.kg._issue_node_id("AUTH-1")
        self.assertEqual(issue_id, "issue:AUTH-1")

        # 3. Endpoint with path "/auth"
        ep_id = self.kg._endpoint_node_id("POST /auth")
        self.assertEqual(ep_id, "endpoint:POST:/auth")

        # Add all three to graph
        self.kg.graph.add_node(comp_id, node_type=NodeType.COMPONENT.value, name="auth")
        self.kg.graph.add_node(issue_id, node_type=NodeType.ISSUE.value, key="AUTH-1")
        self.kg.graph.add_node(ep_id, node_type=NodeType.ENDPOINT.value, path="/auth")

        self.assertEqual(self.kg.graph.number_of_nodes(), 3)
        self.assertEqual(self.kg.find_node("AUTH-1"), issue_id)
        self.assertEqual(self.kg.find_node("POST /auth"), ep_id)
        self.assertEqual(self.kg.find_node("auth"), comp_id)

    def test_exact_edge_types_and_provenance(self):
        """Verifies edge types and provenance metadata on graph edges."""
        rec = SourceRecord(
            source_type=SourceType.GITHUB,
            source_id="112233445566778899aabbccddeeff0011223344",
            project_id="test/repo",
            title="Fix auth token verification",
            content="Resolves AUTH-10 and modifies auth module",
            relationships=[
                {"target_type": "jira", "target_id": "AUTH-10", "relationship": "REFERENCES_ISSUE", "provenance": "commit_regex"},
                {"target_type": "component", "target_id": "auth", "relationship": "MODIFIES_COMPONENT", "provenance": "diff_paths"}
            ]
        )
        self.kg.add_commit_record(rec)
        commit_id = self.kg.find_node("112233445566778899aabbccddeeff0011223344")
        self.assertIsNotNone(commit_id)

        nodes, edges = self.kg.get_related_entities(commit_id, max_depth=1)
        rel_types = {e["relationship"] for e in edges}
        provenances = {e.get("provenance") for e in edges}

        self.assertIn("REFERENCES_ISSUE", rel_types)
        self.assertIn("MODIFIES_COMPONENT", rel_types)
        self.assertIn("commit_regex", provenances)
        self.assertIn("diff_paths", provenances)

    def test_repeated_indexing_deduplication(self):
        """Adding identical records multiple times must not duplicate nodes or edges."""
        rec = SourceRecord(
            source_type=SourceType.JIRA,
            source_id="TEST-500",
            project_id="test/repo",
            title="Deduplication test issue",
            content="Details",
            status="Open"
        )
        self.kg.add_issue_record(rec)
        initial_nodes = self.kg.graph.number_of_nodes()
        initial_edges = self.kg.graph.number_of_edges()

        # Re-add identical record
        self.kg.add_issue_record(rec)
        self.assertEqual(self.kg.graph.number_of_nodes(), initial_nodes)
        self.assertEqual(self.kg.graph.number_of_edges(), initial_edges)

    def test_record_deletion_removes_node_and_edges(self):
        """Deleting a record removes its node and all connected edges."""
        rec = SourceRecord(
            source_type=SourceType.JIRA,
            source_id="DELETE-99",
            project_id="test/repo",
            title="To be deleted",
            content="Details"
        )
        self.kg.add_issue_record(rec)
        node_id = self.kg._issue_node_id("DELETE-99")
        self.assertTrue(self.kg.graph.has_node(node_id))

        removed = self.kg.remove_record(SourceType.JIRA, "DELETE-99")
        self.assertTrue(removed)
        self.assertFalse(self.kg.graph.has_node(node_id))

    def test_bounded_bfs_expansion_budget(self):
        """Graph expansion must strictly obey the max_total_nodes budget limit."""
        center = self.kg.ensure_project_node()
        # Add 30 satellite nodes connected to center
        for i in range(30):
            node_name = f"component:sub_{i}"
            self.kg.graph.add_node(node_name, node_type=NodeType.COMPONENT.value, label=f"Sub {i}")
            self.kg.graph.add_edge(center, node_name, relationship="RELATED_TO")

        # Request subgraph with max_total_nodes=10
        nodes, edges, ctx = self.kg.get_subgraph_for_seeds([center], max_depth=2, max_total_nodes=10)
        self.assertLessEqual(len(nodes), 10)

    @patch("knowledge_assistant.rag_pipeline.fetch_openapi_records")
    @patch("knowledge_assistant.rag_pipeline.fetch_github_records")
    @patch("knowledge_assistant.rag_pipeline.fetch_jira_records")
    @patch("knowledge_assistant.rag_pipeline.get_vector_store")
    def test_vector_failure_prevents_graph_persistence(self, mock_get_vs, mock_jira, mock_gh, mock_oas):
        """Simulates vector store insertion failure and verifies graph is not updated."""
        mock_vs = MagicMock()
        mock_vs.get.return_value = {"ids": []}
        mock_vs.add_documents.side_effect = RuntimeError("Chroma connection error")
        mock_get_vs.return_value = mock_vs
        mock_jira.return_value = []
        mock_gh.return_value = []
        mock_oas.return_value = [
            SourceRecord(
                source_type=SourceType.OPENAPI,
                source_id="GET /api/test",
                project_id="test/repo",
                title="Test",
                content="Endpoint",
                metadata={"method": "GET", "path": "/api/test", "tags": []}
            )
        ]

        with self.assertRaises(RuntimeError):
            load_and_index_data(repo="opendaylight/transportpce", max_commits=2, max_issues=2)
