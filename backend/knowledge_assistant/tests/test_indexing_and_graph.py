"""
Regression tests for indexing pipeline, Knowledge Graph node collisions,
edge provenance, incremental re-indexing, record deletion, bounded traversal,
safe vector updates, full SHA canonicalization, and index freshness checks.
"""

import os
import json
import tempfile
from unittest.mock import patch, MagicMock
from django.test import TestCase

from knowledge_assistant.models_data import SourceRecord, SourceType, NodeType, RelationshipType
from knowledge_assistant.knowledge_graph import SoftwareKnowledgeGraph
from knowledge_assistant.rag_pipeline import (
    load_and_index_data,
    check_index_freshness,
    PROJECT_CONFIGS
)


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

    def test_full_sha_canonical_lookup_and_short_sha_tolerance(self):
        """Verifies full commit SHAs are canonical and short SHAs resolve cleanly."""
        full_sha = "abcdef0123456789abcdef0123456789abcdef01"
        rec = SourceRecord(
            source_type=SourceType.GITHUB,
            source_id=full_sha,
            project_id="test/repo",
            title="Commit test",
            content="Testing full SHA canonical resolution",
            metadata={"sha": full_sha, "short_sha": full_sha[:7]}
        )
        self.kg.add_commit_record(rec)

        expected_node_id = f"commit:{full_sha}"
        self.assertTrue(self.kg.graph.has_node(expected_node_id))

        # Lookup via full SHA
        self.assertEqual(self.kg.find_node(full_sha), expected_node_id)
        # Lookup via short SHA
        self.assertEqual(self.kg.find_node(full_sha[:7]), expected_node_id)
        # Lookup via prefixed full SHA
        self.assertEqual(self.kg.find_node(f"commit:{full_sha}"), expected_node_id)
        # Lookup via prefixed short SHA
        self.assertEqual(self.kg.find_node(f"commit:{full_sha[:7]}"), expected_node_id)

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
    def test_safe_indexing_upserts_before_deleting(self, mock_get_vs, mock_jira, mock_gh, mock_oas):
        """Verifies safe indexing sequence: add_documents occurs before pruning obsolete chunks."""
        call_order = []
        mock_vs = MagicMock()
        mock_vs.get.return_value = {"ids": ["old_chunk_1"]}
        mock_vs.add_documents.side_effect = lambda **kwargs: call_order.append("add_documents")
        mock_vs.delete.side_effect = lambda **kwargs: call_order.append("delete")
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

        load_and_index_data(repo="opendaylight/transportpce", max_commits=2, max_issues=2)
        self.assertEqual(call_order, ["add_documents", "delete"])

    @patch("knowledge_assistant.rag_pipeline.fetch_openapi_records")
    @patch("knowledge_assistant.rag_pipeline.fetch_github_records")
    @patch("knowledge_assistant.rag_pipeline.fetch_jira_records")
    @patch("knowledge_assistant.rag_pipeline.get_vector_store")
    def test_vector_failure_prevents_graph_persistence(self, mock_get_vs, mock_jira, mock_gh, mock_oas):
        """Simulates vector store insertion failure and verifies graph is not updated."""
        mock_vs = MagicMock()
        mock_vs.get.return_value = {"ids": ["old_chunk"]}
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

        # Confirm delete was never called since add_documents failed
        mock_vs.delete.assert_not_called()

    @patch("knowledge_assistant.rag_pipeline.fetch_openapi_records")
    @patch("knowledge_assistant.rag_pipeline.fetch_github_records")
    @patch("knowledge_assistant.rag_pipeline.fetch_jira_records")
    @patch("knowledge_assistant.rag_pipeline.get_vector_store")
    def test_stage2_prune_failure_aborts_pipeline(self, mock_get_vs, mock_jira, mock_gh, mock_oas):
        """Simulates Stage 2 failure (vector prune) and verifies graph/manifest are not committed as READY."""
        mock_vs = MagicMock()
        mock_vs.get.return_value = {"ids": ["obsolete_chunk"]}
        mock_vs.delete.side_effect = RuntimeError("Delete error in Chroma")
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

        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_file = os.path.join(tmp_dir, "manifest.json")
            with patch.dict(PROJECT_CONFIGS["opendaylight/transportpce"], {"manifest_file": manifest_file}):
                with self.assertRaises(RuntimeError) as ctx:
                    load_and_index_data(repo="opendaylight/transportpce", max_commits=1, max_issues=1)
                self.assertIn("Vector pruning failed at Stage 2", str(ctx.exception))

                # Verify manifest was flagged with failure status, NOT READY
                with open(manifest_file, "r") as mf:
                    m = json.load(mf)
                self.assertEqual(m["status"], "FAILED_VECTOR_PRUNING")

    @patch("knowledge_assistant.rag_pipeline.SoftwareKnowledgeGraph.save_to_file")
    @patch("knowledge_assistant.rag_pipeline.fetch_openapi_records")
    @patch("knowledge_assistant.rag_pipeline.fetch_github_records")
    @patch("knowledge_assistant.rag_pipeline.fetch_jira_records")
    @patch("knowledge_assistant.rag_pipeline.get_vector_store")
    def test_stage3_graph_persistence_failure_aborts_pipeline(self, mock_get_vs, mock_jira, mock_gh, mock_oas, mock_save):
        """Simulates Stage 3 failure (graph save) and verifies manifest is flagged as failed."""
        mock_vs = MagicMock()
        mock_vs.get.return_value = {"ids": []}
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
        mock_save.side_effect = IOError("Disk full writing knowledge graph")

        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_file = os.path.join(tmp_dir, "manifest.json")
            with patch.dict(PROJECT_CONFIGS["opendaylight/transportpce"], {"manifest_file": manifest_file}):
                with self.assertRaises(RuntimeError) as ctx:
                    load_and_index_data(repo="opendaylight/transportpce", max_commits=1, max_issues=1)
                self.assertIn("Knowledge graph persistence failed at Stage 3", str(ctx.exception))

                with open(manifest_file, "r") as mf:
                    m = json.load(mf)
                self.assertEqual(m["status"], "FAILED_GRAPH_PERSISTENCE")

    @patch("knowledge_assistant.rag_pipeline.get_vector_store")
    def test_check_index_freshness_deep_validation(self, mock_get_vs):
        """Tests index freshness check detects corrupt JSON, zero nodes, and missing manifest."""
        mock_vs = MagicMock()
        mock_vs._collection.count.return_value = 5
        mock_get_vs.return_value = mock_vs

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            manifest_file = os.path.join(tmp_dir, "test_manifest.json")
            graph_file = os.path.join(tmp_dir, "graph.json")

            m_data = {
                "project": "Test Project",
                "repo": "test/repo",
                "index_version": "2.1",
                "status": "READY"
            }
            with open(manifest_file, "w") as f:
                json.dump(m_data, f)

            with patch.dict(PROJECT_CONFIGS["dhis2/dhis2-core"], {
                "chroma_dir": tmp_dir,
                "graph_file": graph_file,
                "manifest_file": manifest_file
            }):
                # 1. Missing graph file
                freshness = check_index_freshness("dhis2/dhis2-core")
                self.assertFalse(freshness["is_fresh"])
                self.assertIn("Knowledge Graph JSON missing", freshness["issues"])

                # 2. Corrupt / empty JSON graph file
                with open(graph_file, "w") as f:
                    f.write("NOT_VALID_JSON{{{")
                freshness_corrupt = check_index_freshness("dhis2/dhis2-core")
                self.assertFalse(freshness_corrupt["is_fresh"])
                self.assertTrue(any("corrupted" in iss.lower() for iss in freshness_corrupt["issues"]))

                # 3. Valid JSON with zero nodes
                with open(graph_file, "w") as f:
                    json.dump({"nodes": []}, f)
                freshness_zero = check_index_freshness("dhis2/dhis2-core")
                self.assertFalse(freshness_zero["is_fresh"])
                self.assertIn("Knowledge Graph has zero nodes", freshness_zero["issues"])

                # 4. Valid graph with nodes and valid vector store
                with open(graph_file, "w") as f:
                    json.dump({"nodes": [{"id": "node-1"}]}, f)

                freshness_valid = check_index_freshness("dhis2/dhis2-core")
                self.assertTrue(freshness_valid["is_fresh"])
                self.assertEqual(freshness_valid["issues"], [])
