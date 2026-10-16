"""
Unit tests for Software Engineering Knowledge Graph construction, persistence, and traversal.
"""

import os
import tempfile
from django.test import TestCase
from knowledge_assistant.models_data import SourceRecord, SourceType
from knowledge_assistant.knowledge_graph import SoftwareKnowledgeGraph


class KnowledgeGraphTests(TestCase):
    def setUp(self):
        self.kg = SoftwareKnowledgeGraph(project_id="test/sample-repo")

    def test_node_and_edge_creation(self):
        # 1. Add Jira issue
        issue_record = SourceRecord(
            source_type=SourceType.JIRA,
            source_id="TEST-101",
            project_id="test/sample-repo",
            title="User login failure",
            content="Bug in password auth check",
            status="Open",
            relationships=[{"target_type": "component", "target_id": "auth", "relationship": "RELATED_TO"}]
        )
        self.kg.add_issue_record(issue_record)

        # 2. Add Commit referencing that Jira issue
        commit_record = SourceRecord(
            source_type=SourceType.GITHUB,
            source_id="1a2b3c4",
            project_id="test/sample-repo",
            title="Fix password auth check",
            content="Resolved TEST-101 by fixing hash check",
            author="Alice",
            relationships=[
                {"target_type": "jira", "target_id": "TEST-101", "relationship": "REFERENCES_ISSUE"},
                {"target_type": "component", "target_id": "auth", "relationship": "MODIFIES_COMPONENT"}
            ]
        )
        self.kg.add_commit_record(commit_record)

        # 3. Add Endpoint
        endpoint_record = SourceRecord(
            source_type=SourceType.OPENAPI,
            source_id="POST /api/login",
            project_id="test/sample-repo",
            title="Login endpoint",
            content="Authenticates user",
            metadata={"method": "POST", "path": "/api/login", "tags": ["auth"]}
        )
        self.kg.add_endpoint_record(endpoint_record)

        stats = self.kg.get_stats()
        self.assertGreaterEqual(stats["total_nodes"], 4)
        self.assertGreaterEqual(stats["total_edges"], 4)

        # Check relationships breakdown
        edges = stats["edges_by_relationship"]
        self.assertIn("REFERENCES_ISSUE", edges)
        self.assertIn("BELONGS_TO", edges)
        self.assertIn("EXPOSES_ENDPOINT", edges)

    def test_bounded_graph_traversal(self):
        # Build connected chain: Commit -> Issue -> Component
        commit_rec = SourceRecord(
            source_type=SourceType.GITHUB,
            source_id="9988776",
            project_id="test/sample-repo",
            title="Fix analytics",
            content="Relates to DHIS2-99",
            relationships=[{"target_type": "jira", "target_id": "DHIS2-99", "relationship": "REFERENCES_ISSUE"}]
        )
        self.kg.add_commit_record(commit_rec)

        issue_rec = SourceRecord(
            source_type=SourceType.JIRA,
            source_id="DHIS2-99",
            project_id="test/sample-repo",
            title="Analytics bug",
            content="Issue details",
            relationships=[{"target_type": "component", "target_id": "analytics", "relationship": "RELATED_TO"}]
        )
        self.kg.add_issue_record(issue_rec)

        # Start traversal from commit
        commit_node = self.kg._commit_node_id("9988776")
        nodes, edges = self.kg.get_related_entities(commit_node, max_depth=2)

        node_ids = [n["id"] for n in nodes]
        self.assertIn(commit_node, node_ids)
        self.assertIn("issue:DHIS2-99", node_ids)

        context_str = self.kg.format_graph_context(["9988776"], max_depth=2)
        self.assertIn("REFERENCES_ISSUE", context_str)
        self.assertIn("DHIS2-99", context_str)

    def test_graph_persistence_and_reload(self):
        issue_record = SourceRecord(
            source_type=SourceType.JIRA,
            source_id="PERSIST-1",
            project_id="test/sample-repo",
            title="Persistence verification",
            content="Content",
            status="Closed"
        )
        self.kg.add_issue_record(issue_record)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self.kg.save_to_file(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))

            new_kg = SoftwareKnowledgeGraph(project_id="test/sample-repo")
            success = new_kg.load_from_file(tmp_path)
            self.assertTrue(success)
            self.assertEqual(new_kg.graph.number_of_nodes(), self.kg.graph.number_of_nodes())
            self.assertTrue(new_kg.graph.has_node("issue:PERSIST-1"))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_edge_provenance_preservation(self):
        commit_rec = SourceRecord(
            source_type=SourceType.GITHUB,
            source_id="abc9988",
            project_id="test/sample-repo",
            title="Fix networking bug",
            content="Linked to TRNSPRTPCE-10",
            relationships=[{
                "target_type": "jira",
                "target_id": "TRNSPRTPCE-10",
                "relationship": "REFERENCES_ISSUE",
                "provenance": "explicit_git_commit"
            }]
        )
        self.kg.add_commit_record(commit_rec)
        node_id = self.kg._commit_node_id("abc9988")
        nodes, edges = self.kg.get_related_entities(node_id, max_depth=1)
        self.assertTrue(any(e.get("provenance") == "explicit_git_commit" for e in edges))

    def test_canonical_sha_handling(self):
        full_sha = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
        short_sha = "a1b2c3d"
        commit_rec = SourceRecord(
            source_type=SourceType.GITHUB,
            source_id=full_sha,
            project_id="test/sample-repo",
            title="Refactor core module",
            content="Cleanup code",
        )
        self.kg.add_commit_record(commit_rec)

        # Both full sha and short sha should be located by find_node
        node_from_full = self.kg.find_node(full_sha)
        node_from_short = self.kg.find_node(short_sha)
        self.assertIsNotNone(node_from_full)
        self.assertEqual(node_from_full, node_from_short)
