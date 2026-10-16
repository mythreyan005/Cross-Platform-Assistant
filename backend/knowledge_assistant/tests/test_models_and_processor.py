"""
Unit tests for models, document preprocessing, and chunking.
"""

from django.test import TestCase
from knowledge_assistant.models_data import SourceRecord, SourceType, NodeType, RelationshipType
from knowledge_assistant.document_processor import DocumentProcessor


class DataModelAndProcessorTests(TestCase):
    def setUp(self):
        self.processor = DocumentProcessor(chunk_size=200, chunk_overlap=40)

    def test_source_record_creation_and_dict(self):
        record = SourceRecord(
            source_type=SourceType.GITHUB,
            source_id="abc1234",
            project_id="test/repo",
            title="Fix bug in auth module",
            content="Detailed commit message fixing issue DHIS2-101",
            author="Dev",
            relationships=[{"target_type": "jira", "target_id": "DHIS2-101", "relationship": "REFERENCES_ISSUE"}]
        )
        data = record.to_dict()
        self.assertEqual(data["source_type"], "github")
        self.assertEqual(data["source_id"], "abc1234")
        self.assertEqual(len(data["relationships"]), 1)
        self.assertEqual(data["relationships"][0]["target_id"], "DHIS2-101")

    def test_clean_text_preserves_identifiers(self):
        raw = "Commit  hash   90a4ce9f   fixes issue   TRNSPRTPCE-980   at endpoint   /api/v1/network\r\n\r\nNext line."
        cleaned = DocumentProcessor.clean_text(raw)
        self.assertIn("90a4ce9f", cleaned)
        self.assertIn("TRNSPRTPCE-980", cleaned)
        self.assertIn("/api/v1/network", cleaned)
        self.assertNotIn("   ", cleaned)  # extra spaces collapsed

    def test_chunking_and_deterministic_ids(self):
        record = SourceRecord(
            source_type=SourceType.JIRA,
            source_id="DHIS2-500",
            project_id="dhis2/dhis2-core",
            title="Analytics dimension aggregation",
            content="A " * 300  # Long content that will chunk
        )
        chunks = self.processor.process_record(record)
        self.assertGreater(len(chunks), 1)

        # Verify deterministic ID format
        for idx, (chunk_id, doc) in enumerate(chunks):
            self.assertIn("dhis2_dhis2-core:jira:DHIS2-500:", chunk_id)
            self.assertEqual(doc.metadata["chunk_index"], idx)
            self.assertEqual(doc.metadata["source_id"], "DHIS2-500")
            self.assertEqual(doc.metadata["source_type"], "jira")
