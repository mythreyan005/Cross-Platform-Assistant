"""
Unit tests for GitHub, Jira, and OpenAPI fetchers with mocked external services.
"""

from unittest.mock import patch, MagicMock
from django.test import TestCase
from knowledge_assistant.github_fetcher import extract_jira_keys, fetch_github_records
from knowledge_assistant.jira_fetcher import parse_adf_content, fetch_jira_records
from knowledge_assistant.swagger_fetcher import parse_openapi_spec


class FetcherTests(TestCase):
    def test_extract_jira_keys_regex(self):
        text = "feat(auth): add MFA support for DHIS2-20751 and TRNSPRTPCE-980; see also DHIS2-20751"
        keys = extract_jira_keys(text)
        self.assertEqual(keys, ["DHIS2-20751", "TRNSPRTPCE-980"])

    def test_parse_adf_content(self):
        adf = {
            "type": "doc",
            "content": [
                {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": "This is a Jira bug description."}]
                },
                {
                    "type": "codeBlock",
                    "content": [{"type": "text", "text": "def test(): pass"}]
                }
            ]
        }
        parsed = parse_adf_content(adf)
        self.assertIn("This is a Jira bug description.", parsed)
        self.assertIn("```def test(): pass```", parsed)

    def test_parse_openapi_spec(self):
        sample_spec = {
            "openapi": "3.0.0",
            "info": {"title": "Sample API"},
            "paths": {
                "/api/test": {
                    "get": {
                        "summary": "Test GET endpoint",
                        "description": "Returns test response",
                        "tags": ["Testing"],
                        "parameters": [{"name": "id", "in": "query"}],
                        "responses": {"200": {"description": "OK"}}
                    }
                }
            }
        }
        records = parse_openapi_spec(sample_spec, project_id="sample-proj")
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec.source_id, "GET /api/test")
        self.assertEqual(rec.metadata["method"], "GET")
        self.assertIn("Testing", rec.metadata["tags"])
        self.assertEqual(rec.relationships[0]["target_id"], "Testing")

    def test_parse_openapi_yaml(self):
        yaml_content = """
openapi: 3.0.0
info:
  title: Sample YAML API
  version: 1.0.0
paths:
  /api/v1/status:
    get:
      summary: Get system health
      description: Returns health status
      tags:
        - Health
      responses:
        '200':
          description: OK
"""
        from knowledge_assistant.swagger_fetcher import load_spec_from_string_or_bytes
        spec_dict = load_spec_from_string_or_bytes(yaml_content)
        self.assertIsNotNone(spec_dict)
        self.assertEqual(spec_dict["info"]["title"], "Sample YAML API")
        records = parse_openapi_spec(spec_dict, project_id="yaml-proj")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].source_id, "GET /api/v1/status")
        self.assertIn("Health", records[0].metadata["tags"])

