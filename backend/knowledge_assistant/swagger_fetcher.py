"""
Swagger / OpenAPI Specification Fetcher for Cross-Platform Knowledge Assistant.
Extracts API endpoints, HTTP methods, operation summaries, tags, and documentation.
Supports both JSON and YAML OpenAPI specifications.
"""

import os
import json
import logging
from typing import List, Optional, Dict, Any
import requests

try:
    import yaml
except ImportError:
    yaml = None

try:
    from .models_data import SourceRecord, SourceType
except ImportError:
    from models_data import SourceRecord, SourceType

logger = logging.getLogger(__name__)


def parse_openapi_spec(spec_data: Dict[str, Any], project_id: str, base_url: str = "") -> List[SourceRecord]:
    """
    Parses an OpenAPI/Swagger dictionary and outputs unified SourceRecord items.
    """
    records: List[SourceRecord] = []
    if not isinstance(spec_data, dict):
        logger.warning(f"Malformed OpenAPI specification provided for {project_id}.")
        return records

    paths = spec_data.get("paths", {})
    api_title = spec_data.get("info", {}).get("title", "API")

    for path, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue

        for method, operation in path_item.items():
            method_lower = method.lower()
            if method_lower not in ("get", "post", "put", "delete", "patch", "options", "head"):
                continue

            if not isinstance(operation, dict):
                continue

            summary = operation.get("summary", "").strip() or f"{method.upper()} {path}"
            description = operation.get("description", "").strip()
            operation_id = operation.get("operationId", "")
            tags = operation.get("tags", [])

            # Extract parameter names
            params = []
            for p in operation.get("parameters", []):
                if isinstance(p, dict) and p.get("name"):
                    params.append(f"{p['name']} ({p.get('in', 'query')})")

            # Extract response codes
            responses = list(operation.get("responses", {}).keys()) if isinstance(operation.get("responses"), dict) else []

            relationships = []
            for tag in tags:
                relationships.append({
                    "target_type": "component",
                    "target_id": tag,
                    "relationship": "RELATED_TO"
                })

            endpoint_id = f"{method.upper()} {path}"
            content = (
                f"API Endpoint: {endpoint_id}\n"
                f"Specification: {api_title}\n"
                f"Summary: {summary}\n"
                f"Tags/Components: {', '.join(tags) if tags else 'General'}\n"
                f"Parameters: {', '.join(params) if params else 'None'}\n"
                f"Responses: {', '.join(responses) if responses else 'Standard'}\n"
                f"Description: {description if description else 'No detailed description provided.'}"
            )

            record = SourceRecord(
                source_type=SourceType.OPENAPI,
                source_id=endpoint_id,
                project_id=project_id,
                title=f"Endpoint [{method.upper()}]: {path} - {summary}",
                content=content,
                url=f"{base_url.rstrip('/')}{path}" if base_url else None,
                status="active",
                metadata={
                    "method": method.upper(),
                    "path": path,
                    "summary": summary,
                    "operation_id": operation_id,
                    "tags": tags,
                    "parameters": params,
                    "responses": responses,
                },
                relationships=relationships
            )
            records.append(record)

    logger.info(f"Parsed {len(records)} API endpoints for {project_id}.")
    return records


def load_spec_from_string_or_bytes(raw_content: str) -> Optional[Dict[str, Any]]:
    """Tries JSON first, then falls back to YAML."""
    try:
        return json.loads(raw_content)
    except Exception:
        if yaml is not None:
            try:
                return yaml.safe_load(raw_content)
            except Exception as ye:
                logger.warning(f"Failed to parse OpenAPI as YAML: {ye}")
    return None


def fetch_openapi_records(
    source_url_or_path: str,
    project_id: str = "default",
    base_url: str = ""
) -> List[SourceRecord]:
    """
    Loads OpenAPI / Swagger from a web URL or local JSON/YAML file and returns SourceRecords.
    """
    logger.info(f"Loading OpenAPI specification from: {source_url_or_path}...")

    # Case 1: Local file path
    if os.path.exists(source_url_or_path):
        try:
            with open(source_url_or_path, "r", encoding="utf-8") as f:
                content = f.read()
                spec_data = load_spec_from_string_or_bytes(content)
                if spec_data:
                    return parse_openapi_spec(spec_data, project_id, base_url)
                logger.error(f"Unrecognized OpenAPI format in {source_url_or_path}")
                return []
        except Exception as e:
            logger.error(f"Failed to read local OpenAPI specification file: {e}")
            return []

    # Case 2: Remote HTTP URL
    if source_url_or_path.startswith("http://") or source_url_or_path.startswith("https://"):
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            response = requests.get(source_url_or_path, headers=headers, timeout=12)
            response.raise_for_status()
            spec_data = load_spec_from_string_or_bytes(response.text)
            if spec_data:
                return parse_openapi_spec(spec_data, project_id, base_url)
            logger.warning(f"Could not parse remote OpenAPI content from {source_url_or_path}")
            return []
        except Exception as e:
            logger.warning(f"Failed to fetch remote OpenAPI JSON/YAML from {source_url_or_path}: {e}")
            return []

    logger.warning(f"OpenAPI source neither exists locally nor is a valid URL: {source_url_or_path}")
    return []


def fetch_swagger_docs(swagger_url: str) -> List[Dict[str, Any]]:
    """
    Legacy compatibility wrapper returning list of endpoint dicts.
    """
    records = fetch_openapi_records(swagger_url)
    legacy = []
    for r in records:
        legacy.append({
            'endpoint': r.source_id,
            'summary': r.metadata.get('summary', r.title),
            'description': r.content
        })
    return legacy
