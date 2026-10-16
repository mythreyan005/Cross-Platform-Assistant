"""
Jira Data Fetcher for Cross-Platform Knowledge Assistant.
Extracts Jira issues, descriptions, statuses, and linked issues from Atlassian instances.
"""

import os
import logging
from typing import List, Optional, Dict, Any
import requests
from requests.auth import HTTPBasicAuth

try:
    from .models_data import SourceRecord, SourceType
except ImportError:
    from models_data import SourceRecord, SourceType

logger = logging.getLogger(__name__)


def parse_adf_content(content_obj: Any) -> str:
    """
    Recursively extracts human-readable text from Atlassian Document Format (ADF) JSON.
    """
    if isinstance(content_obj, str):
        return content_obj

    if not isinstance(content_obj, dict):
        return ""

    text_parts = []
    node_type = content_obj.get("type", "")

    if node_type == "text":
        return content_obj.get("text", "")

    if "content" in content_obj and isinstance(content_obj["content"], list):
        for child in content_obj["content"]:
            child_text = parse_adf_content(child)
            if child_text:
                text_parts.append(child_text)

    joined = " ".join(text_parts).strip()
    if node_type in ("paragraph", "heading"):
        return f"\n{joined}\n"
    if node_type == "bulletList":
        return f"\n{joined}\n"
    if node_type == "codeBlock":
        return f"\n```{joined}```\n"

    return joined


def fetch_jira_records(
    base_url: str,
    project_key: str,
    email: Optional[str] = None,
    api_token: Optional[str] = None,
    max_results: int = 25
) -> List[SourceRecord]:
    """
    Fetches Jira issues and converts them into unified SourceRecord objects.
    """
    if not email:
        email = os.getenv("JIRA_EMAIL")
    if not api_token:
        api_token = os.getenv("JIRA_API_TOKEN")

    records: List[SourceRecord] = []
    search_url = f"{base_url.rstrip('/')}/rest/api/3/search/jql"

    # Setup Basic Auth only when both email and token are provided
    auth = None
    if email and api_token:
        auth = HTTPBasicAuth(email, api_token)

    page_size = min(max_results, 50)
    start_at = 0
    total_fetched = 0

    while total_fetched < max_results:
        params = {
            'jql': f"project={project_key} ORDER BY updated DESC",
            'startAt': start_at,
            'maxResults': page_size,
            'fields': 'summary,description,status,issuetype,priority,issuelinks,components,resolution'
        }

        try:
            logger.info(f"Fetching Jira issues from {base_url} (project: {project_key}, startAt: {start_at})...")
            response = requests.get(search_url, params=params, auth=auth, timeout=12)

            if response.status_code in (401, 403):
                logger.warning(f"Jira unauthorized for {project_key}. Ensure JIRA_EMAIL and JIRA_API_TOKEN are set.")
                break

            response.raise_for_status()
            data = response.json()
            issues = data.get('issues', [])
            if not issues:
                break

            for issue in issues:
                fields = issue.get('fields', {})
                key = issue.get('key', '')
                summary = fields.get('summary', 'No summary')
                status_obj = fields.get('status', {})
                status_name = status_obj.get('name', 'Unknown')
                issue_type = fields.get('issuetype', {}).get('name', 'Issue')
                priority = fields.get('priority', {}).get('name', 'Normal')

                description_obj = fields.get('description')
                description_text = parse_adf_content(description_obj).strip()

                components = [c.get('name', '') for c in fields.get('components', []) if c.get('name')]

                # Extract linked issues
                relationships = []
                for comp in components:
                    relationships.append({
                        "target_type": "component",
                        "target_id": comp,
                        "relationship": "RELATED_TO"
                    })

                for link in fields.get('issuelinks', []):
                    linked_key = None
                    if 'outwardIssue' in link:
                        linked_key = link['outwardIssue'].get('key')
                    elif 'inwardIssue' in link:
                        linked_key = link['inwardIssue'].get('key')
                    if linked_key:
                        relationships.append({
                            "target_type": "jira",
                            "target_id": linked_key,
                            "relationship": "RELATED_TO"
                        })

                issue_url = f"{base_url.rstrip('/')}/browse/{key}"

                record = SourceRecord(
                    source_type=SourceType.JIRA,
                    source_id=key,
                    project_id=project_key,
                    title=f"Jira {key} [{status_name}]: {summary}",
                    content=(
                        f"Jira Issue {key}: {summary}\n"
                        f"Type: {issue_type} | Status: {status_name} | Priority: {priority}\n"
                        f"Components: {', '.join(components) if components else 'None'}\n"
                        f"Description: {description_text if description_text else 'No description provided.'}"
                    ),
                    url=issue_url,
                    status=status_name,
                    metadata={
                        "key": key,
                        "summary": summary,
                        "status": status_name,
                        "issue_type": issue_type,
                        "priority": priority,
                        "components": components,
                    },
                    relationships=relationships
                )
                records.append(record)
                total_fetched += 1
                if total_fetched >= max_results:
                    break

            start_at += len(issues)
            if start_at >= data.get('total', 0):
                break

        except requests.exceptions.RequestException as e:
            logger.warning(f"Error fetching Jira issues from {base_url}: {e}")
            break
        except Exception as e:
            logger.warning(f"Unexpected error in Jira fetcher: {e}")
            break

    logger.info(f"Retrieved {len(records)} Jira issues for {project_key}.")
    return records


def fetch_jira_issues(
    base_url: str,
    project_key: str,
    email: Optional[str] = None,
    api_token: Optional[str] = None,
    max_results: int = 10
) -> List[Dict[str, Any]]:
    """
    Legacy compatibility wrapper returning list of dictionaries.
    """
    records = fetch_jira_records(base_url, project_key, email, api_token, max_results)
    legacy = []
    for r in records:
        legacy.append({
            'key': r.source_id,
            'summary': r.metadata.get('summary', r.title),
            'status': r.status,
            'description': r.content,
            'url': r.url
        })
    return legacy


if __name__ == "__main__":
    print("Testing Jira fetcher...")
    recs = fetch_jira_records("https://dhis2.atlassian.net", "DHIS2", max_results=5)
    print(f"Retrieved {len(recs)} issues:")
    for rc in recs:
        print(f"- {rc.source_id}: {rc.title}")
