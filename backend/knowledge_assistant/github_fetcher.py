"""
GitHub Data Fetcher for Cross-Platform Knowledge Assistant.
Extracts commit histories, file changes, and references to Jira issues.
"""

import os
import re
import logging
from typing import List, Optional, Dict, Any
from github import Github
from github.GithubException import RateLimitExceededException, BadCredentialsException, GithubException

try:
    from .models_data import SourceRecord, SourceType
except ImportError:
    from models_data import SourceRecord, SourceType

logger = logging.getLogger(__name__)

# Regex pattern for Jira issue keys (e.g., DHIS2-12345, TRNSPRTPCE-980)
JIRA_REGEX = re.compile(r'\b[A-Z][A-Z0-9]+-[0-9]+\b')


def extract_jira_keys(message: str) -> List[str]:
    """Finds all unique Jira keys in a commit message or text."""
    if not message:
        return []
    return sorted(list(set(JIRA_REGEX.findall(message))))


def fetch_github_records(
    repo_name: str,
    github_token: Optional[str] = None,
    max_commits: int = 25
) -> List[SourceRecord]:
    """
    Fetches recent commits from GitHub and normalizes them into unified SourceRecord objects.
    """
    if github_token is None:
        github_token = os.getenv("GITHUB_TOKEN")

    records: List[SourceRecord] = []

    try:
        if github_token:
            g = Github(github_token, retry=2, timeout=15)
        else:
            g = Github(retry=1, timeout=15)

        logger.info(f"Fetching up to {max_commits} commits from GitHub repository: {repo_name}...")
        repo = g.get_repo(repo_name)
        commits = repo.get_commits()

        count = 0
        for commit in commits:
            if count >= max_commits:
                break

            sha = commit.sha
            commit_data = commit.commit
            message = commit_data.message or ""
            author_name = commit_data.author.name if commit_data.author else "Unknown"
            commit_date = commit_data.author.date.isoformat() if (commit_data.author and commit_data.author.date) else None
            html_url = commit.html_url or f"https://github.com/{repo_name}/commit/{sha}"

            # Extract modified file paths and stats where available
            modified_files: List[str] = []
            try:
                if commit.files:
                    modified_files = [f.filename for f in commit.files[:15] if f.filename]
            except Exception:
                pass

            # Extract referenced Jira keys
            jira_keys = extract_jira_keys(message)

            # Build relationships list
            relationships = []
            for jk in jira_keys:
                relationships.append({
                    "target_type": "jira",
                    "target_id": jk,
                    "relationship": "REFERENCES_ISSUE"
                })

            # Derive components or modules from file paths
            components = set()
            for fp in modified_files:
                parts = fp.split('/')
                if len(parts) > 1:
                    components.add(parts[0])  # top-level directory as component
            for comp in components:
                relationships.append({
                    "target_type": "component",
                    "target_id": comp,
                    "relationship": "MODIFIES_COMPONENT"
                })

            summary_title = message.split('\n')[0][:120]

            record = SourceRecord(
                source_type=SourceType.GITHUB,
                source_id=sha,
                project_id=repo_name,
                title=f"Commit {sha[:7]}: {summary_title}",
                content=(
                    f"GitHub Commit {sha[:7]} by {author_name} ({commit_date or 'No date'})\n"
                    f"Repository: {repo_name}\n"
                    f"Message: {message}\n"
                    f"Linked Jira Issues: {', '.join(jira_keys) if jira_keys else 'None'}\n"
                    f"Modified Files: {', '.join(modified_files[:10]) if modified_files else 'None'}"
                ),
                url=html_url,
                author=author_name,
                timestamp=commit_date,
                metadata={
                    "sha": sha,
                    "short_sha": sha[:7],
                    "repo_name": repo_name,
                    "jira_keys": jira_keys,
                    "modified_files": modified_files,
                    "components": list(components),
                },
                relationships=relationships
            )
            records.append(record)
            count += 1

        logger.info(f"Successfully retrieved {len(records)} GitHub commits for {repo_name}.")

    except RateLimitExceededException:
        logger.warning(f"GitHub API rate limit exceeded for {repo_name}. Using cached/fallback data.")
    except BadCredentialsException:
        logger.error("Invalid GitHub token provided. Please check GITHUB_TOKEN in your .env.")
    except Exception as e:
        logger.warning(f"Error fetching data from GitHub for {repo_name}: {e}")

    return records


def fetch_github_commits(repo_name: str, github_token: Optional[str] = None, max_commits: int = 20) -> List[Dict[str, Any]]:
    """
    Legacy compatibility wrapper returning dictionary items.
    """
    records = fetch_github_records(repo_name, github_token, max_commits)
    legacy_list = []
    for r in records:
        legacy_list.append({
            'sha': r.metadata.get('short_sha', r.source_id),
            'author': r.author,
            'message': r.metadata.get('sha', ''),
            'linked_jira_keys': r.metadata.get('jira_keys', []),
            'url': r.url,
            'timestamp': r.timestamp,
        })
    return legacy_list


if __name__ == "__main__":
    test_repo = "opendaylight/transportpce"
    print(f"Testing GitHub fetcher for {test_repo}...")
    sample_records = fetch_github_records(test_repo, max_commits=5)
    print(f"Fetched {len(sample_records)} records.")
    for rec in sample_records:
        print(f"- {rec.source_id} | {rec.title} | Jira: {rec.metadata.get('jira_keys')}")
