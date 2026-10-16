"""
Unified Data Model for Multi-Source Ingestion and Knowledge Graph Representation.
Provides consistent interfaces for GitHub commits, Jira issues, and OpenAPI endpoints.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


class SourceType(str, Enum):
    GITHUB = "github"
    JIRA = "jira"
    OPENAPI = "openapi"


class RelationshipType(str, Enum):
    BELONGS_TO = "BELONGS_TO"
    REFERENCES_ISSUE = "REFERENCES_ISSUE"
    MODIFIES_COMPONENT = "MODIFIES_COMPONENT"
    EXPOSES_ENDPOINT = "EXPOSES_ENDPOINT"
    RELATED_TO = "RELATED_TO"


class NodeType(str, Enum):
    PROJECT = "project"
    COMMIT = "commit"
    ISSUE = "issue"
    ENDPOINT = "endpoint"
    COMPONENT = "component"


@dataclass
class SourceRecord:
    """
    Normalized representation of any data item ingested from external systems.
    """
    source_type: SourceType
    source_id: str                      # Commit SHA, Jira issue key, Endpoint path
    project_id: str                     # e.g., 'dhis2/dhis2-core'
    title: str                          # Short summary or commit title
    content: str                        # Full descriptive text
    url: Optional[str] = None           # Direct link to GitHub, Jira, or docs
    author: Optional[str] = None        # Author or reporter
    status: Optional[str] = None        # Jira status or endpoint state
    timestamp: Optional[str] = None     # Creation or commit date (ISO string)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # References to other entities (e.g. Jira keys mentioned in commits)
    relationships: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type.value if isinstance(self.source_type, SourceType) else self.source_type,
            "source_id": self.source_id,
            "project_id": self.project_id,
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "author": self.author,
            "status": self.status,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "relationships": self.relationships,
        }


@dataclass
class GraphNode:
    """Represents a node in the Software Engineering Knowledge Graph."""
    id: str
    node_type: NodeType
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """Represents a directed relationship in the Knowledge Graph."""
    source_id: str
    target_id: str
    relationship: RelationshipType
    properties: Dict[str, Any] = field(default_factory=dict)
