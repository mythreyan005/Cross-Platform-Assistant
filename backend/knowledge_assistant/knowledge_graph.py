"""
Software Engineering Knowledge Graph (NetworkX MultiDiGraph).
Provides genuine graph construction, persistence, incremental updates,
and bounded relationship traversal across GitHub, Jira, and OpenAPI data.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
import networkx as nx

try:
    from .models_data import SourceRecord, SourceType, NodeType, RelationshipType
except ImportError:
    from models_data import SourceRecord, SourceType, NodeType, RelationshipType

logger = logging.getLogger(__name__)


class SoftwareKnowledgeGraph:
    """
    Genuine Knowledge Graph representing software engineering artifacts and their relationships.
    Built on NetworkX with project-level isolation and JSON persistence.
    """

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.graph = nx.MultiDiGraph(project_id=project_id)

    @staticmethod
    def _project_node_id(project_id: str) -> str:
        return f"project:{project_id}"

    @staticmethod
    def _commit_node_id(sha: str) -> str:
        return f"commit:{sha[:7]}"

    @staticmethod
    def _issue_node_id(key: str) -> str:
        return f"issue:{key.upper()}"

    @staticmethod
    def _endpoint_node_id(source_id: str) -> str:
        # source_id format: "GET /api/users" or "/api/users"
        parts = source_id.strip().split()
        if len(parts) == 2:
            method, path = parts[0].upper(), parts[1]
            return f"endpoint:{method}:{path}"
        return f"endpoint:{source_id}"

    @staticmethod
    def _component_node_id(name: str) -> str:
        return f"component:{name.lower()}"

    def ensure_project_node(self, project_name: Optional[str] = None) -> str:
        """Ensures the central Project node exists."""
        node_id = self._project_node_id(self.project_id)
        if not self.graph.has_node(node_id):
            self.graph.add_node(
                node_id,
                node_type=NodeType.PROJECT.value,
                label=project_name or self.project_id,
                project_id=self.project_id
            )
        return node_id

    def add_commit_record(self, record: SourceRecord) -> str:
        """Adds a GitHub commit node and connects it to the project and any linked Jira issues."""
        proj_id = self.ensure_project_node()
        node_id = self._commit_node_id(record.source_id)

        self.graph.add_node(
            node_id,
            node_type=NodeType.COMMIT.value,
            label=f"Commit {record.source_id}",
            sha=record.metadata.get("sha", record.source_id),
            short_sha=record.source_id,
            author=record.author or "Unknown",
            title=record.title,
            message=record.content,
            timestamp=record.timestamp,
            url=record.url,
            project_id=self.project_id
        )

        # Commit BELONGS_TO Project
        self.add_edge_if_not_exists(
            node_id, proj_id,
            relationship=RelationshipType.BELONGS_TO.value
        )

        # Process relationships extracted from commit (e.g. Jira issue keys)
        for rel in record.relationships:
            target_type = rel.get("target_type")
            target_id = rel.get("target_id")
            rel_kind = rel.get("relationship", RelationshipType.REFERENCES_ISSUE.value)

            if target_type == "jira" and target_id:
                issue_node_id = self._issue_node_id(target_id)
                if not self.graph.has_node(issue_node_id):
                    # Create placeholder issue node until Jira data arrives
                    self.graph.add_node(
                        issue_node_id,
                        node_type=NodeType.ISSUE.value,
                        label=f"Issue {target_id.upper()}",
                        key=target_id.upper(),
                        project_id=self.project_id,
                        status="Referenced"
                    )
                    self.add_edge_if_not_exists(issue_node_id, proj_id, RelationshipType.BELONGS_TO.value)

                # Commit REFERENCES_ISSUE Issue
                self.add_edge_if_not_exists(node_id, issue_node_id, relationship=rel_kind)

            elif target_type == "component" and target_id:
                comp_node_id = self._component_node_id(target_id)
                if not self.graph.has_node(comp_node_id):
                    self.graph.add_node(
                        comp_node_id,
                        node_type=NodeType.COMPONENT.value,
                        label=f"Component {target_id}",
                        name=target_id,
                        project_id=self.project_id
                    )
                self.add_edge_if_not_exists(node_id, comp_node_id, relationship=RelationshipType.MODIFIES_COMPONENT.value)

        return node_id

    def add_issue_record(self, record: SourceRecord) -> str:
        """Adds a Jira issue node and connects it to the project and linked components/issues."""
        proj_id = self.ensure_project_node()
        node_id = self._issue_node_id(record.source_id)

        # Update or add issue node
        self.graph.add_node(
            node_id,
            node_type=NodeType.ISSUE.value,
            label=f"Issue {record.source_id}",
            key=record.source_id,
            summary=record.metadata.get("summary", record.title),
            status=record.status or "Open",
            priority=record.metadata.get("priority", "Normal"),
            issue_type=record.metadata.get("issue_type", "Issue"),
            description=record.content,
            url=record.url,
            project_id=self.project_id
        )

        # Issue BELONGS_TO Project
        self.add_edge_if_not_exists(node_id, proj_id, relationship=RelationshipType.BELONGS_TO.value)

        # Process relationships
        for rel in record.relationships:
            target_type = rel.get("target_type")
            target_id = rel.get("target_id")

            if target_type == "component" and target_id:
                comp_node_id = self._component_node_id(target_id)
                if not self.graph.has_node(comp_node_id):
                    self.graph.add_node(
                        comp_node_id,
                        node_type=NodeType.COMPONENT.value,
                        label=f"Component {target_id}",
                        name=target_id,
                        project_id=self.project_id
                    )
                self.add_edge_if_not_exists(node_id, comp_node_id, relationship=RelationshipType.RELATED_TO.value)

            elif target_type == "jira" and target_id:
                other_issue_id = self._issue_node_id(target_id)
                if not self.graph.has_node(other_issue_id):
                    self.graph.add_node(
                        other_issue_id,
                        node_type=NodeType.ISSUE.value,
                        label=f"Issue {target_id}",
                        key=target_id,
                        project_id=self.project_id
                    )
                self.add_edge_if_not_exists(node_id, other_issue_id, relationship=RelationshipType.RELATED_TO.value)

        return node_id

    def add_endpoint_record(self, record: SourceRecord) -> str:
        """Adds an OpenAPI endpoint node and connects it to the project and tags/components."""
        proj_id = self.ensure_project_node()
        node_id = self._endpoint_node_id(record.source_id)

        method = record.metadata.get("method", "GET")
        path = record.metadata.get("path", record.source_id)
        summary = record.metadata.get("summary", record.title)

        self.graph.add_node(
            node_id,
            node_type=NodeType.ENDPOINT.value,
            label=f"{method} {path}",
            method=method,
            path=path,
            summary=summary,
            tags=record.metadata.get("tags", []),
            description=record.content,
            url=record.url,
            project_id=self.project_id
        )

        # Project EXPOSES_ENDPOINT Endpoint
        self.add_edge_if_not_exists(proj_id, node_id, relationship=RelationshipType.EXPOSES_ENDPOINT.value)

        # Connect tags to components
        for tag in record.metadata.get("tags", []):
            comp_node_id = self._component_node_id(tag)
            if not self.graph.has_node(comp_node_id):
                self.graph.add_node(
                    comp_node_id,
                    node_type=NodeType.COMPONENT.value,
                    label=f"Component {tag}",
                    name=tag,
                    project_id=self.project_id
                )
            self.add_edge_if_not_exists(node_id, comp_node_id, relationship=RelationshipType.RELATED_TO.value)

        return node_id

    def add_edge_if_not_exists(
        self,
        u: str,
        v: str,
        relationship: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """Adds a directed edge between u and v if it does not already exist with the same relationship."""
        if self.graph.has_edge(u, v):
            for edge_data in self.graph.get_edge_data(u, v).values():
                if edge_data.get("relationship") == relationship:
                    return  # Already exists

        props = properties or {}
        props["relationship"] = relationship
        self.graph.add_edge(u, v, **props)

    def find_node(self, identifier: str) -> Optional[str]:
        """
        Fuzzy or exact lookup for node ID given a keyword, SHA, issue key, or path.
        """
        clean_id = identifier.strip()

        # Try exact node IDs
        for prefix in (self._issue_node_id, self._commit_node_id, self._endpoint_node_id, self._component_node_id):
            cand = prefix(clean_id)
            if self.graph.has_node(cand):
                return cand

        # Try case-insensitive attribute match
        clean_lower = clean_id.lower()
        for n, data in self.graph.nodes(data=True):
            if data.get("key", "").lower() == clean_lower:
                return n
            if data.get("short_sha", "").lower() == clean_lower:
                return n
            if data.get("name", "").lower() == clean_lower:
                return n
            if data.get("path", "").lower() == clean_lower:
                return n

        return None

    def get_related_entities(
        self,
        node_id: str,
        max_depth: int = 2,
        max_nodes: int = 25
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Bounded BFS traversal to discover connected nodes and edges starting from node_id.
        Returns (nodes_list, edges_list).
        """
        if not self.graph.has_node(node_id):
            return [], []

        visited_nodes: Set[str] = {node_id}
        discovered_edges: List[Dict[str, Any]] = []
        queue: List[Tuple[str, int]] = [(node_id, 0)]

        while queue and len(visited_nodes) < max_nodes:
            curr, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            # Check outbound and inbound edges
            for nbr in self.graph.successors(curr):
                for k, edata in self.graph.get_edge_data(curr, nbr).items():
                    rel = edata.get("relationship", "RELATED_TO")
                    discovered_edges.append({
                        "source": curr,
                        "target": nbr,
                        "relationship": rel
                    })
                if nbr not in visited_nodes:
                    visited_nodes.add(nbr)
                    queue.append((nbr, depth + 1))

            for nbr in self.graph.predecessors(curr):
                for k, edata in self.graph.get_edge_data(nbr, curr).items():
                    rel = edata.get("relationship", "RELATED_TO")
                    discovered_edges.append({
                        "source": nbr,
                        "target": curr,
                        "relationship": rel
                    })
                if nbr not in visited_nodes:
                    visited_nodes.add(nbr)
                    queue.append((nbr, depth + 1))

        # Format nodes
        nodes_result = []
        for n in visited_nodes:
            ndata = self.graph.nodes[n]
            nodes_result.append({
                "id": n,
                "type": ndata.get("node_type", "unknown"),
                "label": ndata.get("label", n),
                "properties": {k: v for k, v in ndata.items() if k not in ("message", "description")}
            })

        return nodes_result, discovered_edges

    def format_graph_context(self, seed_entity_ids: List[str], max_depth: int = 2) -> str:
        """
        Extracts subgraphs for seed entities and renders human-readable factual relational statements.
        """
        all_nodes: Dict[str, Dict[str, Any]] = {}
        all_edges: Set[Tuple[str, str, str]] = set()

        for seed in seed_entity_ids:
            actual_node = self.find_node(seed)
            if not actual_node:
                continue

            nodes, edges = self.get_related_entities(actual_node, max_depth=max_depth)
            for n in nodes:
                all_nodes[n["id"]] = n
            for e in edges:
                all_edges.add((e["source"], e["target"], e["relationship"]))

        if not all_edges and not all_nodes:
            return ""

        lines = ["=== KNOWLEDGE GRAPH RELATIONSHIPS (GROUND TRUTH) ==="]

        # Format explicit edges as factual statements
        for src, tgt, rel in sorted(list(all_edges)):
            src_node = all_nodes.get(src, self.graph.nodes.get(src, {}))
            tgt_node = all_nodes.get(tgt, self.graph.nodes.get(tgt, {}))

            src_label = src_node.get("label", src) if isinstance(src_node, dict) else src
            tgt_label = tgt_node.get("label", tgt) if isinstance(tgt_node, dict) else tgt

            # Extra metadata context
            extra = []
            if isinstance(src_node, dict):
                props = src_node.get("properties", src_node)
                if props.get("author"):
                    extra.append(f"Author: {props['author']}")
                if props.get("status"):
                    extra.append(f"Status: {props['status']}")
            extra_str = f" ({', '.join(extra)})" if extra else ""

            lines.append(f"• [{rel}] {src_label}{extra_str}  -->  {tgt_label}")

        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """Returns node and edge summary statistics."""
        node_counts: Dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            ntype = data.get("node_type", "other")
            node_counts[ntype] = node_counts.get(ntype, 0) + 1

        edge_counts: Dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            rel = data.get("relationship", "other")
            edge_counts[rel] = edge_counts.get(rel, 0) + 1

        return {
            "project_id": self.project_id,
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "nodes_by_type": node_counts,
            "edges_by_relationship": edge_counts
        }

    def save_to_file(self, filepath: str) -> bool:
        """Serializes the graph to JSON format on disk."""
        try:
            data = {
                "project_id": self.project_id,
                "nodes": [
                    {"id": n, **attrs} for n, attrs in self.graph.nodes(data=True)
                ],
                "edges": [
                    {"source": u, "target": v, **attrs} for u, v, attrs in self.graph.edges(data=True)
                ]
            }
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved Knowledge Graph ({self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges) to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save Knowledge Graph to {filepath}: {e}")
            return False

    def load_from_file(self, filepath: str) -> bool:
        """Loads nodes and edges from a persisted JSON file."""
        if not os.path.exists(filepath):
            return False

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.graph.clear()
            for node in data.get("nodes", []):
                nid = node.pop("id")
                self.graph.add_node(nid, **node)

            for edge in data.get("edges", []):
                u = edge.pop("source")
                v = edge.pop("target")
                self.graph.add_edge(u, v, **edge)

            logger.info(f"Loaded Knowledge Graph from {filepath}: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")
            return True
        except Exception as e:
            logger.error(f"Failed to load Knowledge Graph from {filepath}: {e}")
            return False
