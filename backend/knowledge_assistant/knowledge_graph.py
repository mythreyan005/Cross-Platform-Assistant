"""
Software Engineering Knowledge Graph (NetworkX MultiDiGraph).
Provides canonical node representation, provenance tracking on all edges,
persistence, incremental updates, and bounded relationship traversal across
GitHub commits, Jira issues, and OpenAPI specifications.
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
    Built on NetworkX with project-level isolation, provenance tracking, and JSON persistence.
    """

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.graph = nx.MultiDiGraph(project_id=project_id)

    @staticmethod
    def _project_node_id(project_id: str) -> str:
        return f"project:{project_id}"

    @staticmethod
    def _commit_node_id(sha: str) -> str:
        # Use full canonical SHA if >= 7 chars, lowercased to prevent collisions
        clean_sha = sha.strip().lower()
        return f"commit:{clean_sha}"

    @staticmethod
    def _issue_node_id(key: str) -> str:
        return f"issue:{key.strip().upper()}"

    @staticmethod
    def _endpoint_node_id(source_id: str) -> str:
        parts = source_id.strip().split()
        if len(parts) == 2:
            method, path = parts[0].upper(), parts[1]
            return f"endpoint:{method}:{path}"
        return f"endpoint:{source_id.strip()}"

    @staticmethod
    def _component_node_id(name: str) -> str:
        return f"component:{name.strip().lower()}"

    def ensure_project_node(self, project_name: Optional[str] = None) -> str:
        """Ensures the central Project node exists."""
        node_id = self._project_node_id(self.project_id)
        if not self.graph.has_node(node_id):
            self.graph.add_node(
                node_id,
                node_type=NodeType.PROJECT.value,
                label=project_name or self.project_id,
                project_id=self.project_id,
                source_type="project_meta",
                provenance="repository_definition"
            )
        return node_id

    def add_commit_record(self, record: SourceRecord) -> str:
        """Adds a GitHub commit node and connects it to the project, modified components, and linked Jira issues."""
        proj_id = self.ensure_project_node()
        sha = record.metadata.get("sha", record.source_id).lower()
        node_id = self._commit_node_id(sha)

        self.graph.add_node(
            node_id,
            node_type=NodeType.COMMIT.value,
            label=f"Commit {sha[:7]}",
            sha=sha,
            short_sha=sha[:7],
            author=record.author or "Unknown",
            title=record.title,
            message=record.content,
            timestamp=record.timestamp,
            url=record.url,
            project_id=self.project_id,
            source_type=SourceType.GITHUB.value,
            provenance="github_api_commit"
        )

        # Commit BELONGS_TO Project
        self.add_edge_if_not_exists(
            node_id, proj_id,
            relationship=RelationshipType.BELONGS_TO.value,
            properties={
                "provenance": "git_repository_origin",
                "source_type": SourceType.GITHUB.value,
                "timestamp": record.timestamp or ""
            }
        )

        # Process relationships extracted from commit
        for rel in record.relationships:
            target_type = rel.get("target_type")
            target_id = rel.get("target_id")
            rel_kind = rel.get("relationship", RelationshipType.REFERENCES_ISSUE.value)

            if target_type == "jira" and target_id:
                issue_node_id = self._issue_node_id(target_id)
                if not self.graph.has_node(issue_node_id):
                    # Placeholder node for referenced issue not yet independently ingested
                    self.graph.add_node(
                        issue_node_id,
                        node_type=NodeType.ISSUE.value,
                        label=f"Issue {target_id.upper()}",
                        key=target_id.upper(),
                        project_id=self.project_id,
                        status="ReferencedInCommit",
                        source_type=SourceType.JIRA.value,
                        provenance="commit_message_reference"
                    )
                    self.add_edge_if_not_exists(
                        issue_node_id, proj_id,
                        relationship=RelationshipType.BELONGS_TO.value,
                        properties={"provenance": "derived_jira_project"}
                    )

                # Commit REFERENCES_ISSUE Issue (provenance: regex extraction from commit message)
                self.add_edge_if_not_exists(
                    node_id, issue_node_id,
                    relationship=rel_kind,
                    properties={
                        "provenance": rel.get("provenance", "commit_message_regex"),
                        "regex_pattern": r"\b[A-Z][A-Z0-9]+-[0-9]+\b",
                        "commit_sha": sha[:7],
                        "issue_key": target_id.upper(),
                        "timestamp": record.timestamp or ""
                    }
                )

            elif target_type == "component" and target_id:
                comp_node_id = self._component_node_id(target_id)
                if not self.graph.has_node(comp_node_id):
                    self.graph.add_node(
                        comp_node_id,
                        node_type=NodeType.COMPONENT.value,
                        label=f"Component {target_id}",
                        name=target_id,
                        project_id=self.project_id,
                        source_type="file_tree",
                        provenance="file_path_hierarchy"
                    )
                self.add_edge_if_not_exists(
                    node_id, comp_node_id,
                    relationship=RelationshipType.MODIFIES_COMPONENT.value,
                    properties={
                        "provenance": rel.get("provenance", "commit_modified_files"),
                        "component_name": target_id,
                        "timestamp": record.timestamp or ""
                    }
                )

        return node_id

    def add_issue_record(self, record: SourceRecord) -> str:
        """Adds a Jira issue node and connects it to the project and linked components/issues."""
        proj_id = self.ensure_project_node()
        node_id = self._issue_node_id(record.source_id)

        self.graph.add_node(
            node_id,
            node_type=NodeType.ISSUE.value,
            label=f"Issue {record.source_id.upper()}",
            key=record.source_id.upper(),
            summary=record.metadata.get("summary", record.title),
            status=record.status or "Open",
            priority=record.metadata.get("priority", "Normal"),
            issue_type=record.metadata.get("issue_type", "Issue"),
            description=record.content,
            url=record.url,
            project_id=self.project_id,
            source_type=SourceType.JIRA.value,
            provenance="jira_rest_api"
        )

        # Issue BELONGS_TO Project
        self.add_edge_if_not_exists(
            node_id, proj_id,
            relationship=RelationshipType.BELONGS_TO.value,
            properties={"provenance": "jira_project_mapping"}
        )

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
                        project_id=self.project_id,
                        provenance="jira_component_field"
                    )
                self.add_edge_if_not_exists(
                    node_id, comp_node_id,
                    relationship=RelationshipType.RELATED_TO.value,
                    properties={"provenance": "jira_issue_components"}
                )

            elif target_type == "jira" and target_id:
                other_issue_id = self._issue_node_id(target_id)
                if not self.graph.has_node(other_issue_id):
                    self.graph.add_node(
                        other_issue_id,
                        node_type=NodeType.ISSUE.value,
                        label=f"Issue {target_id.upper()}",
                        key=target_id.upper(),
                        project_id=self.project_id,
                        status="LinkedIssue",
                        provenance="jira_issue_link"
                    )
                self.add_edge_if_not_exists(
                    node_id, other_issue_id,
                    relationship=RelationshipType.RELATED_TO.value,
                    properties={"provenance": "jira_issuelinks_field"}
                )

        return node_id

    def add_endpoint_record(self, record: SourceRecord) -> str:
        """Adds an OpenAPI endpoint node and connects it to the project and tags/components."""
        proj_id = self.ensure_project_node()
        node_id = self._endpoint_node_id(record.source_id)

        method = record.metadata.get("method", "GET").upper()
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
            project_id=self.project_id,
            source_type=SourceType.OPENAPI.value,
            provenance="openapi_spec_definition"
        )

        # Project EXPOSES_ENDPOINT Endpoint
        self.add_edge_if_not_exists(
            proj_id, node_id,
            relationship=RelationshipType.EXPOSES_ENDPOINT.value,
            properties={"provenance": "openapi_specification"}
        )

        # Connect tags to components
        for tag in record.metadata.get("tags", []):
            comp_node_id = self._component_node_id(tag)
            if not self.graph.has_node(comp_node_id):
                self.graph.add_node(
                    comp_node_id,
                    node_type=NodeType.COMPONENT.value,
                    label=f"Component {tag}",
                    name=tag,
                    project_id=self.project_id,
                    provenance="openapi_tag"
                )
            self.add_edge_if_not_exists(
                node_id, comp_node_id,
                relationship=RelationshipType.RELATED_TO.value,
                properties={"provenance": "openapi_tag_mapping"}
            )

        return node_id

    def add_edge_if_not_exists(
        self,
        u: str,
        v: str,
        relationship: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """Adds a directed edge between u and v if it does not already exist with the same relationship."""
        if not self.graph.has_node(u) or not self.graph.has_node(v):
            logger.warning(f"Skipping edge creation: node {u} or {v} does not exist in graph.")
            return

        if self.graph.has_edge(u, v):
            for edge_data in self.graph.get_edge_data(u, v).values():
                if edge_data.get("relationship") == relationship:
                    return  # Deduplicated

        props = properties or {}
        props["relationship"] = relationship
        self.graph.add_edge(u, v, **props)

    def find_node(self, identifier: str) -> Optional[str]:
        """
        Fuzzy or exact lookup for node ID given a keyword, SHA, issue key, or endpoint path.
        """
        clean_id = identifier.strip()

        # 0. If it is already an exact node ID in the graph, return it directly
        if self.graph.has_node(clean_id):
            return clean_id

        # 1. Try exact node ID prefix construction
        for prefix in (self._issue_node_id, self._commit_node_id, self._endpoint_node_id, self._component_node_id):
            cand = prefix(clean_id)
            if self.graph.has_node(cand):
                return cand

        # 2. Try attribute-based matching
        clean_lower = clean_id.lower()
        for n, data in self.graph.nodes(data=True):
            if data.get("key", "").lower() == clean_lower:
                return n
            if data.get("name", "").lower() == clean_lower:
                return n
            if data.get("path", "").lower() == clean_lower:
                return n

        # Check commit SHA matching with prefix tolerance (>= 7 chars)
        for n, data in self.graph.nodes(data=True):
            if data.get("node_type") == NodeType.COMMIT.value:
                full_sha = data.get("sha", "").lower()
                short_sha = data.get("short_sha", "").lower()
                if full_sha == clean_lower or short_sha == clean_lower:
                    return n
                if len(clean_lower) >= 7 and (full_sha.startswith(clean_lower) or clean_lower.startswith(short_sha)):
                    return n

        # 3. Substring matching for component keywords
        for n, data in self.graph.nodes(data=True):
            if data.get("node_type") == NodeType.COMPONENT.value:
                comp_name = data.get("name", "").lower()
                if comp_name and (comp_name == clean_lower or (len(clean_lower) > 3 and clean_lower in comp_name)):
                    return n

        return None

    def remove_record(self, source_type: Any, source_id: str) -> bool:
        """
        Removes a record node and all its incident edges from the Knowledge Graph.
        """
        st = source_type.value if hasattr(source_type, "value") else str(source_type).lower()
        node_id = None
        if st in (SourceType.GITHUB.value, "github", "commit"):
            node_id = self.find_node(source_id) or self._commit_node_id(source_id)
        elif st in (SourceType.JIRA.value, "jira", "issue"):
            node_id = self._issue_node_id(source_id)
        elif st in (SourceType.OPENAPI.value, "openapi", "endpoint"):
            node_id = self._endpoint_node_id(source_id)

        if node_id and self.graph.has_node(node_id):
            self.graph.remove_node(node_id)
            logger.info(f"Removed node {node_id} and its incident edges from Knowledge Graph.")
            return True
        return False

    def get_related_entities(
        self,
        node_id: str,
        max_depth: int = 2,
        max_nodes: int = 25
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Bounded BFS traversal to discover connected nodes and edges with cycle detection.
        Returns: (nodes_list, edges_list)
        """
        nodes, edges, _ = self.get_subgraph_for_seeds([node_id], max_depth=max_depth, max_total_nodes=max_nodes)
        return nodes, edges

    def get_subgraph_for_seeds(
        self,
        seed_identifiers: List[str],
        max_depth: int = 2,
        max_total_nodes: int = 25
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
        """
        Bounded, cycle-safe BFS traversal across multiple seed entities with a global node budget.
        Deduplicates nodes and edges, preserving exact edge provenance and structured edge data.
        Returns: (nodes_list, edges_list, formatted_graph_context_string)
        """
        resolved_seeds = []
        for s in seed_identifiers:
            nid = s if self.graph.has_node(s) else self.find_node(s)
            if nid and nid not in resolved_seeds:
                resolved_seeds.append(nid)

        if not resolved_seeds:
            return [], [], ""

        visited_nodes: Set[str] = set()
        queue: List[Tuple[str, int]] = [(s, 0) for s in resolved_seeds]
        for s in resolved_seeds:
            visited_nodes.add(s)

        discovered_edges: List[Dict[str, Any]] = []
        seen_edges: Set[Tuple[str, str, str]] = set()

        while queue and len(visited_nodes) < max_total_nodes:
            curr, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            # Forward edges
            for nbr in self.graph.successors(curr):
                for k, edata in self.graph.get_edge_data(curr, nbr).items():
                    rel = edata.get("relationship", "RELATED_TO")
                    edge_key = (curr, nbr, rel)
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        discovered_edges.append({
                            "source": curr,
                            "target": nbr,
                            "relationship": rel,
                            "provenance": edata.get("provenance", "inferred")
                        })
                if nbr not in visited_nodes and len(visited_nodes) < max_total_nodes:
                    visited_nodes.add(nbr)
                    queue.append((nbr, depth + 1))

            # Backward edges
            for nbr in self.graph.predecessors(curr):
                for k, edata in self.graph.get_edge_data(nbr, curr).items():
                    rel = edata.get("relationship", "RELATED_TO")
                    edge_key = (nbr, curr, rel)
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        discovered_edges.append({
                            "source": nbr,
                            "target": curr,
                            "relationship": rel,
                            "provenance": edata.get("provenance", "inferred")
                        })
                if nbr not in visited_nodes and len(visited_nodes) < max_total_nodes:
                    visited_nodes.add(nbr)
                    queue.append((nbr, depth + 1))

        nodes_result = []
        for n in visited_nodes:
            ndata = self.graph.nodes[n]
            nodes_result.append({
                "id": n,
                "type": ndata.get("node_type", "unknown"),
                "label": ndata.get("label", n),
                "properties": {k: v for k, v in ndata.items() if k not in ("message", "description")}
            })

        # Format factual relational statements with provenance
        lines = []
        if discovered_edges:
            lines.append("=== VERIFIED KNOWLEDGE GRAPH RELATIONSHIPS (PROVENANCE-BACKED) ===")
            nodes_map = {n["id"]: n for n in nodes_result}
            for e in sorted(discovered_edges, key=lambda x: (x["source"], x["target"], x["relationship"])):
                src, tgt, rel, prov = e["source"], e["target"], e["relationship"], e.get("provenance", "ground_truth")
                src_node = nodes_map.get(src, self.graph.nodes.get(src, {}))
                tgt_node = nodes_map.get(tgt, self.graph.nodes.get(tgt, {}))

                src_label = src_node.get("label", src) if isinstance(src_node, dict) else src
                tgt_label = tgt_node.get("label", tgt) if isinstance(tgt_node, dict) else tgt

                extra = []
                if isinstance(src_node, dict):
                    props = src_node.get("properties", src_node)
                    if props.get("author"):
                        extra.append(f"Author: {props['author']}")
                    if props.get("status"):
                        extra.append(f"Status: {props['status']}")
                extra_str = f" ({', '.join(extra)})" if extra else ""

                lines.append(f"• [{rel}] {src_label}{extra_str}  -->  {tgt_label}  [Provenance: {prov}]")

        formatted_context = "\n".join(lines)
        return nodes_result, discovered_edges, formatted_context

    def format_graph_context(self, seed_entity_ids: List[str], max_depth: int = 2) -> str:
        """
        Extracts subgraphs for seed entities and renders human-readable factual relational statements with provenance.
        """
        _, _, context_str = self.get_subgraph_for_seeds(seed_entity_ids, max_depth=max_depth)
        return context_str

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
