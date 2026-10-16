"""
Hybrid Graph-RAG Retrieval Pipeline.
Combines ChromaDB vector semantic search with NetworkX Knowledge Graph traversal
and passes fused evidence to Groq LLM for grounded answer generation.
"""

import os
import re
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

try:
    from .knowledge_graph import SoftwareKnowledgeGraph
    from .models_data import SourceType
except ImportError:
    from knowledge_graph import SoftwareKnowledgeGraph
    from models_data import SourceType

logger = logging.getLogger(__name__)

# Patterns for identifying software engineering entities in queries and text
JIRA_PATTERN = re.compile(r'\b[A-Z][A-Z0-9]+-[0-9]+\b')
SHA_PATTERN = re.compile(r'\b[0-9a-f]{7,40}\b', re.IGNORECASE)
ENDPOINT_PATTERN = re.compile(r'(?:GET|POST|PUT|DELETE|PATCH)\s+/[^\s,;]+|/[a-zA-Z0-9_/{}.-]{3,}', re.IGNORECASE)


def extract_entities_from_text(text: str) -> Dict[str, List[str]]:
    """
    Extracts Jira keys, commit SHAs, and API endpoint paths from user query or text.
    """
    if not text:
        return {"jira_keys": [], "commit_shas": [], "endpoints": []}

    jira_keys = sorted(list(set(JIRA_PATTERN.findall(text))))
    shas = sorted(list(set(SHA_PATTERN.findall(text))))
    # Filter common hex false positives if needed
    shas = [s for s in shas if not s.isdigit()]

    endpoints = []
    for match in ENDPOINT_PATTERN.findall(text):
        cleaned = match.strip().strip("'\"()[]")
        if "/" in cleaned and len(cleaned) > 2:
            endpoints.append(cleaned)
    endpoints = sorted(list(set(endpoints)))

    return {
        "jira_keys": jira_keys,
        "commit_shas": shas,
        "endpoints": endpoints
    }


class HybridGraphRAG:
    """
    Genuine Hybrid Graph-RAG pipeline orchestrating vector retrieval, graph traversal,
    context fusion, and LLM grounded generation.
    """

    def __init__(
        self,
        project_id: str,
        vector_store: Chroma,
        knowledge_graph: SoftwareKnowledgeGraph,
        groq_api_key: str,
        model_name: str = "qwen/qwen3.8-27b",
        top_k: int = 5,
        graph_depth: int = 2
    ):
        self.project_id = project_id
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.groq_api_key = groq_api_key
        self.model_name = model_name
        self.top_k = top_k
        self.graph_depth = graph_depth

        self.llm = ChatGroq(
            temperature=0.0,
            model_name=self.model_name,
            groq_api_key=self.groq_api_key,
            timeout=30
        )

    def retrieve(self, question: str) -> Tuple[List[Document], str, List[Dict[str, Any]], List[Dict[str, Any]], str, Optional[str]]:
        """
        Executes hybrid retrieval:
        1. Vector search in ChromaDB.
        2. Entity extraction from query + vector chunks.
        3. Knowledge Graph traversal.
        4. Structured evidence fusion.

        Returns: (vector_documents, graph_context_str, graph_entities, graph_edges, retrieval_mode, fallback_reason)
        """
        retrieval_mode = "vector_only_fallback"
        fallback_reason = None
        vector_docs: List[Document] = []

        # 1. Vector Retrieval
        try:
            vector_docs = self.vector_store.similarity_search(question, k=self.top_k)
            logger.info(f"Retrieved {len(vector_docs)} vector candidates from ChromaDB.")
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")

        # 2. Entity Identification
        query_entities = extract_entities_from_text(question)
        seed_entities: Set[str] = set()

        for group in query_entities.values():
            seed_entities.update(group)

        # Also inspect top vector docs for additional entities
        for doc in vector_docs:
            source_id = doc.metadata.get("source_id", "")
            if source_id:
                seed_entities.add(source_id)

            jira_str = doc.metadata.get("jira_keys", "")
            if jira_str:
                for k in jira_str.split(","):
                    if k.strip():
                        seed_entities.add(k.strip())

        # 3. Knowledge Graph Traversal (Bounded and Deduplicated across seeds)
        graph_context = ""
        graph_entities: List[Dict[str, Any]] = []
        graph_edges: List[Dict[str, Any]] = []

        try:
            if seed_entities and self.knowledge_graph.graph.number_of_nodes() > 0:
                logger.info(f"Traversing Knowledge Graph from seed entities: {seed_entities}")
                graph_entities, graph_edges, graph_context = self.knowledge_graph.get_subgraph_for_seeds(
                    list(seed_entities),
                    max_depth=self.graph_depth,
                    max_total_nodes=25
                )

                if graph_context and len(graph_entities) > 0:
                    retrieval_mode = "hybrid_graph_rag"
                    fallback_reason = None
                    logger.info(f"Hybrid Graph-RAG active: {len(graph_entities)} nodes, {len(graph_edges)} edges connected.")
                else:
                    retrieval_mode = "vector_only_fallback"
                    fallback_reason = "NO_CONNECTING_RELATIONSHIPS"
                    logger.info("Graph traversal found no connecting relationships for seeds; vector-only fallback.")
            else:
                retrieval_mode = "vector_only_fallback"
                fallback_reason = "NO_SEED_ENTITIES" if not seed_entities else "EMPTY_GRAPH"
                logger.info("No identifiable entities found or empty graph; vector-only fallback.")

        except Exception as e:
            logger.error(f"Graph traversal error: {e}. Falling back to vector retrieval.")
            retrieval_mode = "vector_only_fallback"
            fallback_reason = f"GRAPH_TRAVERSAL_ERROR: {e}"
            graph_context = ""
            graph_entities = []
            graph_edges = []

        return vector_docs, graph_context, graph_entities, graph_edges, retrieval_mode, fallback_reason

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Runs the full Hybrid Graph-RAG process and formats a grounded response with citations.
        """
        vector_docs, graph_context, graph_entities, graph_edges, retrieval_mode, fallback_reason = self.retrieve(question)
        is_full_hybrid = (retrieval_mode == "hybrid_graph_rag")

        # Extract verified sources from vector docs and graph entities
        sources_map: Dict[str, Dict[str, Any]] = {}
        for doc in vector_docs:
            src_type = doc.metadata.get("source_type", "doc")
            src_id = doc.metadata.get("source_id", "unknown")
            title = doc.metadata.get("title", f"{src_type}: {src_id}")
            url = doc.metadata.get("url", "")
            key = f"{src_type}:{src_id}"

            if key not in sources_map:
                sources_map[key] = {
                    "source_type": src_type,
                    "source_id": src_id,
                    "title": title,
                    "url": url,
                    "status": doc.metadata.get("status", "")
                }

        for ge in graph_entities:
            g_type = ge.get("type", "entity")
            g_id = ge.get("id", "")
            g_label = ge.get("label", g_id)
            g_key = f"{g_type}:{g_id}"
            if g_key not in sources_map and g_type != "project":
                sources_map[g_key] = {
                    "source_type": g_type,
                    "source_id": g_id,
                    "title": g_label,
                    "url": ge.get("properties", {}).get("url", ""),
                    "status": ge.get("properties", {}).get("status", "")
                }

        # Check for empty retrieval
        if not vector_docs and not graph_entities:
            return {
                "answer": "No relevant documents or knowledge graph entities were found for this query in the repository.",
                "repo": self.project_id,
                "retrieval_mode": retrieval_mode,
                "is_full_hybrid": False,
                "fallback_reason": fallback_reason or "EMPTY_RETRIEVAL",
                "sources": [],
                "graph_entities": [],
                "graph_edges": [],
                "evidence": [],
                "evidence_texts": [],
                "graph_context": "",
                "evidence_count": 0,
                "graph_node_count": 0,
                "graph_edge_count": 0,
                "status": "EMPTY_RETRIEVAL"
            }

        # Build fused context
        context_parts = []
        if graph_context:
            context_parts.append(graph_context)

        context_parts.append("=== RETRIEVED DOCUMENTATION & SOURCE EVIDENCE ===")
        if vector_docs:
            for idx, doc in enumerate(vector_docs, 1):
                src = doc.metadata.get("source_type", "source").upper()
                sid = doc.metadata.get("source_id", f"doc_{idx}")
                context_parts.append(f"[SOURCE {idx}: {src} {sid}]\n{doc.page_content.strip()}")
        else:
            context_parts.append("No direct vector documents found for this query.")

        full_context = "\n\n".join(context_parts)

        # System prompt enforcing strict source grounding and citation
        system_prompt = (
            f"You are a Software Engineering Knowledge Assistant specialized in '{self.project_id}'.\n"
            "Your task is to answer developer questions thoroughly, accurately, and authoritatively using "
            "ONLY the provided Knowledge Graph relationships and retrieved documentation context.\n\n"
            "CRITICAL GUIDELINES:\n"
            "1. Grounding: Answer strictly using the facts in the Context. Do not invent commits, tickets, or endpoints.\n"
            "2. Hybrid Synthesis: If the Knowledge Graph shows relationships (e.g. Commit A REFERENCES_ISSUE Jira B), "
            "explicitly connect these insights in your answer to explain how code changes, issues, and endpoints interact.\n"
            "3. Precise Citations: Mention specific GitHub commit SHAs (e.g. `90a4ce9`), Jira issue keys (e.g. `DHIS2-20751`), "
            "and HTTP API endpoints (e.g. `GET /api/dataValueSets`) whenever referencing evidence.\n"
            "4. Missing Info: If the provided context does not contain sufficient details to answer, state clearly that "
            "the evidence is insufficient rather than guessing.\n"
            "5. Formatting: Use clean GitHub-flavored markdown with bullet points, code spans, and bold headers."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Context:\n{context}\n\nDeveloper Question: {question}")
        ])

        chain = prompt | self.llm

        try:
            logger.info(f"Invoking Groq LLM with {retrieval_mode} context...")
            response = chain.invoke({"context": full_context, "question": question})
            answer_text = response.content
            run_status = "SUCCESS" if is_full_hybrid else "HYBRID_FALLBACK_VECTOR"
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            fallback_snippets = [doc.page_content.strip()[:200] for doc in vector_docs[:3]]
            answer_text = f"Extractive context ({len(vector_docs)} documents, {len(graph_entities)} graph entities retrieved):\n" + "\n\n".join(fallback_snippets)
            run_status = "EXTRACTIVE_FALLBACK"

        evidence_list = [d.page_content for d in vector_docs]
        return {
            "answer": answer_text,
            "repo": self.project_id,
            "retrieval_mode": retrieval_mode,
            "is_full_hybrid": is_full_hybrid,
            "fallback_reason": fallback_reason,
            "sources": list(sources_map.values()),
            "graph_entities": graph_entities,
            "graph_edges": graph_edges,
            "evidence": evidence_list,
            "evidence_texts": evidence_list,
            "graph_context": graph_context,
            "evidence_count": len(vector_docs),
            "graph_node_count": len(graph_entities),
            "graph_edge_count": len(graph_edges),
            "status": run_status
        }
