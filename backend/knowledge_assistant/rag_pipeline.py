"""
RAG Pipeline Orchestrator for Cross-Platform Knowledge Assistant.
Coordinates multi-source data ingestion, document preprocessing, ChromaDB vector indexing,
Knowledge Graph construction, and Hybrid Graph-RAG answer generation.
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

try:
    from .models_data import SourceRecord, SourceType
    from .github_fetcher import fetch_github_records
    from .jira_fetcher import fetch_jira_records
    from .swagger_fetcher import fetch_openapi_records
    from .document_processor import DocumentProcessor
    from .knowledge_graph import SoftwareKnowledgeGraph
    from .hybrid_rag import HybridGraphRAG
except ImportError:
    from models_data import SourceRecord, SourceType
    from github_fetcher import fetch_github_records
    from jira_fetcher import fetch_jira_records
    from swagger_fetcher import fetch_openapi_records
    from document_processor import DocumentProcessor
    from knowledge_graph import SoftwareKnowledgeGraph
    from hybrid_rag import HybridGraphRAG

BASE_DIR = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / '.env')
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Free local HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
doc_processor = DocumentProcessor(chunk_size=700, chunk_overlap=120)

PROJECT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "dhis2/dhis2-core": {
        "name": "DHIS2 Core Platform",
        "slug": "dhis2",
        "jira_url": os.getenv("JIRA_BASE_URL", "https://dhis2.atlassian.net"),
        "jira_project": "DHIS2",
        "github_repo": "dhis2/dhis2-core",
        "chroma_dir": str(BASE_DIR / "chroma_db_dhis2"),
        "graph_file": str(BASE_DIR / "knowledge_graph_dhis2.json"),
        "openapi_source": str(BASE_DIR / "specs" / "dhis2_openapi.json"),
    },
    "opendaylight/transportpce": {
        "name": "TransportPCE (OpenDaylight)",
        "slug": "transportpce",
        "jira_url": "https://lf-opendaylight.atlassian.net",
        "jira_project": "TRNSPRTPCE",
        "github_repo": "opendaylight/transportpce",
        "chroma_dir": str(BASE_DIR / "chroma_db_transportpce"),
        "graph_file": str(BASE_DIR / "knowledge_graph_transportpce.json"),
        "openapi_source": str(BASE_DIR / "specs" / "transportpce_openapi.json"),
    }
}


def normalize_repo(repo: Optional[str], question: str = "") -> str:
    """
    Determines the target project repository key.
    Prioritizes explicit keywords in the question if provided.
    """
    q = (question or "").lower()
    has_dhis = "dhis" in q
    has_transport = "transport" in q or "opendaylight" in q or "pce" in q

    if has_dhis and not has_transport:
        return "dhis2/dhis2-core"
    elif has_transport and not has_dhis:
        return "opendaylight/transportpce"

    if not repo:
        return "dhis2/dhis2-core"

    r = repo.lower().strip()
    if "transport" in r or "opendaylight" in r:
        return "opendaylight/transportpce"
    return "dhis2/dhis2-core"


def get_knowledge_graph(repo_key: str) -> SoftwareKnowledgeGraph:
    """Loads or initializes the Knowledge Graph for a specific project."""
    config = PROJECT_CONFIGS.get(repo_key, PROJECT_CONFIGS["dhis2/dhis2-core"])
    kg = SoftwareKnowledgeGraph(project_id=repo_key)
    graph_file = config["graph_file"]
    if os.path.exists(graph_file):
        kg.load_from_file(graph_file)
    else:
        kg.ensure_project_node(config["name"])
    return kg


def get_vector_store(repo_key: str) -> Chroma:
    """Initializes the ChromaDB vector store instance for a project."""
    config = PROJECT_CONFIGS.get(repo_key, PROJECT_CONFIGS["dhis2/dhis2-core"])
    chroma_dir = config["chroma_dir"]
    os.makedirs(chroma_dir, exist_ok=True)
    return Chroma(persist_directory=chroma_dir, embedding_function=embeddings)


def load_and_index_data(repo: str = "dhis2/dhis2-core", max_commits: int = 25, max_issues: int = 20) -> Dict[str, Any]:
    """
    Ingests data across GitHub, Jira, and OpenAPI for the target repository,
    populates both ChromaDB and SoftwareKnowledgeGraph with deduplication,
    and saves the updated knowledge graph to disk.
    """
    repo_key = normalize_repo(repo)
    config = PROJECT_CONFIGS.get(repo_key, PROJECT_CONFIGS["dhis2/dhis2-core"])

    print(f"\n==================================================")
    print(f"Indexing data for project: {config['name']} ({repo_key})")
    print(f"Target ChromaDB: {config['chroma_dir']}")
    print(f"Target Knowledge Graph: {config['graph_file']}")
    print(f"==================================================")

    vector_store = get_vector_store(repo_key)
    new_kg = SoftwareKnowledgeGraph(project_id=repo_key)
    new_kg.ensure_project_node(config["name"])

    all_records: List[SourceRecord] = []

    # 1. Ingest Jira Issues
    print(f"Fetching Jira Issues for {config['name']} ({config['jira_project']})...")
    jira_records = fetch_jira_records(
        base_url=config["jira_url"],
        project_key=config["jira_project"],
        max_results=max_issues
    )
    for jrec in jira_records:
        new_kg.add_issue_record(jrec)
        all_records.append(jrec)
    print(f"Retrieved {len(jira_records)} Jira issue records.")

    # 2. Ingest GitHub Commits
    print(f"Fetching GitHub Commits for {config['github_repo']}...")
    github_records = fetch_github_records(
        repo_name=config["github_repo"],
        max_commits=max_commits
    )
    for grec in github_records:
        new_kg.add_commit_record(grec)
        all_records.append(grec)
    print(f"Retrieved {len(github_records)} GitHub commit records.")

    # 3. Ingest OpenAPI / Swagger Specifications
    print(f"Parsing OpenAPI specifications from {config['openapi_source']}...")
    openapi_records = fetch_openapi_records(
        source_url_or_path=config["openapi_source"],
        project_id=repo_key
    )
    for orec in openapi_records:
        new_kg.add_endpoint_record(orec)
        all_records.append(orec)
    print(f"Retrieved {len(openapi_records)} OpenAPI endpoint records.")

    # 4. Process chunks for Vector Store
    chunked_pairs = doc_processor.process_records(all_records)
    print(f"Split {len(all_records)} source records into {len(chunked_pairs)} vector chunks.")

    new_chunk_ids = [cid for cid, _ in chunked_pairs]
    docs = [d for _, d in chunked_pairs]

    # 5. Synchronize ChromaDB: remove obsolete chunks and insert/update current chunks
    try:
        existing_data = vector_store.get()
        existing_ids = set(existing_data.get("ids", []))
        obsolete_ids = existing_ids - set(new_chunk_ids)

        if obsolete_ids:
            print(f"Pruning {len(obsolete_ids)} obsolete vector chunks from ChromaDB...")
            vector_store.delete(ids=list(obsolete_ids))

        if docs:
            vector_store.add_documents(documents=docs, ids=new_chunk_ids)
            print(f"Successfully indexed {len(docs)} chunks into ChromaDB.")
    except Exception as e:
        logger.error(f"Critical error during vector store update: {e}", exc_info=True)
        # CRITICAL: Do NOT persist new graph if vector store fails, to prevent state inconsistency!
        raise RuntimeError(f"Vector store indexing failed: {e}") from e

    # 6. Atomic consistency: Only persist Knowledge Graph after vector store successfully updates
    new_kg.save_to_file(config["graph_file"])
    stats = new_kg.get_stats()
    print(f"Saved Knowledge Graph with {stats['total_nodes']} nodes and {stats['total_edges']} edges.\n")

    return {
        "project": config["name"],
        "repo": repo_key,
        "jira_issues": len(jira_records),
        "github_commits": len(github_records),
        "openapi_endpoints": len(openapi_records),
        "vector_chunks": len(chunked_pairs),
        "graph_stats": stats
    }


def ask_ai_hybrid(
    question: str,
    groq_api_key: Optional[str] = None,
    repo: str = "dhis2/dhis2-core",
    top_k: int = 5,
    graph_depth: int = 2
) -> Dict[str, Any]:
    """
    Executes Hybrid Graph-RAG retrieval and answers the question.
    Returns structured dictionary with answer, sources, retrieval_mode, and graph_entities.
    """
    key = groq_api_key or os.getenv("GROQ_API_KEY", "")
    if not key:
        raise ValueError("GROQ_API_KEY must be provided or configured in .env")

    repo_key = normalize_repo(repo, question)
    config = PROJECT_CONFIGS.get(repo_key, PROJECT_CONFIGS["dhis2/dhis2-core"])

    # Ensure vector store and knowledge graph are populated
    chroma_dir = config["chroma_dir"]
    graph_file = config["graph_file"]

    if not os.path.exists(chroma_dir) or not os.path.exists(graph_file):
        print(f"Data store missing for {config['name']}. Triggering initial indexing...")
        load_and_index_data(repo_key)

    vector_store = get_vector_store(repo_key)
    kg = get_knowledge_graph(repo_key)

    hybrid_rag = HybridGraphRAG(
        project_id=repo_key,
        vector_store=vector_store,
        knowledge_graph=kg,
        groq_api_key=key,
        top_k=top_k,
        graph_depth=graph_depth
    )

    return hybrid_rag.answer_question(question)


def ask_ai(
    question: str,
    groq_api_key: Optional[str] = None,
    repo: str = "dhis2/dhis2-core",
    return_structured: bool = False
) -> Union[str, Dict[str, Any]]:
    """
    Backwards-compatible interface.
    Returns string answer if return_structured=False, or structured dict if return_structured=True.
    """
    result = ask_ai_hybrid(question, groq_api_key, repo=repo)
    if return_structured:
        return result
    return result.get("answer", "")


def get_project_sync_status(repo: str = "dhis2/dhis2-core") -> Dict[str, Any]:
    """
    Inspects actual local ChromaDB and Knowledge Graph stores to report real synchronization state.
    """
    repo_key = normalize_repo(repo)
    config = PROJECT_CONFIGS.get(repo_key, PROJECT_CONFIGS["dhis2/dhis2-core"])

    chroma_dir = config["chroma_dir"]
    graph_file = config["graph_file"]

    vector_count = 0
    if os.path.exists(chroma_dir):
        try:
            vs = get_vector_store(repo_key)
            vector_count = vs._collection.count()
        except Exception:
            pass

    graph_stats = {
        "total_nodes": 0,
        "total_edges": 0,
        "nodes_by_type": {},
        "edges_by_relationship": {}
    }
    if os.path.exists(graph_file):
        try:
            kg = get_knowledge_graph(repo_key)
            graph_stats = kg.get_stats()
        except Exception:
            pass

    is_synced = vector_count > 0 and graph_stats["total_nodes"] > 0

    return {
        "repo": repo_key,
        "name": config["name"],
        "is_synced": is_synced,
        "vector_chunks_count": vector_count,
        "graph_nodes_count": graph_stats["total_nodes"],
        "graph_edges_count": graph_stats["total_edges"],
        "nodes_breakdown": graph_stats["nodes_by_type"],
        "edges_breakdown": graph_stats["edges_by_relationship"]
    }


if __name__ == "__main__":
    print("=== Testing Hybrid Graph-RAG Pipeline ===")
    test_key = os.getenv("GROQ_API_KEY")
    if not test_key:
        print("Notice: Set GROQ_API_KEY to test LLM generation.")

    # Reindex DHIS2 and TransportPCE
    print("\n1. Indexing DHIS2...")
    load_and_index_data("dhis2/dhis2-core", max_commits=10, max_issues=10)

    print("\n2. Indexing TransportPCE...")
    load_and_index_data("opendaylight/transportpce", max_commits=10, max_issues=10)

    if test_key:
        print("\n3. Testing Hybrid Query...")
        res = ask_ai_hybrid(
            "What bug fixes or commits are related to DHIS2 issues and what API endpoints are available?",
            groq_api_key=test_key,
            repo="dhis2/dhis2-core"
        )
        print("\nRetrieval Mode:", res.get("retrieval_mode"))
        print("Verified Sources Count:", len(res.get("sources", [])))
        print("\nAnswer:\n", res.get("answer"))
