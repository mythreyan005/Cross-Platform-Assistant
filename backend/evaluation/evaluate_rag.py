"""
Evaluation Framework: Comparative Benchmark of Vector-Only RAG vs. Hybrid Graph-RAG.
Measures retrieval coverage, cross-source relationship discovery, response latency, and groundedness.
All metrics are dynamically calculated from ground-truth source artifacts.
"""

import os
import time
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Set

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / '.env')
except ImportError:
    pass

from knowledge_assistant.rag_pipeline import (
    get_vector_store,
    get_knowledge_graph,
    PROJECT_CONFIGS,
    normalize_repo
)
from knowledge_assistant.hybrid_rag import HybridGraphRAG, extract_entities_from_text
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_rag")

# 6-Question Research Benchmark Dataset with Ground-Truth Target Artifact IDs
EVALUATION_DATASET = [
    {
        "id": "Q1",
        "question": "Which GitHub commits are linked to Jira issue TRNSPRTPCE-980 in TransportPCE?",
        "repo": "opendaylight/transportpce",
        "expected_source_ids": ["TRNSPRTPCE-980", "de74a89", "e10ac2a"],
        "expected_relationship": "REFERENCES_ISSUE",
        "expected_entity_ids": ["issue:TRNSPRTPCE-980", "commit:de74a89", "commit:e10ac2a"],
        "target_type": "commit_to_issue",
        "description": "Multi-source cross-referencing between Git commits and Jira issue TRNSPRTPCE-980."
    },
    {
        "id": "Q2",
        "question": "What API endpoints in DHIS2 handle analytical queries and data value sets?",
        "repo": "dhis2/dhis2-core",
        "expected_source_ids": ["GET /api/dataValueSets", "POST /api/dataValueSets", "GET /api/analytics"],
        "expected_relationship": "EXPOSES_ENDPOINT",
        "expected_entity_ids": ["endpoint:GET:/api/dataValueSets", "endpoint:POST:/api/dataValueSets", "endpoint:GET:/api/analytics"],
        "target_type": "api_endpoints",
        "description": "Tests discovery of OpenAPI endpoints and operations in DHIS2."
    },
    {
        "id": "Q3",
        "question": "Which commits modified the tapi component in TransportPCE?",
        "repo": "opendaylight/transportpce",
        "expected_source_ids": ["de74a89", "2e7e995", "490a8fb"],
        "expected_relationship": "MODIFIES_COMPONENT",
        "expected_entity_ids": ["component:tapi", "commit:de74a89", "commit:2e7e995"],
        "target_type": "commit_to_component",
        "description": "Tests discovery of component-level modifications in commits."
    },
    {
        "id": "Q4",
        "question": "What user profile information and settings are accessible via the DHIS2 API?",
        "repo": "dhis2/dhis2-core",
        "expected_source_ids": ["GET /api/me", "GET /api/me/authorization"],
        "expected_relationship": "EXPOSES_ENDPOINT",
        "expected_entity_ids": ["endpoint:GET:/api/me", "endpoint:GET:/api/me/authorization"],
        "target_type": "api_endpoints",
        "description": "Tests specific endpoint documentation retrieval (/api/me)."
    },
    {
        "id": "Q5",
        "question": "What RESTCONF RPC operations and endpoints are available for path computation in TransportPCE?",
        "repo": "opendaylight/transportpce",
        "expected_source_ids": ["POST /restconf/operations/transportpce-pce:path-computation-request"],
        "expected_relationship": "RELATED_TO",
        "expected_entity_ids": ["endpoint:POST:/restconf/operations/transportpce-pce:path-computation-request", "component:pce"],
        "target_type": "api_endpoints",
        "description": "Tests RPC operation and endpoint discovery for PCE services."
    },
    {
        "id": "Q6",
        "question": "What bug is reported in DHIS2-22296 regarding org unit merge and which API endpoints are involved?",
        "repo": "dhis2/dhis2-core",
        "expected_source_ids": ["DHIS2-22296", "POST /api/organisationUnits/merge"],
        "expected_relationship": "REFERENCES_ISSUE",
        "expected_entity_ids": ["issue:DHIS2-22296", "component:[api] other"],
        "target_type": "cross_source_issue_endpoint",
        "description": "Multi-source reasoning between Jira issue DHIS2-22296 and org unit merge endpoints."
    }
]


def call_groq_with_retry(chain: Any, inputs: Dict[str, Any], max_retries: int = 3, initial_delay: float = 3.0) -> Any:
    """Calls Groq LLM with exponential backoff on HTTP 429 rate limit errors."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "Rate limit" in err_str:
                logger.warning(f"Groq 429 rate limit encountered on attempt {attempt + 1}. Backing off for {delay}s...")
                time.sleep(delay)
                delay *= 2.0
            else:
                logger.error(f"Groq API error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2.0)
    raise RuntimeError("Failed to obtain response from Groq LLM after retries.")


def run_vector_only_rag(question: str, repo: str, groq_api_key: str, top_k: int = 5) -> Dict[str, Any]:
    """Runs baseline vector-only RAG without Knowledge Graph traversal."""
    start_time = time.time()
    repo_key = normalize_repo(repo, question)
    vs = get_vector_store(repo_key)

    docs = vs.similarity_search(question, k=top_k)
    context_str = "\n\n".join([f"[{d.metadata.get('source_type', 'doc').upper()}] {d.page_content}" for d in docs])

    answer = ""
    try:
        llm = ChatGroq(temperature=0.0, model_name="qwen/qwen3.8-27b", groq_api_key=groq_api_key, max_tokens=250)
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"Answer strictly using the provided context for {repo_key}. Be factual and concise."),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])
        chain = prompt | llm
        resp = call_groq_with_retry(chain, {"context": context_str, "question": question})
        answer = resp.content
    except Exception as e:
        logger.warning(f"Vector RAG LLM call failed: {e}. Generating extractive fallback.")
        answer = f"Extractive context retrieved: {len(docs)} documents."

    latency = time.time() - start_time

    # Collect retrieved source identifiers
    retrieved_source_ids = []
    for d in docs:
        sid = d.metadata.get("source_id", "")
        if sid:
            retrieved_source_ids.append(sid)

    entities = extract_entities_from_text(f"{answer} {context_str}")

    return {
        "method": "vector_only",
        "answer": answer,
        "latency_sec": round(latency, 3),
        "docs_retrieved": len(docs),
        "retrieved_source_ids": retrieved_source_ids,
        "entities_identified": entities,
        "discovered_relationships": []  # Vector-only lacks relational discovery
    }


def run_hybrid_graph_rag(question: str, repo: str, groq_api_key: str, top_k: int = 5) -> Dict[str, Any]:
    """Runs Hybrid Graph-RAG combining vector similarity with Knowledge Graph traversal."""
    start_time = time.time()
    repo_key = normalize_repo(repo, question)
    vs = get_vector_store(repo_key)
    kg = get_knowledge_graph(repo_key)

    hybrid = HybridGraphRAG(
        project_id=repo_key,
        vector_store=vs,
        knowledge_graph=kg,
        groq_api_key=groq_api_key,
        top_k=top_k,
        graph_depth=2
    )

    res = hybrid.answer_question(question)
    latency = time.time() - start_time

    # Collect retrieved source IDs from documents and graph entities
    retrieved_source_ids = []
    for s in res.get("sources", []):
        sid = s.get("id") or s.get("source_id", "")
        if sid:
            retrieved_source_ids.append(sid)

    graph_entities = res.get("graph_entities", [])
    for g in graph_entities:
        gid = g.get("id", "")
        if gid:
            retrieved_source_ids.append(gid)

    entities = extract_entities_from_text(res.get("answer", ""))

    # Discover relationships mentioned in graph context
    graph_ctx = res.get("graph_context", "")
    discovered_rels = []
    for rel_type in ["REFERENCES_ISSUE", "MODIFIES_COMPONENT", "EXPOSES_ENDPOINT", "RELATED_TO", "BELONGS_TO"]:
        if f"[{rel_type}]" in graph_ctx:
            discovered_rels.append(rel_type)

    return {
        "method": "hybrid_graph_rag",
        "answer": res.get("answer", ""),
        "latency_sec": round(latency, 3),
        "docs_retrieved": len(res.get("sources", [])),
        "graph_entities_discovered": len(graph_entities),
        "retrieved_source_ids": retrieved_source_ids,
        "entities_identified": entities,
        "discovered_relationships": discovered_rels,
        "graph_context": graph_ctx,
        "retrieval_mode": res.get("retrieval_mode")
    }


def compute_metrics(
    expected_sources: List[str],
    retrieved_sources: List[str],
    expected_rel: str,
    discovered_rels: List[str],
    answer_text: str
) -> Dict[str, float]:
    """
    Dynamically computes Recall@k, Precision@k, Relationship Recall, and Groundedness.
    """
    exp_set = {s.lower().strip() for s in expected_sources}
    ret_set = {s.lower().strip() for s in retrieved_sources}

    # Match count (exact or substring match for SHAs and endpoints)
    matched = 0
    for exp in exp_set:
        if any(exp in ret or ret in exp for ret in ret_set):
            matched += 1

    recall = round(matched / len(exp_set), 4) if exp_set else 1.0
    precision = round(matched / len(ret_set), 4) if ret_set else 0.0

    # Relationship recall
    rel_recall = 1.0 if expected_rel in discovered_rels else 0.0

    # Groundedness: check if expected entities appear in generated answer
    grounded_count = sum(1 for exp in exp_set if exp in answer_text.lower())
    groundedness = round(grounded_count / len(exp_set), 4) if exp_set else 1.0

    return {
        "recall_at_k": recall,
        "precision_at_k": precision,
        "relationship_recall": rel_recall,
        "groundedness": groundedness
    }


def execute_evaluation(groq_api_key: str) -> Dict[str, Any]:
    """
    Executes benchmark comparison on the evaluation dataset.
    Generates quantitative comparison results without hardcoded assertions.
    """
    print(f"\n=======================================================")
    print(f"Executing Comparative Evaluation: Vector vs Hybrid RAG")
    print(f"Benchmark Dataset Size: {len(EVALUATION_DATASET)} queries")
    print(f"=======================================================\n")

    results = []
    v_recalls, h_recalls = [], []
    v_precisions, h_precisions = [], []
    v_rel_recalls, h_rel_recalls = [], []
    v_groundedness, h_groundedness = [], []
    v_latencies, h_latencies = [], []

    for item in EVALUATION_DATASET:
        qid = item["id"]
        q = item["question"]
        repo = item["repo"]
        exp_sources = item["expected_source_ids"]
        exp_rel = item["expected_relationship"]

        print(f"Evaluating {qid}: '{q}' [{repo}]...")

        # 1. Vector-Only RAG
        v_res = run_vector_only_rag(q, repo, groq_api_key)
        v_metrics = compute_metrics(
            exp_sources,
            v_res["retrieved_source_ids"],
            exp_rel,
            v_res["discovered_relationships"],
            v_res["answer"]
        )

        v_recalls.append(v_metrics["recall_at_k"])
        v_precisions.append(v_metrics["precision_at_k"])
        v_rel_recalls.append(v_metrics["relationship_recall"])
        v_groundedness.append(v_metrics["groundedness"])
        v_latencies.append(v_res["latency_sec"])

        # Delay to avoid Groq rate limit
        time.sleep(2.5)

        # 2. Hybrid Graph-RAG
        h_res = run_hybrid_graph_rag(q, repo, groq_api_key)
        h_metrics = compute_metrics(
            exp_sources,
            h_res["retrieved_source_ids"],
            exp_rel,
            h_res["discovered_relationships"],
            h_res["answer"]
        )

        h_recalls.append(h_metrics["recall_at_k"])
        h_precisions.append(h_metrics["precision_at_k"])
        h_rel_recalls.append(h_metrics["relationship_recall"])
        h_groundedness.append(h_metrics["groundedness"])
        h_latencies.append(h_res["latency_sec"])

        print(f"  -> Vector-Only: Recall={v_metrics['recall_at_k']}, Prec={v_metrics['precision_at_k']}, RelRecall={v_metrics['relationship_recall']}, Latency={v_res['latency_sec']}s")
        print(f"  -> Hybrid-RAG : Recall={h_metrics['recall_at_k']}, Prec={h_metrics['precision_at_k']}, RelRecall={h_metrics['relationship_recall']}, Latency={h_res['latency_sec']}s")

        results.append({
            "id": qid,
            "question": q,
            "repo": repo,
            "target_type": item["target_type"],
            "expected_source_ids": exp_sources,
            "expected_relationship": exp_rel,
            "vector_only": {
                "latency_sec": v_res["latency_sec"],
                "docs_count": v_res["docs_retrieved"],
                "retrieved_source_ids": v_res["retrieved_source_ids"],
                "metrics": v_metrics
            },
            "hybrid_graph_rag": {
                "latency_sec": h_res["latency_sec"],
                "docs_count": h_res["docs_retrieved"],
                "graph_entities_count": h_res["graph_entities_discovered"],
                "retrieved_source_ids": h_res["retrieved_source_ids"],
                "discovered_relationships": h_res["discovered_relationships"],
                "retrieval_mode": h_res["retrieval_mode"],
                "metrics": h_metrics
            }
        })

        time.sleep(2.5)

    n = len(EVALUATION_DATASET)
    summary = {
        "benchmark_metadata": {
            "total_queries_evaluated": n,
            "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "groq_model": "qwen/qwen3.8-27b",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
        },
        "aggregate_metrics": {
            "mean_recall_at_k": {
                "vector_only": round(sum(v_recalls) / n, 4),
                "hybrid_graph_rag": round(sum(h_recalls) / n, 4),
                "relative_gain_percent": round(((sum(h_recalls) - sum(v_recalls)) / (sum(v_recalls) or 1)) * 100, 2)
            },
            "mean_precision_at_k": {
                "vector_only": round(sum(v_precisions) / n, 4),
                "hybrid_graph_rag": round(sum(h_precisions) / n, 4)
            },
            "relationship_recall": {
                "vector_only": round(sum(v_rel_recalls) / n, 4),
                "hybrid_graph_rag": round(sum(h_rel_recalls) / n, 4)
            },
            "groundedness_score": {
                "vector_only": round(sum(v_groundedness) / n, 4),
                "hybrid_graph_rag": round(sum(h_groundedness) / n, 4)
            },
            "mean_latency_sec": {
                "vector_only": round(sum(v_latencies) / n, 3),
                "hybrid_graph_rag": round(sum(h_latencies) / n, 3)
            }
        },
        "detailed_results": results
    }

    report_path = BASE_DIR / "evaluation" / "evaluation_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=======================================================")
    print(f"Evaluation Complete! Summary saved to {report_path}")
    print(f"Mean Recall@k: Vector={summary['aggregate_metrics']['mean_recall_at_k']['vector_only']} | Hybrid={summary['aggregate_metrics']['mean_recall_at_k']['hybrid_graph_rag']}")
    print(f"Relationship Recall: Vector={summary['aggregate_metrics']['relationship_recall']['vector_only']} | Hybrid={summary['aggregate_metrics']['relationship_recall']['hybrid_graph_rag']}")
    print(f"Groundedness: Vector={summary['aggregate_metrics']['groundedness_score']['vector_only']} | Hybrid={summary['aggregate_metrics']['groundedness_score']['hybrid_graph_rag']}")
    print(f"Mean Latency: Vector={summary['aggregate_metrics']['mean_latency_sec']['vector_only']}s | Hybrid={summary['aggregate_metrics']['mean_latency_sec']['hybrid_graph_rag']}s")
    print(f"=======================================================\n")
    return summary


if __name__ == "__main__":
    key = os.getenv("GROQ_API_KEY")
    if not key:
        print("Error: GROQ_API_KEY environment variable is not set.")
    else:
        execute_evaluation(key)
