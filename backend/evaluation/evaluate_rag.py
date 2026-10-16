"""
Evaluation Framework: Comparative Benchmark of Vector-Only RAG vs. Hybrid Graph-RAG.
Targeted for ICITIIT 2027 Research Evaluation.

Features:
1. Strict, type-aware canonical artifact ID normalization (no arbitrary substring matching).
2. Structured graph edge relationship evaluation (source, target, and relationship type).
3. Clearly defined retrieval cutoff (top-k=5) and relevance judgments.
4. Documented evidence-groundedness assessment with clear protocol and limitations.
5. Explicit status tracking (SUCCESS, EXTRACTIVE_FALLBACK, EMPTY_RETRIEVAL, API_ERROR).
6. Identical corpora, models, and generation parameters across modalities.
7. Saves comprehensive per-query and aggregate experimental data.
"""

import os
import re
import time
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional

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


# ==============================================================================
# 1. CANONICAL ARTIFACT IDENTIFIER NORMALIZATION
# ==============================================================================

COMMIT_HEX_RE = re.compile(r'^[0-9a-f]{7,40}$', re.IGNORECASE)
JIRA_KEY_RE = re.compile(r'^[A-Z][A-Z0-9]+-[0-9]+$')
ENDPOINT_RE = re.compile(r'^(?:(GET|POST|PUT|DELETE|PATCH)\s+)?(/[^\s,;]*)$', re.IGNORECASE)


def normalize_artifact_id(raw_id: str) -> Tuple[str, str]:
    """
    Normalizes a raw artifact identifier into a canonical (type, canonical_value) tuple.
    Supported types:
    - 'commit': Lowercase hex prefix (first 7 characters).
    - 'issue': Uppercase Jira key (e.g. 'DHIS2-22296').
    - 'endpoint': Normalized method and path '{METHOD} {path}' (e.g. 'GET /api/dataValueSets').
    - 'component': Lowercase component name (e.g. 'tapi', '[api] other').
    - 'project': Canonical project key (e.g. 'dhis2/dhis2-core').
    """
    if not raw_id:
        return ("unknown", "")

    clean = raw_id.strip()

    # Prefixed forms
    if clean.startswith("commit:"):
        val = clean[7:].strip().lower()
        return ("commit", val[:7])
    if clean.startswith("issue:"):
        return ("issue", clean[6:].strip().upper())
    if clean.startswith("component:"):
        return ("component", clean[10:].strip().lower())
    if clean.startswith("project:"):
        return ("project", clean[8:].strip().lower())
    if clean.startswith("endpoint:"):
        rest = clean[9:].strip()
        if ":" in rest and not rest.startswith("/"):
            parts = rest.split(":", 1)
            method = parts[0].strip().upper()
            path = parts[1].strip()
        elif " " in rest:
            parts = rest.split(None, 1)
            method = parts[0].strip().upper()
            path = parts[1].strip()
        else:
            method, path = "ANY", rest.strip()
        return ("endpoint", f"{method} {path}")

    # Unprefixed heuristics
    if COMMIT_HEX_RE.match(clean):
        return ("commit", clean.lower()[:7])

    if JIRA_KEY_RE.match(clean.upper()):
        return ("issue", clean.upper())

    ep_match = ENDPOINT_RE.match(clean)
    if ep_match:
        m, p = ep_match.groups()
        method = m.upper() if m else "ANY"
        return ("endpoint", f"{method} {p}")

    # Fallback to component or general token
    return ("component", clean.lower())


def artifact_ids_match(id1: str, id2: str) -> bool:
    """
    Determines if two artifact identifiers correspond to the exact same canonical entity.
    Eliminates arbitrary substring matching.
    """
    t1, v1 = normalize_artifact_id(id1)
    t2, v2 = normalize_artifact_id(id2)

    if not v1 or not v2:
        return False

    if t1 != t2:
        return False

    if t1 == "endpoint":
        m1, p1 = v1.split(" ", 1)
        m2, p2 = v2.split(" ", 1)
        if m1 == "ANY" or m2 == "ANY":
            return p1 == p2
        return m1 == m2 and p1 == p2

    return v1 == v2


# ==============================================================================
# 2. BENCHMARK GROUND TRUTH DATASET
# ==============================================================================

EVALUATION_DATASET = [
    {
        "id": "Q1",
        "question": "Which GitHub commits are linked to Jira issue TRNSPRTPCE-980 in TransportPCE?",
        "repo": "opendaylight/transportpce",
        "expected_source_ids": ["TRNSPRTPCE-980", "de74a89", "e10ac2a"],
        "expected_relationship_edges": [
            {"source": "commit:de74a89", "target": "issue:TRNSPRTPCE-980", "relationship": "REFERENCES_ISSUE"},
            {"source": "commit:e10ac2a", "target": "issue:TRNSPRTPCE-980", "relationship": "REFERENCES_ISSUE"}
        ],
        "target_type": "commit_to_issue",
        "description": "Cross-platform linkage between Git commits and Jira issue TRNSPRTPCE-980."
    },
    {
        "id": "Q2",
        "question": "What API endpoints in DHIS2 handle analytical queries and data value sets?",
        "repo": "dhis2/dhis2-core",
        "expected_source_ids": ["GET /api/dataValueSets", "POST /api/dataValueSets", "GET /api/analytics"],
        "expected_relationship_edges": [
            {"source": "project:dhis2/dhis2-core", "target": "endpoint:GET:/api/dataValueSets", "relationship": "EXPOSES_ENDPOINT"},
            {"source": "project:dhis2/dhis2-core", "target": "endpoint:GET:/api/analytics", "relationship": "EXPOSES_ENDPOINT"}
        ],
        "target_type": "api_endpoints",
        "description": "Tests discovery of OpenAPI endpoints and operations in DHIS2."
    },
    {
        "id": "Q3",
        "question": "Which commits modified the tapi component in TransportPCE?",
        "repo": "opendaylight/transportpce",
        "expected_source_ids": ["de74a89", "2e7e995", "490a8fb"],
        "expected_relationship_edges": [
            {"source": "commit:de74a89", "target": "component:tapi", "relationship": "MODIFIES_COMPONENT"},
            {"source": "commit:2e7e995", "target": "component:tapi", "relationship": "MODIFIES_COMPONENT"}
        ],
        "target_type": "commit_to_component",
        "description": "Tests discovery of component-level modifications in commits."
    },
    {
        "id": "Q4",
        "question": "What user profile information and settings are accessible via the DHIS2 API?",
        "repo": "dhis2/dhis2-core",
        "expected_source_ids": ["GET /api/me", "GET /api/me/authorization"],
        "expected_relationship_edges": [
            {"source": "project:dhis2/dhis2-core", "target": "endpoint:GET:/api/me", "relationship": "EXPOSES_ENDPOINT"}
        ],
        "target_type": "api_endpoints",
        "description": "Tests specific endpoint documentation retrieval (/api/me)."
    },
    {
        "id": "Q5",
        "question": "What RESTCONF RPC operations and endpoints are available for path computation in TransportPCE?",
        "repo": "opendaylight/transportpce",
        "expected_source_ids": ["POST /restconf/operations/transportpce-pce:path-computation-request"],
        "expected_relationship_edges": [
            {"source": "endpoint:POST:/restconf/operations/transportpce-pce:path-computation-request", "target": "component:pce", "relationship": "RELATED_TO"}
        ],
        "target_type": "api_endpoints",
        "description": "Tests RPC operation and endpoint discovery for PCE services."
    },
    {
        "id": "Q6",
        "question": "What bug is reported in DHIS2-22296 regarding org unit merge and which API endpoints are involved?",
        "repo": "dhis2/dhis2-core",
        "expected_source_ids": ["DHIS2-22296", "POST /api/organisationUnits/merge"],
        "expected_relationship_edges": [
            {"source": "issue:DHIS2-22296", "target": "component:[api] other", "relationship": "RELATED_TO"}
        ],
        "target_type": "cross_source_issue_endpoint",
        "description": "Multi-source reasoning between Jira issue DHIS2-22296 and org unit merge endpoints."
    }
]


# ==============================================================================
# 3. METRIC CALCULATION ENGINE
# ==============================================================================

class ClaimVerificationProtocol:
    """
    Formal Claim Verification Protocol for ICITIIT 2027 Scientific Evaluation.
    Protocol Specification:
    1. Sentence Segmentation: Answers are segmented into discrete factual assertions.
    2. Evidence Corpus: Composed of retrieved text passages (ChromaDB) and verified graph triples (NetworkX).
    3. Technical Identifier Constraint: Every technical identifier (Jira issue key, commit SHA, endpoint path)
       mentioned in a claim MUST resolve to an identifier in the evidence corpus. An unverified identifier marks
       the claim as unsupported.
    4. Lexical/Semantic Verification: For claims without technical identifiers, the claim is checked for token
       overlap (significant words > 3 characters) against the evidence text (threshold >= 40%).
    5. Scoring: Groundedness = |Supported Claims| / |Total Claims| (strictly 0.0 if execution failed or empty).
    """
    PASS_OVERLAP_THRESHOLD = 0.40


def calculate_retrieval_metrics(
    expected_ids: List[str],
    retrieved_ids: List[str],
    cutoff_k: Optional[int] = 5
) -> Tuple[float, float, float, List[str]]:
    """
    Computes Recall@k, Precision@k, and F1@k using canonical identifier matching.
    Deduplicates multiple chunks of the same source record to prevent unfair metric inflation.
    
    Returns: (recall, precision, f1, matched_expected_ids)
    """
    if not expected_ids:
        # Vacuously complete if no entities expected
        return 1.0, 1.0, 1.0, []

    if not retrieved_ids:
        return 0.0, 0.0, 0.0, []

    # 1. Deduplicate retrieved source records canonically so multiple chunks of the same source record
    # do not unfairly inflate or distort retrieval statistics
    unique_retrieved = []
    for r in retrieved_ids:
        if not any(artifact_ids_match(r, u) for u in unique_retrieved):
            unique_retrieved.append(r)

    # 2. Apply cutoff k strictly to unique source records
    candidate_ids = unique_retrieved[:cutoff_k] if cutoff_k is not None else unique_retrieved

    # 3. Match against expected ground-truth IDs
    matched_expected = []
    for exp in expected_ids:
        for ret in candidate_ids:
            if artifact_ids_match(exp, ret):
                matched_expected.append(exp)
                break

    recall = round(len(matched_expected) / len(expected_ids), 4)
    precision = round(len(matched_expected) / len(candidate_ids), 4) if candidate_ids else 0.0
    f1 = round((2 * precision * recall) / (precision + recall), 4) if (precision + recall) > 0 else 0.0

    return recall, precision, f1, matched_expected


def evaluate_relationship_recall(
    expected_edges: List[Dict[str, str]],
    retrieved_edges: List[Dict[str, Any]],
    edge_budget: int = 30
) -> Tuple[float, List[Dict[str, str]]]:
    """
    Evaluates relationship recall using actual structured graph edges and their relationship types.
    Enforces a strict edge budget constraint (default 30 edges).
    """
    if not expected_edges:
        return 1.0, []

    if not retrieved_edges:
        return 0.0, []

    bounded_retrieved = retrieved_edges[:edge_budget]

    matched_edges = []
    for exp in expected_edges:
        exp_src = exp["source"]
        exp_tgt = exp["target"]
        exp_rel = exp["relationship"]

        found = False
        for ret in bounded_retrieved:
            ret_src = ret.get("source", "")
            ret_tgt = ret.get("target", "")
            ret_rel = ret.get("relationship", "")

            if (
                artifact_ids_match(exp_src, ret_src)
                and artifact_ids_match(exp_tgt, ret_tgt)
                and exp_rel == ret_rel
            ):
                found = True
                break

        if found:
            matched_edges.append(exp)

    recall = round(len(matched_edges) / len(expected_edges), 4)
    return recall, matched_edges


def evaluate_groundedness(
    answer_text: str,
    retrieved_source_ids: List[str],
    graph_edges: List[Dict[str, Any]],
    evidence_texts: Optional[List[str]] = None,
    status: str = "SUCCESS"
) -> Dict[str, Any]:
    """
    Evidence-based groundedness assessment.
    Eliminates all hardcoded scores.
    API failures and empty retrievals receive 0.0.
    Extractive fallbacks and LLM answers are actively decomposed into claims and verified against evidence.
    """
    if status in ("API_ERROR", "EMPTY_RETRIEVAL"):
        return {
            "score": 0.0,
            "total_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": 0,
            "contradicted_claims": 0,
            "method": "automated_claim_verification",
            "notes": f"Zero groundedness assigned due to status={status}."
        }

    if not answer_text or not answer_text.strip():
        return {
            "score": 0.0,
            "total_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": 0,
            "contradicted_claims": 0,
            "method": "automated_claim_verification",
            "notes": "Empty answer text."
        }

    # 1. Segment answer into sentence-level claims
    raw_sentences = re.split(r'(?<=[.!?])\s+|\n+', answer_text)
    claims = []
    for s in raw_sentences:
        clean_s = s.strip()
        clean_s = re.sub(r'^[#*\-•\d\.\s]+', '', clean_s).strip()
        if len(clean_s) > 15 and not clean_s.startswith("http"):
            claims.append(clean_s)

    if not claims:
        return {
            "score": 0.0,
            "total_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": 0,
            "contradicted_claims": 0,
            "method": "automated_claim_verification",
            "notes": "No substantive claims extracted from answer."
        }

    # 2. Build canonical evidence pool
    evidence_pool: Set[str] = set()
    for sid in retrieved_source_ids:
        t, v = normalize_artifact_id(sid)
        if v:
            evidence_pool.add(v.lower())
    for e in graph_edges:
        s_t, s_v = normalize_artifact_id(e.get("source", ""))
        t_t, t_v = normalize_artifact_id(e.get("target", ""))
        if s_v:
            evidence_pool.add(s_v.lower())
        if t_v:
            evidence_pool.add(t_v.lower())

    full_evidence_text = " ".join(evidence_texts or []).lower()
    for e in graph_edges:
        full_evidence_text += f" {e.get('source', '')} {e.get('relationship', '')} {e.get('target', '')}".lower()

    supported_count = 0
    unsupported_count = 0

    for claim in claims:
        claim_entities = extract_entities_from_text(claim)
        mentioned = (
            claim_entities.get("jira_keys", []) +
            claim_entities.get("commit_shas", []) +
            claim_entities.get("endpoints", [])
        )

        if mentioned:
            # All technical identifiers in the claim must be confirmed in evidence
            all_supported = True
            for ent in mentioned:
                _, ent_val = normalize_artifact_id(ent)
                if not any(artifact_ids_match(ent, ev) or ent_val.lower() in ev for ev in evidence_pool):
                    all_supported = False
                    break
            if all_supported:
                supported_count += 1
            else:
                unsupported_count += 1
        else:
            # Claim without technical identifiers: verify significant token overlap against evidence
            claim_tokens = [w.lower() for w in re.findall(r'\b[a-zA-Z]{4,}\b', claim)]
            if not claim_tokens:
                unsupported_count += 1
                continue
            matched_tokens = sum(1 for tok in claim_tokens if tok in full_evidence_text)
            overlap_ratio = matched_tokens / len(claim_tokens)
            if overlap_ratio >= ClaimVerificationProtocol.PASS_OVERLAP_THRESHOLD:
                supported_count += 1
            else:
                unsupported_count += 1

    score = round(supported_count / len(claims), 4) if claims else 0.0
    return {
        "score": score,
        "total_claims": len(claims),
        "supported_claims": supported_count,
        "unsupported_claims": unsupported_count,
        "contradicted_claims": 0,
        "method": "automated_claim_verification"
    }


# ==============================================================================
# 4. BENCHMARK EXECUTION PIPELINE
# ==============================================================================

def call_groq_with_retry(chain: Any, inputs: Dict[str, Any], max_retries: int = 3, initial_delay: float = 4.0) -> Tuple[str, str]:
    """
    Invokes Groq LLM with exponential backoff on HTTP 429 rate limit errors.
    Returns: (answer_text, status_code)
    """
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            resp = chain.invoke(inputs)
            return resp.content, "SUCCESS"
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "Rate limit" in err_str:
                logger.warning(f"Groq 429 rate limit encountered on attempt {attempt + 1}. Backing off for {delay}s...")
                time.sleep(delay)
                delay *= 2.0
            else:
                logger.error(f"Groq API error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return f"API_ERROR: {e}", "API_ERROR"
                time.sleep(3.0)
    return "Rate limit exceeded after retries.", "EXTRACTIVE_FALLBACK"


def run_vector_only_rag(question: str, repo: str, groq_api_key: str, top_k: int = 5) -> Dict[str, Any]:
    """Runs baseline Vector-Only RAG without Knowledge Graph traversal."""
    start_time = time.time()
    repo_key = normalize_repo(repo, question)
    vs = get_vector_store(repo_key)

    docs = vs.similarity_search(question, k=top_k)
    context_str = "\n\n".join([f"[{d.metadata.get('source_type', 'doc').upper()}] {d.page_content}" for d in docs])
    evidence_texts = [d.page_content for d in docs]

    retrieved_source_ids = []
    for d in docs:
        sid = d.metadata.get("source_id", "")
        if sid:
            retrieved_source_ids.append(sid)

    if not docs:
        return {
            "method": "vector_only",
            "answer": "No relevant documents found in vector store.",
            "status": "EMPTY_RETRIEVAL",
            "latency_sec": round(time.time() - start_time, 3),
            "docs_retrieved": 0,
            "retrieved_source_ids": [],
            "graph_edges": [],
            "evidence_texts": []
        }

    llm = ChatGroq(temperature=0.0, model_name="qwen/qwen3.8-27b", groq_api_key=groq_api_key, max_tokens=250)
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"Answer strictly using the provided context for {repo_key}. Be factual and concise."),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    chain = prompt | llm

    answer_text, run_status = call_groq_with_retry(chain, {"context": context_str, "question": question})
    latency = time.time() - start_time

    return {
        "method": "vector_only",
        "answer": answer_text,
        "status": run_status,
        "latency_sec": round(latency, 3),
        "docs_retrieved": len(docs),
        "retrieved_source_ids": retrieved_source_ids,
        "graph_edges": [],
        "evidence_texts": evidence_texts
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
        model_name="qwen/qwen3.8-27b",
        top_k=top_k,
        graph_depth=2
    )

    res = hybrid.answer_question(question)
    latency = time.time() - start_time

    # Collect retrieved source IDs from documents and graph entities
    retrieved_source_ids = []
    for s in res.get("sources", []):
        sid = s.get("source_id") or s.get("id", "")
        if sid:
            retrieved_source_ids.append(sid)

    graph_edges = res.get("graph_edges", [])
    evidence_texts = res.get("evidence", [])

    return {
        "method": "hybrid_graph_rag",
        "answer": res.get("answer", ""),
        "status": res.get("status", "SUCCESS"),
        "latency_sec": round(latency, 3),
        "docs_retrieved": res.get("evidence_count", 0),
        "graph_nodes_count": res.get("graph_node_count", 0),
        "graph_edges_count": len(graph_edges),
        "retrieved_source_ids": retrieved_source_ids,
        "graph_edges": graph_edges,
        "evidence_texts": evidence_texts,
        "retrieval_mode": res.get("retrieval_mode", "vector_only")
    }


def execute_evaluation(groq_api_key: str) -> Dict[str, Any]:
    """
    Executes benchmark comparison on the evaluation dataset.
    Fair comparison at cutoff k=5 for both vector baseline and hybrid graph RAG.
    Tracks operational reliability and groundedness dynamically.
    """
    print(f"\n=======================================================")
    print(f"Executing Comparative Evaluation: Vector vs Hybrid RAG")
    print(f"Benchmark Dataset Size: {len(EVALUATION_DATASET)} queries")
    print(f"Fair Retrieval Cutoff: k = 5 documents (both modalities)")
    print(f"Graph Traversal Budget: max 25 nodes, max 30 edges")
    print(f"LLM Configuration: qwen/qwen3.8-27b (temperature=0.0)")
    print(f"=======================================================\n")

    results = []
    v_recalls_at_5, h_recalls_at_5 = [], []
    v_precisions_at_5, h_precisions_at_5 = [], []
    v_f1s_at_5, h_f1s_at_5 = [], []
    h_expanded_recalls = []
    v_rel_recalls, h_rel_recalls = [], []
    v_groundedness, h_groundedness = [], []
    v_latencies, h_latencies = [], []

    # Reliability counters
    v_status_counts = {"SUCCESS": 0, "EXTRACTIVE_FALLBACK": 0, "EMPTY_RETRIEVAL": 0, "API_ERROR": 0}
    h_status_counts = {"SUCCESS": 0, "EXTRACTIVE_FALLBACK": 0, "EMPTY_RETRIEVAL": 0, "API_ERROR": 0}

    for item in EVALUATION_DATASET:
        qid = item["id"]
        q = item["question"]
        repo = item["repo"]
        exp_sources = item["expected_source_ids"]
        exp_edges = item["expected_relationship_edges"]

        print(f"Evaluating {qid}: '{q}' [{repo}]...")

        # 1. Vector-Only RAG (cutoff k=5)
        v_res = run_vector_only_rag(q, repo, groq_api_key, top_k=5)
        v_status_counts[v_res["status"]] = v_status_counts.get(v_res["status"], 0) + 1
        v_recall, v_prec, v_f1, v_matched = calculate_retrieval_metrics(exp_sources, v_res["retrieved_source_ids"], cutoff_k=5)
        v_rel_recall, _ = evaluate_relationship_recall(exp_edges, v_res["graph_edges"], edge_budget=30)
        v_ground_res = evaluate_groundedness(v_res["answer"], v_res["retrieved_source_ids"], v_res["graph_edges"], v_res["evidence_texts"], v_res["status"])
        v_ground = v_ground_res["score"]

        v_recalls_at_5.append(v_recall)
        v_precisions_at_5.append(v_prec)
        v_f1s_at_5.append(v_f1)
        v_rel_recalls.append(v_rel_recall)
        v_groundedness.append(v_ground)
        v_latencies.append(v_res["latency_sec"])

        time.sleep(3.0)

        # 2. Hybrid Graph-RAG (FAIR COMPARISON: evaluated at cutoff k=5 for document level, plus context-wide recall)
        h_res = run_hybrid_graph_rag(q, repo, groq_api_key, top_k=5)
        h_status_counts[h_res["status"]] = h_status_counts.get(h_res["status"], 0) + 1
        h_recall_5, h_prec_5, h_f1_5, h_matched_5 = calculate_retrieval_metrics(exp_sources, h_res["retrieved_source_ids"], cutoff_k=5)
        h_exp_rec, h_exp_prec, h_exp_f1, h_exp_matched = calculate_retrieval_metrics(exp_sources, h_res["retrieved_source_ids"], cutoff_k=None)
        h_rel_recall, h_matched_edges = evaluate_relationship_recall(exp_edges, h_res["graph_edges"], edge_budget=30)
        h_ground_res = evaluate_groundedness(h_res["answer"], h_res["retrieved_source_ids"], h_res["graph_edges"], h_res["evidence_texts"], h_res["status"])
        h_ground = h_ground_res["score"]

        h_recalls_at_5.append(h_recall_5)
        h_precisions_at_5.append(h_prec_5)
        h_f1s_at_5.append(h_f1_5)
        h_expanded_recalls.append(h_exp_rec)
        h_rel_recalls.append(h_rel_recall)
        h_groundedness.append(h_ground)
        h_latencies.append(h_res["latency_sec"])

        print(f"  -> Vector-Only: Recall@5={v_recall}, Prec@5={v_prec}, F1@5={v_f1}, RelRecall={v_rel_recall}, Grounded={v_ground}, Latency={v_res['latency_sec']}s [{v_res['status']}]")
        print(f"  -> Hybrid-RAG : Recall@5={h_recall_5}, Prec@5={h_prec_5}, F1@5={h_f1_5}, ExpRecall={h_exp_rec}, RelRecall={h_rel_recall}, Grounded={h_ground}, Latency={h_res['latency_sec']}s [{h_res['status']}]")

        results.append({
            "id": qid,
            "question": q,
            "repo": repo,
            "target_type": item["target_type"],
            "expected_source_ids": exp_sources,
            "expected_relationship_edges": exp_edges,
            "vector_only": {
                "status": v_res["status"],
                "latency_sec": v_res["latency_sec"],
                "docs_count": v_res["docs_retrieved"],
                "retrieved_source_ids": v_res["retrieved_source_ids"],
                "matched_expected_ids": v_matched,
                "metrics": {
                    "recall_at_5": v_recall,
                    "precision_at_5": v_prec,
                    "f1_at_5": v_f1,
                    "relationship_recall": v_rel_recall,
                    "groundedness": v_ground,
                    "groundedness_breakdown": v_ground_res
                },
                "answer_snippet": v_res["answer"][:300]
            },
            "hybrid_graph_rag": {
                "status": h_res["status"],
                "retrieval_mode": h_res["retrieval_mode"],
                "latency_sec": h_res["latency_sec"],
                "docs_count": h_res["docs_retrieved"],
                "graph_nodes_count": h_res["graph_nodes_count"],
                "graph_edges_count": h_res["graph_edges_count"],
                "retrieved_source_ids": h_res["retrieved_source_ids"],
                "matched_expected_ids_at_5": h_matched_5,
                "matched_expected_ids_expanded": h_exp_matched,
                "matched_relationship_edges": h_matched_edges,
                "actual_graph_edges": h_res["graph_edges"],
                "metrics": {
                    "recall_at_5": h_recall_5,
                    "precision_at_5": h_prec_5,
                    "f1_at_5": h_f1_5,
                    "expanded_context_recall": h_exp_rec,
                    "relationship_recall": h_rel_recall,
                    "groundedness": h_ground,
                    "groundedness_breakdown": h_ground_res
                },
                "answer_snippet": h_res["answer"][:300]
            }
        })

        time.sleep(3.0)

    n = len(EVALUATION_DATASET)
    summary = {
        "benchmark_metadata": {
            "total_queries_evaluated": n,
            "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "groq_model": "qwen/qwen3.8-27b",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "document_retrieval_cutoff_k": 5,
            "graph_traversal_limits": {
                "max_depth": 2,
                "max_nodes": 25,
                "max_edges_budget": 30
            }
        },
        "operational_reliability": {
            "vector_only": {
                "success_count": v_status_counts["SUCCESS"],
                "fallback_count": v_status_counts["EXTRACTIVE_FALLBACK"],
                "empty_retrieval_count": v_status_counts["EMPTY_RETRIEVAL"],
                "api_error_count": v_status_counts["API_ERROR"],
                "success_rate": round(v_status_counts["SUCCESS"] / n, 4),
                "failure_rate": round((v_status_counts["EMPTY_RETRIEVAL"] + v_status_counts["API_ERROR"]) / n, 4)
            },
            "hybrid_graph_rag": {
                "success_count": h_status_counts["SUCCESS"],
                "fallback_count": h_status_counts["EXTRACTIVE_FALLBACK"],
                "empty_retrieval_count": h_status_counts["EMPTY_RETRIEVAL"],
                "api_error_count": h_status_counts["API_ERROR"],
                "success_rate": round(h_status_counts["SUCCESS"] / n, 4),
                "failure_rate": round((h_status_counts["EMPTY_RETRIEVAL"] + h_status_counts["API_ERROR"]) / n, 4)
            }
        },
        "aggregate_metrics": {
            "retrieval_at_cutoff_k5": {
                "mean_recall_at_5": {
                    "vector_only": round(sum(v_recalls_at_5) / n, 4),
                    "hybrid_graph_rag": round(sum(h_recalls_at_5) / n, 4)
                },
                "mean_precision_at_5": {
                    "vector_only": round(sum(v_precisions_at_5) / n, 4),
                    "hybrid_graph_rag": round(sum(h_precisions_at_5) / n, 4)
                },
                "mean_f1_at_5": {
                    "vector_only": round(sum(v_f1s_at_5) / n, 4),
                    "hybrid_graph_rag": round(sum(h_f1s_at_5) / n, 4)
                }
            },
            "graph_relationship_reasoning": {
                "relationship_recall_budget_30": {
                    "vector_only": round(sum(v_rel_recalls) / n, 4),
                    "hybrid_graph_rag": round(sum(h_rel_recalls) / n, 4)
                }
            },
            "expanded_context_retrieval": {
                "hybrid_expanded_context_recall": round(sum(h_expanded_recalls) / n, 4)
            },
            "answer_groundedness": {
                "mean_groundedness_score": {
                    "vector_only": round(sum(v_groundedness) / n, 4),
                    "hybrid_graph_rag": round(sum(h_groundedness) / n, 4)
                }
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
    print(f"Recall@5: Vector={summary['aggregate_metrics']['retrieval_at_cutoff_k5']['mean_recall_at_5']['vector_only']} | Hybrid={summary['aggregate_metrics']['retrieval_at_cutoff_k5']['mean_recall_at_5']['hybrid_graph_rag']}")
    print(f"Expanded Recall: Hybrid={summary['aggregate_metrics']['expanded_context_retrieval']['hybrid_expanded_context_recall']}")
    print(f"Relationship Recall (budget=30): Vector={summary['aggregate_metrics']['graph_relationship_reasoning']['relationship_recall_budget_30']['vector_only']} | Hybrid={summary['aggregate_metrics']['graph_relationship_reasoning']['relationship_recall_budget_30']['hybrid_graph_rag']}")
    print(f"Groundedness: Vector={summary['aggregate_metrics']['answer_groundedness']['mean_groundedness_score']['vector_only']} | Hybrid={summary['aggregate_metrics']['answer_groundedness']['mean_groundedness_score']['hybrid_graph_rag']}")
    print(f"Mean Latency: Vector={summary['aggregate_metrics']['mean_latency_sec']['vector_only']}s | Hybrid={summary['aggregate_metrics']['mean_latency_sec']['hybrid_graph_rag']}s")
    print(f"=======================================================\n")
    return summary


if __name__ == "__main__":
    key = os.getenv("GROQ_API_KEY")
    if not key:
        print("Error: GROQ_API_KEY environment variable is not set.")
    else:
        execute_evaluation(key)
