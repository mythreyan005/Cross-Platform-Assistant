import os
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Import the fetchers
try:
    from .jira_fetcher import fetch_jira_issues
    from .github_fetcher import fetch_github_commits
except ImportError:
    from jira_fetcher import fetch_jira_issues
    from github_fetcher import fetch_github_commits

BASE_DIR = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / '.env')
except ImportError:
    pass

# Free local HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

PROJECT_CONFIGS = {
    "dhis2/dhis2-core": {
        "name": "DHIS2 Core Platform",
        "jira_url": "https://dhis2.atlassian.net",
        "jira_project": "DHIS2",
        "github_repo": "dhis2/dhis2-core",
        "chroma_dir": str(BASE_DIR / "chroma_db_dhis2"),
        "swagger_endpoints": [
            "API Endpoint: GET /api/dataValueSets. Description: Queries and imports data value sets for aggregated and tracker health data.",
            "API Endpoint: GET /api/analytics. Description: Analytical reporting engine across dimensions (data elements, indicators, periods, organisation units).",
            "API Endpoint: GET /api/organisationUnits. Description: Manages administrative organizational hierarchy, parent-child relationships, and geo-location coordinates.",
            "API Endpoint: GET /api/programs. Description: Configures clinical health tracker programs, event stages, program rules, and program indicators.",
            "API Endpoint: GET /api/trackedEntityInstances. Description: Handles tracked entity records for individuals, patients, and disease surveillance.",
            "API Endpoint: GET /api/events. Description: Records and queries single-event health interventions without repeated tracker enrollments.",
            "API Endpoint: GET /api/dataElements. Description: Metadata repository of health indicators, categories, and raw data element definitions.",
            "API Endpoint: GET /api/me. Description: Returns the profile, authorities, user groups, and credentials of the authenticated user."
        ]
    },
    "opendaylight/transportpce": {
        "name": "TransportPCE (OpenDaylight)",
        "jira_url": "https://lf-opendaylight.atlassian.net",
        "jira_project": "TRNSPRTPCE",
        "github_repo": "opendaylight/transportpce",
        "chroma_dir": str(BASE_DIR / "chroma_db_transportpce"),
        "swagger_endpoints": [
            "API Endpoint: GET /api/v1/network. Description: Retrieves current optical fiber network topology, ROADM node configurations, and optical link connectivity.",
            "API Endpoint: POST /api/v1/service-path. Description: Calculates path computation element (PCE) optical wavelength routes, cross-connects, and port mappings.",
            "API Endpoint: GET /restconf/config/ietf-network:networks. Description: OpenDaylight RESTCONF endpoint for optical software-defined networking topology models.",
            "API Endpoint: POST /restconf/operations/transportpce-pce:path-computation-request. Description: Invokes PCE path calculation algorithm for optical transport services."
        ]
    }
}

def normalize_repo(repo, question=""):
    """
    Intelligently determines the target project.
    If the question specifically refers to a project by name, prioritizes that project.
    Otherwise, defaults to the user's selected repository.
    """
    q = (question or "").lower()
    has_dhis = "dhis" in q
    has_transport = "transport" in q or "opendaylight" in q

    # Question specifically requests a project
    if has_dhis and not has_transport:
        return "dhis2/dhis2-core"
    elif has_transport and not has_dhis:
        return "opendaylight/transportpce"

    # Default to selected repo
    if not repo:
        return "dhis2/dhis2-core"
    r = repo.lower().strip()
    if "transport" in r or "opendaylight" in r:
        return "opendaylight/transportpce"
    return "dhis2/dhis2-core"

def load_and_index_data(repo="dhis2/dhis2-core"):
    """
    Pulls data from Jira, GitHub, and Swagger for the specified repository and saves into ChromaDB.
    """
    repo_key = normalize_repo(repo)
    config = PROJECT_CONFIGS[repo_key]
    chroma_dir = config["chroma_dir"]

    print(f"\n==================================================")
    print(f"Indexing data for project: {config['name']} ({repo_key})")
    print(f"Target vector store: {chroma_dir}")
    print(f"==================================================")

    db = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
    documents = []

    # 1. Fetch Jira Issues
    print(f"Fetching Jira Issues for {config['name']} ({config['jira_project']})...")
    try:
        jira_issues = fetch_jira_issues(config["jira_url"], config["jira_project"], max_results=10)
        for issue in jira_issues:
            content = (
                f"Project: {config['name']}\n"
                f"Jira Ticket {issue['key']}: {issue['summary']}\n"
                f"Status: {issue['status']}\n"
                f"Description: {issue['description']}"
            )
            documents.append(Document(
                page_content=content,
                metadata={"source": "jira", "id": issue['key'], "repo": repo_key, "project": config['name']}
            ))
        print(f"Fetched and parsed {len(jira_issues)} Jira issues.")
    except Exception as e:
        print(f"Warning: Skipping Jira due to error: {e}")

    # 2. Fetch GitHub Commits
    print(f"Fetching GitHub Commits for {config['github_repo']}...")
    try:
        commits = fetch_github_commits(config["github_repo"], max_commits=15)
        for commit in commits:
            jira_links = ", ".join(commit['linked_jira_keys']) if commit['linked_jira_keys'] else "None"
            content = (
                f"Project: {config['name']} ({config['github_repo']})\n"
                f"GitHub Commit {commit['sha']} by {commit['author']}\n"
                f"Message: {commit['message']}\n"
                f"Linked Jira Tickets: {jira_links}"
            )
            documents.append(Document(
                page_content=content,
                metadata={"source": "github", "id": commit['sha'], "repo": repo_key, "project": config['name']}
            ))
        print(f"Fetched {len(commits)} GitHub commits.")
    except Exception as e:
        print(f"Warning: Skipping GitHub due to error: {e}")

    # 3. Read Swagger / API Documentation
    print(f"Adding API Documentation specifications for {config['name']}...")
    for endpoint in config["swagger_endpoints"]:
        content = f"Project: {config['name']} ({repo_key})\n{endpoint}"
        documents.append(Document(
            page_content=content,
            metadata={"source": "swagger", "repo": repo_key, "project": config['name']}
        ))

    print(f"Adding total of {len(documents)} documents to ChromaDB at {chroma_dir}...")
    db.add_documents(documents)
    print(f"Successfully saved {config['name']} data to ChromaDB!\n")

def ask_ai(question, groq_api_key, repo="dhis2/dhis2-core"):
    """
    Takes a user question, searches project-specific ChromaDB, and asks Groq LLM for the answer.
    """
    repo_key = normalize_repo(repo, question)
    config = PROJECT_CONFIGS[repo_key]
    chroma_dir = config["chroma_dir"]

    # Ensure database exists and is populated
    if not os.path.exists(chroma_dir) or not os.listdir(chroma_dir):
        print(f"ChromaDB for {config['name']} is empty or not found. Populating now...")
        load_and_index_data(repo_key)

    db = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 5})

    llm = ChatGroq(
        temperature=0,
        model_name="qwen/qwen3.8-27b",
        groq_api_key=groq_api_key
    )

    system_prompt = (
        f"You are a Software Engineering Knowledge Assistant specialized in {config['name']} ({repo_key}). "
        f"Answer the user's question clearly, thoroughly, and informatively using the provided context, which contains "
        f"GitHub commits, Jira issues, and API documentation for {config['name']}. "
        "Summarize relevant bug fixes, code changes, features, or architecture endpoints clearly. "
        "If the context provides relevant details, extract and explain them nicely with commit hashes, Jira keys, or endpoints. "
        "Do not answer with a simple refusal unless there is genuinely no relevant information in the context."
        "\n\nContext:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print(f"Querying [{config['name']}] with question: '{question}'...")
    response = rag_chain.invoke({"input": question})
    return response["answer"]

if __name__ == "__main__":
    print("=== Indexing and Testing Both Projects ===")
    GROQ_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_KEY:
        print("Warning: GROQ_API_KEY is not set in environment or .env file.")

    # Index both projects so both databases are fully populated
    print("\n1. Indexing DHIS2...")
    load_and_index_data("dhis2/dhis2-core")

    print("\n2. Indexing TransportPCE...")
    load_and_index_data("opendaylight/transportpce")

    # Test TransportPCE query
    q1 = "Explain recent bug fixes and improvements in TransportPCE."
    ans1 = ask_ai(q1, GROQ_KEY, repo="opendaylight/transportpce")
    print("\n[AI Answer - TransportPCE]:")
    print(ans1)

    # Test DHIS2 query
    q2 = "Explain recent bug fixes and improvements in DHIS2."
    ans2 = ask_ai(q2, GROQ_KEY, repo="dhis2/dhis2-core")
    print("\n[AI Answer - DHIS2]:")
    print(ans2)
