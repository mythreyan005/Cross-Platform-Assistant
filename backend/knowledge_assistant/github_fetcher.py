import os
import re
from github import Github
from github.GithubException import RateLimitExceededException

def fetch_github_commits(repo_name, github_token=None, max_commits=20):
    """
    Fetches recent commits from a GitHub repository and extracts Jira keys from the commit messages.
    """
    if github_token is None:
        github_token = os.getenv("GITHUB_TOKEN")
    all_commits = []
    
    # Regex pattern for Jira keys (e.g., TRNSPRTPCE-980, DHIS2-20751)
    # Matches 2 or more uppercase letters/numbers, a hyphen, and 1 or more numbers
    jira_regex = re.compile(r'[A-Z][A-Z0-9]+-[0-9]+')
    
    # Authenticate with GitHub (using token if provided, else anonymous)
    # Note: Anonymous API calls are heavily rate-limited by GitHub (60 requests/hour).
    if github_token:
        g = Github(github_token, retry=0)
    else:
        g = Github(retry=0)
        
    try:
        print(f"Fetching up to {max_commits} commits from GitHub repo: {repo_name}...")
        repo = g.get_repo(repo_name)
        commits = repo.get_commits()
        
        count = 0
        for commit in commits:
            if count >= max_commits:
                break
                
            message = commit.commit.message
            sha = commit.sha
            author = commit.commit.author.name if commit.commit.author else "Unknown"
            
            # Find all Jira keys in the commit message
            # set() removes duplicates if the developer typed the same key twice
            jira_keys = list(set(jira_regex.findall(message)))
            
            clean_commit = {
                'sha': sha[:7],  # Short SHA for readability
                'author': author,
                'message': message,
                'linked_jira_keys': jira_keys
            }
            
            all_commits.append(clean_commit)
            count += 1
            
    except RateLimitExceededException:
        print("GitHub API rate limit exceeded! You need to provide a GitHub Personal Access Token.")
    except Exception as e:
        print(f"Error fetching data from GitHub: {e}")
        
    return all_commits

if __name__ == "__main__":
    # You can change this to "dhis2/dhis2-core" later!
    REPO_NAME = "opendaylight/transportpce"
    
    # Optional: Put your GitHub token here if you get Rate Limit errors
    GITHUB_TOKEN = None 
    
    print("Starting GitHub script...")
    commits = fetch_github_commits(REPO_NAME, GITHUB_TOKEN, max_commits=10)
    
    print(f"\nSuccessfully fetched {len(commits)} commits!\n")
    for c in commits:
        print(f"Commit [{c['sha']}] by {c['author']}")
        
        # This is where the magic happens! We show which Jira ticket this code belongs to.
        if c['linked_jira_keys']:
            print(f"🔗 Linked to Jira: {c['linked_jira_keys']}")
        else:
            print("🔗 Linked to Jira: None")
            
        # Print just the first line of the commit message to keep the output clean
        first_line = c['message'].split('\n')[0]
        print(f"Message: {first_line}\n")
        print("-" * 40)
