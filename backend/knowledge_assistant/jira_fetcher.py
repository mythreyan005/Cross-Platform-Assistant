import os
import requests

def fetch_jira_issues(base_url, project_key, email=None, api_token=None, max_results=10):
    """
    Fetches issues from a Jira instance using an API Token.
    """
    if not email:
        email = os.getenv("JIRA_EMAIL")
    if not api_token:
        api_token = os.getenv("JIRA_API_TOKEN")

    all_issues = []
    jql = f"project={project_key}"
    search_url = f"{base_url}/rest/api/3/search/jql"
    
    params = {
        'jql': jql,
        'startAt': 0,
        'maxResults': max_results,
        'fields': 'summary,description,status'
    }
    
    # Setup authentication if provided
    auth = None
    if email and api_token:
        from requests.auth import HTTPBasicAuth
        auth = HTTPBasicAuth(email, api_token)
    
    try:
        print(f"Fetching up to {max_results} Jira issues from {project_key}...")
        # Pass the auth object to the request
        response = requests.get(search_url, params=params, auth=auth)
        
        # This will raise an error if the URL is bad or unauthorized
        response.raise_for_status()  
        
        data = response.json()
        issues = data.get('issues', [])
        
        for issue in issues:
            fields = issue.get('fields', {})
            
            # Extract basic fields
            key = issue.get('key')
            summary = fields.get('summary', '')
            status_name = fields.get('status', {}).get('name', '')
            
            # Jira API v3 returns descriptions in a complex format called Atlassian Document Format (ADF).
            # We need to dig into the JSON to extract the actual plain text.
            description_obj = fields.get('description')
            description_text = ""
            
            if isinstance(description_obj, dict) and 'content' in description_obj:
                for block in description_obj.get('content', []):
                    for content_node in block.get('content', []):
                        if content_node.get('type') == 'text':
                            description_text += content_node.get('text', '') + " "
            elif isinstance(description_obj, str):
                description_text = description_obj
            
            # Create a clean dictionary for our database/AI
            clean_issue = {
                'key': key,
                'summary': summary,
                'status': status_name,
                'description': description_text.strip()
            }
            all_issues.append(clean_issue)
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Jira: {e}")
        
    return all_issues

# Testing the function directly when we run this file
if __name__ == "__main__":
    BASE_URL = "https://lf-opendaylight.atlassian.net"  # Or your own Jira URL
    PROJECT_KEY = "TRNSPRTPCE"
    
    # Atlassian now blocks unauthenticated API access. You must use a token.
    JIRA_EMAIL = "mythreyansk@gmail.com"
    JIRA_API_TOKEN = "ATATT3xFfGF0TLr1k0yKk9Kg3poj2ZJZkOoadMNik4KJdM1ZVKmu-kI1FGZG7ln6_UUdytDp68XWnLY-ygrRcHvSKwlNJvBe0_fpSFZ5KSybP8rJ6bjpk0jIANgU6wCZ76E8KH50n3VfRYUeSmQVFRWZua_JmLl45D01c8QHeDgxS9nvSl6nHjM=DC885D54"
    
    print("Starting Jira script...")
    issues = fetch_jira_issues(BASE_URL, PROJECT_KEY, JIRA_EMAIL, JIRA_API_TOKEN, max_results=5)
    
    print(f"\nSuccessfully fetched {len(issues)} issues!\n")
    for iss in issues:
        print(f"[{iss['key']}] Status: {iss['status']}")
        print(f"Summary: {iss['summary']}")
        print(f"Description snippet: {iss['description'][:100]}...\n")
        print("-" * 40)
