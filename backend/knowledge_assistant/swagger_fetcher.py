import requests

def fetch_swagger_docs(swagger_url):
    """
    Fetches an OpenAPI/Swagger JSON file and extracts the endpoints and their descriptions.
    """
    all_endpoints = []
    
    try:
        print(f"Fetching API documentation from: {swagger_url}...")
        
        # We need a headers dictionary to mimic a browser, sometimes raw JSON links block Python
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(swagger_url, headers=headers)
        response.raise_for_status()
        
        swagger_data = response.json()
        paths = swagger_data.get('paths', {})
        
        for path, methods in paths.items():
            for method, details in methods.items():
                clean_endpoint = {
                    'endpoint': f"{method.upper()} {path}",
                    'summary': details.get('summary', 'No summary provided'),
                    'description': details.get('description', 'No description provided')
                }
                all_endpoints.append(clean_endpoint)
                
    except Exception as e:
        print(f"Error fetching Swagger docs: {e}")
        
    return all_endpoints

if __name__ == "__main__":
    # Note: The Sonatype link your friend sent is a package repository page, not the raw JSON file. 
    # To parse Swagger, we need the raw JSON. Here is a standard public Swagger JSON as an example:
    RAW_SWAGGER_URL = "https://petstore.swagger.io/v2/swagger.json"
    
    print("Starting Swagger script...")
    endpoints = fetch_swagger_docs(RAW_SWAGGER_URL)
    
    print(f"\nSuccessfully fetched {len(endpoints)} API endpoints!\n")
    
    # Just print the first 5 so we don't flood the terminal
    for ep in endpoints[:5]:
        print(f"[{ep['endpoint']}]")
        print(f"Summary: {ep['summary']}")
        print(f"Description: {ep['description']}\n")
        print("-" * 40)
