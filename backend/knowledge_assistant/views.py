import os
import json
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Import the LangChain brain
from .rag_pipeline import ask_ai

@csrf_exempt
def chat_api(request):
    """
    This is the endpoint that the React frontend will talk to.
    It takes a user message and target repository, passes it to LangChain,
    and returns the AI's answer for that specific project.
    """
    if request.method == "POST":
        try:
            groq_key = os.getenv("GROQ_API_KEY") or getattr(settings, "GROQ_API_KEY", "")
            if not groq_key:
                return JsonResponse({"error": "GROQ_API_KEY is not configured in .env"}, status=500)
            
            # Parse JSON coming from React
            body = json.loads(request.body)
            user_message = body.get("message", "").strip()
            target_repo = body.get("repo", "dhis2/dhis2-core")
            
            if not user_message:
                return JsonResponse({"error": "No message provided"}, status=400)
                
            print(f"Received question for [{target_repo}]: '{user_message}'")
            
            # Ask the AI with the project repository context
            ai_answer = ask_ai(user_message, groq_key, repo=target_repo)
            
            # Return answer to frontend
            return JsonResponse({"answer": ai_answer, "repo": target_repo})
            
        except Exception as e:
            print(f"Error in chat API: {e}")
            return JsonResponse({"error": str(e)}, status=500)
            
    return JsonResponse({"error": "Only POST requests are allowed"}, status=405)
