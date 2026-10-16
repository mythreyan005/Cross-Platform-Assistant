"""
API Views for Cross-Platform Knowledge Assistant.
Provides endpoints for chat queries, project sync status, and admin re-indexing.
"""

import os
import logging
from django.conf import settings
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.throttling import AnonRateThrottle, UserRateThrottle, ScopedRateThrottle

try:
    from .rag_pipeline import ask_ai_hybrid, get_project_sync_status, load_and_index_data, normalize_repo
except ImportError:
    from rag_pipeline import ask_ai_hybrid, get_project_sync_status, load_and_index_data, normalize_repo

logger = logging.getLogger(__name__)


class ChatAPIView(APIView):
    """
    Core Chat Endpoint for Hybrid Graph-RAG queries.
    Accepts question and repository, retrieves hybrid evidence, and returns structured answer.
    """
    permission_classes = [AllowAny]
    throttle_classes = [AnonRateThrottle, UserRateThrottle]

    def post(self, request):
        try:
            groq_key = os.getenv("GROQ_API_KEY") or getattr(settings, "GROQ_API_KEY", "")
            if not groq_key:
                logger.error("GROQ_API_KEY is missing from environment.")
                return Response(
                    {"error": "AI service configuration is missing on the server."},
                    status=status.HTTP_503_SERVICE_UNAVAILABLE
                )

            # Validate payload
            message = request.data.get("message", "").strip()
            target_repo = request.data.get("repo", "dhis2/dhis2-core").strip()

            if not message:
                return Response(
                    {"error": "A non-empty 'message' field is required."},
                    status=status.HTTP_400_BAD_REQUEST
                )

            if len(message) > 1000:
                return Response(
                    {"error": "Message exceeds the maximum permitted length of 1000 characters."},
                    status=status.HTTP_400_BAD_REQUEST
                )

            repo_key = normalize_repo(target_repo, message)
            logger.info(f"Processing chat query for [{repo_key}]: '{message[:60]}...'")

            # Execute Hybrid Graph-RAG retrieval
            result = ask_ai_hybrid(
                question=message,
                groq_api_key=groq_key,
                repo=repo_key
            )

            return Response(result, status=status.HTTP_200_OK)

        except ValueError as ve:
            logger.warning(f"Validation error in chat API: {ve}")
            return Response({"error": str(ve)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error(f"Internal error processing chat query: {e}", exc_info=False)
            return Response(
                {"error": "An error occurred while generating the answer. Please try again later."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class StatusAPIView(APIView):
    """
    Verification endpoint reporting real synchronization state and graph metrics for projects.
    """
    permission_classes = [AllowAny]

    def get(self, request):
        repo = request.query_params.get("repo", "dhis2/dhis2-core")
        try:
            status_data = get_project_sync_status(repo)
            return Response(status_data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error fetching sync status: {e}")
            return Response(
                {"error": "Unable to check sync status."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ReindexAPIView(APIView):
    """
    Protected endpoint to trigger multi-source ingestion and Knowledge Graph rebuild.
    """
    permission_classes = [IsAuthenticated]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = 'indexing'

    def post(self, request):
        repo = request.data.get("repo", "dhis2/dhis2-core")
        try:
            logger.info(f"User {request.user.username} triggered re-indexing for {repo}")
            result = load_and_index_data(repo=repo)
            return Response(
                {"message": "Re-indexing completed successfully.", "details": result},
                status=status.HTTP_200_OK
            )
        except Exception as e:
            logger.error(f"Error during re-indexing: {e}", exc_info=True)
            return Response(
                {"error": "Failed to re-index repository."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# Legacy functional view wrapper for backwards compatibility
chat_api = ChatAPIView.as_view()
