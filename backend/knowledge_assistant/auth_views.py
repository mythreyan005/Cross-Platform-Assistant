"""
Authentication and Authorization Views for Cross-Platform Knowledge Assistant.
Provides user registration, token-based login, session invalidation, and profile checks.
"""

import logging
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.authtoken.models import Token

logger = logging.getLogger(__name__)


class RegisterView(APIView):
    """
    User Registration Endpoint.
    Creates a new user account with secure password hashing and returns an auth token.
    """
    permission_classes = [AllowAny]

    def post(self, request):
        username = request.data.get("username", "").strip()
        email = request.data.get("email", "").strip()
        password = request.data.get("password", "").strip()

        if not username or not password:
            return Response(
                {"error": "Username and password are required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        if len(password) < 8:
            return Response(
                {"error": "Password must be at least 8 characters long."},
                status=status.HTTP_400_BAD_REQUEST
            )

        if User.objects.filter(username__iexact=username).exists():
            return Response(
                {"error": "Username is already taken."},
                status=status.HTTP_400_BAD_REQUEST
            )

        if email and User.objects.filter(email__iexact=email).exists():
            return Response(
                {"error": "An account with this email already exists."},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            user = User.objects.create_user(username=username, email=email, password=password)
            token, _ = Token.objects.get_or_create(user=user)
            logger.info(f"New user registered: {username}")
            return Response(
                {
                    "message": "User registered successfully.",
                    "token": token.key,
                    "user": {
                        "id": user.id,
                        "username": user.username,
                        "email": user.email,
                    }
                },
                status=status.HTTP_201_CREATED
            )
        except Exception as e:
            logger.error(f"Error during registration: {e}")
            return Response(
                {"error": "An internal error occurred during registration."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class LoginView(APIView):
    """
    User Login Endpoint.
    Authenticates credentials and returns a secure authorization token.
    """
    permission_classes = [AllowAny]

    def post(self, request):
        username = request.data.get("username", "").strip()
        password = request.data.get("password", "").strip()

        if not username or not password:
            return Response(
                {"error": "Please provide both username and password."},
                status=status.HTTP_400_BAD_REQUEST
            )

        user = authenticate(username=username, password=password)
        if not user:
            return Response(
                {"error": "Invalid username or password."},
                status=status.HTTP_401_UNAUTHORIZED
            )

        token, _ = Token.objects.get_or_create(user=user)
        logger.info(f"User logged in: {username}")
        return Response(
            {
                "message": "Login successful.",
                "token": token.key,
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "is_staff": user.is_staff,
                }
            },
            status=status.HTTP_200_OK
        )


class LogoutView(APIView):
    """
    User Logout Endpoint.
    Invalidates the current session token on the server.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            # Delete token to invalidate
            request.user.auth_token.delete()
            return Response({"message": "Successfully logged out."}, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error during logout: {e}")
            return Response({"error": "Logout failed."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class UserMeView(APIView):
    """
    Profile inspection endpoint for the currently authenticated user.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        return Response({
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "is_staff": user.is_staff,
            "date_joined": user.date_joined.isoformat()
        })
