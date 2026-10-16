"""
Integration tests for authentication, authorization, and API endpoints.
"""

from django.test import TestCase
from django.contrib.auth.models import User
from rest_framework.test import APIClient
from rest_framework import status
from rest_framework.authtoken.models import Token


class AuthAndAPITests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            username="testengineer",
            email="test@example.com",
            password="StrongPassword123!"
        )
        self.token = Token.objects.create(user=self.user)

    def test_user_registration_success(self):
        payload = {
            "username": "newdev",
            "email": "newdev@example.com",
            "password": "Password789!"
        }
        res = self.client.post("/api/auth/register/", payload, format="json")
        self.assertEqual(res.status_code, status.HTTP_201_CREATED)
        self.assertIn("token", res.data)
        self.assertEqual(res.data["user"]["username"], "newdev")

    def test_user_registration_validation(self):
        # Short password
        res = self.client.post("/api/auth/register/", {"username": "abc", "password": "123"}, format="json")
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)

        # Duplicate username
        res = self.client.post("/api/auth/register/", {"username": "testengineer", "password": "ValidPass123!"}, format="json")
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)

    def test_user_login_success(self):
        payload = {"username": "testengineer", "password": "StrongPassword123!"}
        res = self.client.post("/api/auth/login/", payload, format="json")
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertIn("token", res.data)
        self.assertEqual(res.data["token"], self.token.key)

    def test_user_login_failure(self):
        payload = {"username": "testengineer", "password": "WrongPassword"}
        res = self.client.post("/api/auth/login/", payload, format="json")
        self.assertEqual(res.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_user_logout_and_invalidation(self):
        # Unauthenticated logout rejected
        res = self.client.post("/api/auth/logout/")
        self.assertEqual(res.status_code, status.HTTP_401_UNAUTHORIZED)

        # Authenticated logout deletes token
        self.client.credentials(HTTP_AUTHORIZATION=f"Token {self.token.key}")
        res = self.client.post("/api/auth/logout/")
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertFalse(Token.objects.filter(key=self.token.key).exists())

    def test_status_api(self):
        res = self.client.get("/api/status/?repo=dhis2/dhis2-core")
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertIn("is_synced", res.data)
        self.assertIn("graph_nodes_count", res.data)

    def test_chat_api_validation(self):
        # Empty message
        res = self.client.post("/api/chat/", {"message": ""}, format="json")
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)

        # Message too long
        long_msg = "x" * 1005
        res = self.client.post("/api/chat/", {"message": long_msg}, format="json")
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)

    def test_reindex_api_requires_auth(self):
        # Unauthenticated request must be rejected
        res = self.client.post("/api/reindex/", {"repo": "dhis2/dhis2-core"}, format="json")
        self.assertEqual(res.status_code, status.HTTP_401_UNAUTHORIZED)
