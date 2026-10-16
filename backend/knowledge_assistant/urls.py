"""
URL Configuration for knowledge_assistant project.
Maps chat, status, re-indexing, and authentication endpoints.
"""

from django.contrib import admin
from django.urls import path
from knowledge_assistant.views import ChatAPIView, StatusAPIView, ReindexAPIView
from knowledge_assistant.auth_views import RegisterView, LoginView, LogoutView, UserMeView

urlpatterns = [
    path('admin/', admin.site.urls),

    # Core Knowledge Assistant Endpoints
    path('api/chat/', ChatAPIView.as_view(), name='chat_api'),
    path('api/status/', StatusAPIView.as_view(), name='status_api'),
    path('api/reindex/', ReindexAPIView.as_view(), name='reindex_api'),

    # Authentication & Authorization Endpoints
    path('api/auth/register/', RegisterView.as_view(), name='auth_register'),
    path('api/auth/login/', LoginView.as_view(), name='auth_login'),
    path('api/auth/logout/', LogoutView.as_view(), name='auth_logout'),
    path('api/auth/me/', UserMeView.as_view(), name='auth_me'),
]
