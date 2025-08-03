"""
External integrations package for fair credit scoring system.

This package provides integrations with external services including
GitHub, notification systems, authentication providers, and monitoring services.

Modules:
    - github: GitHub API integration and webhook handling
    - notifications: Email, Slack, and other notification services
    - auth: Authentication and authorization services
    - monitoring: External monitoring and alerting integrations
    - cloud: Cloud service integrations (AWS, GCP, Azure)
"""

__version__ = "0.2.0"

from .github.client import GitHubClient
from .notifications.email import EmailService
from .notifications.slack import SlackService
from .auth.oauth import OAuthService
from .monitoring.prometheus import PrometheusExporter

__all__ = [
    "GitHubClient",
    "EmailService", 
    "SlackService",
    "OAuthService",
    "PrometheusExporter"
]