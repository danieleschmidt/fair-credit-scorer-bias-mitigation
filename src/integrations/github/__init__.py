"""
GitHub integration module.

Provides GitHub API client, webhook handlers, and GitHub Actions integration
for automated repository management and CI/CD workflows.
"""

__version__ = "0.2.0"

from .actions import ActionsService
from .client import GitHubClient
from .webhooks import WebhookHandler

__all__ = ["GitHubClient", "WebhookHandler", "ActionsService"]
