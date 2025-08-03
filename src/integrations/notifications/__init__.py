"""
Notification services module.

Provides email, Slack, and other notification integrations for
alerting and communication purposes.
"""

__version__ = "0.2.0"

from .email import EmailService
from .slack import SlackService
from .base import NotificationService

__all__ = ["EmailService", "SlackService", "NotificationService"]