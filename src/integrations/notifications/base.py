"""
Base notification service interface.

Provides abstract base class for notification services with
common functionality and standardized interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from ...logging_config import get_logger

logger = get_logger(__name__)


class NotificationPriority(Enum):
    """Notification priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationStatus(Enum):
    """Notification delivery status."""
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    DELIVERED = "delivered"


@dataclass
class Notification:
    """Notification message container."""
    id: str
    title: str
    message: str
    priority: NotificationPriority
    recipients: List[str]
    channels: List[str]
    metadata: Dict[str, Any]
    created_at: datetime
    status: NotificationStatus = NotificationStatus.PENDING
    error_message: Optional[str] = None
    delivered_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert notification to dictionary."""
        return {
            'id': self.id,
            'title': self.title,
            'message': self.message,
            'priority': self.priority.value,
            'recipients': self.recipients,
            'channels': self.channels,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'status': self.status.value,
            'error_message': self.error_message,
            'delivered_at': self.delivered_at.isoformat() if self.delivered_at else None
        }


class NotificationService(ABC):
    """
    Abstract base class for notification services.

    Provides common interface and functionality for different
    notification channels (email, Slack, SMS, etc.).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize notification service.

        Args:
            config: Service-specific configuration
        """
        self.config = config or {}
        self.service_name = self.__class__.__name__
        self._notification_history: List[Notification] = []

        logger.info(f"{self.service_name} initialized")

    @abstractmethod
    async def send_notification(
        self,
        notification: Notification
    ) -> bool:
        """
        Send notification through this service.

        Args:
            notification: Notification to send

        Returns:
            Success status
        """
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate service configuration.

        Returns:
            True if configuration is valid
        """
        pass

    def create_notification(
        self,
        title: str,
        message: str,
        recipients: List[str],
        priority: NotificationPriority = NotificationPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Notification:
        """
        Create a new notification.

        Args:
            title: Notification title
            message: Notification message
            recipients: List of recipients
            priority: Notification priority
            metadata: Additional metadata

        Returns:
            Notification object
        """
        import uuid

        notification = Notification(
            id=str(uuid.uuid4()),
            title=title,
            message=message,
            priority=priority,
            recipients=recipients,
            channels=[self.service_name],
            metadata=metadata or {},
            created_at=datetime.utcnow()
        )

        return notification

    async def send_simple_notification(
        self,
        title: str,
        message: str,
        recipients: List[str],
        priority: NotificationPriority = NotificationPriority.NORMAL
    ) -> bool:
        """
        Send a simple notification.

        Args:
            title: Notification title
            message: Notification message
            recipients: List of recipients
            priority: Notification priority

        Returns:
            Success status
        """
        notification = self.create_notification(title, message, recipients, priority)
        return await self.send_notification(notification)

    def get_notification_history(self, limit: Optional[int] = None) -> List[Notification]:
        """
        Get notification history.

        Args:
            limit: Maximum number of notifications to return

        Returns:
            List of notifications
        """
        history = sorted(self._notification_history, key=lambda n: n.created_at, reverse=True)

        if limit:
            history = history[:limit]

        return history

    def get_notification_stats(self) -> Dict[str, Any]:
        """
        Get notification statistics.

        Returns:
            Statistics dictionary
        """
        total = len(self._notification_history)

        if total == 0:
            return {
                'total': 0,
                'sent': 0,
                'failed': 0,
                'success_rate': 0.0
            }

        sent = len([n for n in self._notification_history if n.status == NotificationStatus.SENT])
        failed = len([n for n in self._notification_history if n.status == NotificationStatus.FAILED])

        return {
            'total': total,
            'sent': sent,
            'failed': failed,
            'success_rate': sent / total if total > 0 else 0.0,
            'service_name': self.service_name
        }

    def _update_notification_status(
        self,
        notification: Notification,
        status: NotificationStatus,
        error_message: Optional[str] = None
    ):
        """Update notification status and add to history."""
        notification.status = status
        notification.error_message = error_message

        if status in [NotificationStatus.SENT, NotificationStatus.DELIVERED]:
            notification.delivered_at = datetime.utcnow()

        # Add to history if not already present
        if notification not in self._notification_history:
            self._notification_history.append(notification)

        logger.info(f"Notification {notification.id} status updated to {status.value}")

    def format_message(
        self,
        template: str,
        variables: Dict[str, Any]
    ) -> str:
        """
        Format message using template variables.

        Args:
            template: Message template with placeholders
            variables: Variables to substitute

        Returns:
            Formatted message
        """
        try:
            return template.format(**variables)
        except KeyError as e:
            logger.error(f"Missing template variable: {e}")
            return template
        except Exception as e:
            logger.error(f"Template formatting error: {e}")
            return template
