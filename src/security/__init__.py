"""
Security and compliance package for fair credit scoring.

This package provides comprehensive security features including
authentication, authorization, input validation, and compliance monitoring.

Modules:
    - authentication: User authentication and session management
    - authorization: Role-based access control (RBAC)
    - validation: Input validation and sanitization
    - compliance: Regulatory compliance monitoring and reporting
    - encryption: Data encryption and key management
    - audit: Security audit logging and monitoring
"""

__version__ = "0.2.0"

from .audit import AuditLog, EventType, SecurityAuditor, SecurityEvent, Severity
from .authentication import AuthenticationManager, Session, SessionManager, Token, User
from .authorization import (
    AccessRequest,
    AccessResult,
    Action,
    Permission,
    RBACManager,
    ResourceType,
    Role,
)
from .validation import (
    DataSanitizer,
    InputValidator,
    ValidationResult,
    ValidationRule,
    create_credit_score_validator,
)

__all__ = [
    # Authentication
    "AuthenticationManager",
    "SessionManager",
    "Token",
    "User",
    "Session",

    # Authorization
    "RBACManager",
    "Permission",
    "Role",
    "AccessRequest",
    "AccessResult",
    "ResourceType",
    "Action",

    # Validation
    "InputValidator",
    "DataSanitizer",
    "ValidationRule",
    "ValidationResult",
    "create_credit_score_validator",

    # Audit
    "SecurityAuditor",
    "AuditLog",
    "SecurityEvent",
    "EventType",
    "Severity"
]
