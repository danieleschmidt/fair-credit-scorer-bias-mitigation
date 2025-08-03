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

from .authentication import AuthenticationManager, SessionManager, Token, User, Session
from .authorization import RBACManager, Permission, Role, AccessRequest, AccessResult, ResourceType, Action
from .validation import InputValidator, DataSanitizer, ValidationRule, ValidationResult, create_credit_score_validator
from .audit import SecurityAuditor, AuditLog, SecurityEvent, EventType, Severity

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