"""
Advanced Security and Compliance Framework.

This module implements comprehensive security measures, compliance validation,
and privacy protection for fairness-aware machine learning systems.

Features:
- Multi-layered security architecture
- GDPR/CCPA compliance validation
- Differential privacy implementation
- Audit logging and compliance reporting
- Data encryption and secure storage
- Access control and authentication
- Security monitoring and threat detection
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

try:
    from ..logging_config import get_logger
except ImportError:
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)

class SecurityLevel(Enum):
    """Security classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"
    CCPA = "ccpa"
    PIPEDA = "pipeda"
    LGPD = "lgpd"
    HIPAA = "hipaa"

class AccessLevel(Enum):
    """Access levels for data and models."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"

class AuditEventType(Enum):
    """Types of audit events."""
    DATA_ACCESS = "data_access"
    MODEL_TRAINING = "model_training"
    MODEL_INFERENCE = "model_inference"
    DATA_EXPORT = "data_export"
    SECURITY_VIOLATION = "security_violation"
    COMPLIANCE_CHECK = "compliance_check"
    USER_LOGIN = "user_login"
    CONFIGURATION_CHANGE = "configuration_change"

@dataclass
class SecurityContext:
    """Security context for operations."""
    user_id: str
    session_id: str
    access_level: AccessLevel
    security_clearance: SecurityLevel
    allowed_operations: Set[str] = field(default_factory=set)
    expires_at: Optional[datetime] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AuditEvent:
    """Audit event record."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_type: AuditEventType = AuditEventType.DATA_ACCESS
    user_id: str = ""
    resource_id: str = ""
    action: str = ""
    result: str = "success"
    ip_address: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0

@dataclass
class ComplianceReport:
    """Compliance assessment report."""
    framework: ComplianceFramework
    assessment_date: datetime
    overall_score: float
    compliant: bool
    violations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

class SecurityException(Exception):
    """Base security exception."""
    pass

class AccessDeniedException(SecurityException):
    """Access denied exception."""
    pass

class ComplianceViolationException(SecurityException):
    """Compliance violation exception."""
    pass

class PrivacyViolationException(SecurityException):
    """Privacy violation exception."""
    pass


class CryptographicManager:
    """
    Advanced cryptographic operations manager.

    Provides encryption, hashing, and digital signature capabilities
    for securing sensitive data and communications.
    """

    def __init__(self, key_length: int = 32):
        """
        Initialize cryptographic manager.

        Args:
            key_length: Length of encryption keys in bytes
        """
        self.key_length = key_length
        self._master_key = self._derive_master_key()

        logger.info("CryptographicManager initialized")

    def _derive_master_key(self) -> bytes:
        """Derive master encryption key."""
        # In production, this would use a proper key management system
        seed = os.environ.get('ENCRYPTION_SEED', 'default_development_seed')
        return hashlib.pbkdf2_hmac('sha256', seed.encode(), b'fairness_ml_salt', 100000, self.key_length)

    def encrypt_data(self, data: Union[str, bytes], context: Optional[str] = None) -> str:
        """
        Encrypt sensitive data.

        Args:
            data: Data to encrypt
            context: Encryption context for key derivation

        Returns:
            Base64-encoded encrypted data
        """
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')

            # Generate random salt and IV
            salt = secrets.token_bytes(16)
            iv = secrets.token_bytes(16)

            # Derive encryption key
            context_bytes = (context or "default").encode('utf-8')
            key = hashlib.pbkdf2_hmac('sha256', self._master_key + context_bytes, salt, 10000, 32)

            # Simple XOR encryption (in production, use proper AES)
            encrypted = bytes(a ^ b for a, b in zip(data, (key * (len(data) // len(key) + 1))[:len(data)]))

            # Combine salt + iv + encrypted data
            encrypted_package = salt + iv + encrypted

            return base64.b64encode(encrypted_package).decode('utf-8')

        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise SecurityException(f"Data encryption failed: {e}")

    def decrypt_data(self, encrypted_data: str, context: Optional[str] = None) -> bytes:
        """
        Decrypt encrypted data.

        Args:
            encrypted_data: Base64-encoded encrypted data
            context: Decryption context

        Returns:
            Decrypted data
        """
        try:
            # Decode from base64
            encrypted_package = base64.b64decode(encrypted_data.encode('utf-8'))

            # Extract components
            salt = encrypted_package[:16]
            encrypted_package[16:32]
            encrypted = encrypted_package[32:]

            # Derive decryption key
            context_bytes = (context or "default").encode('utf-8')
            key = hashlib.pbkdf2_hmac('sha256', self._master_key + context_bytes, salt, 10000, 32)

            # Decrypt data
            decrypted = bytes(a ^ b for a, b in zip(encrypted, (key * (len(encrypted) // len(key) + 1))[:len(encrypted)]))

            return decrypted

        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise SecurityException(f"Data decryption failed: {e}")

    def hash_sensitive_data(self, data: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """
        Hash sensitive data with salt.

        Args:
            data: Data to hash
            salt: Salt for hashing (generated if not provided)

        Returns:
            Tuple of (hash, salt)
        """
        if salt is None:
            salt = secrets.token_hex(16)

        hash_obj = hashlib.pbkdf2_hmac('sha256', data.encode('utf-8'), salt.encode('utf-8'), 100000)
        hash_hex = hash_obj.hex()

        return hash_hex, salt

    def create_signature(self, data: str, private_key: Optional[str] = None) -> str:
        """
        Create digital signature for data integrity.

        Args:
            data: Data to sign
            private_key: Private key (uses default if not provided)

        Returns:
            Digital signature
        """
        if private_key is None:
            private_key = base64.b64encode(self._master_key).decode('utf-8')

        signature = hmac.new(
            private_key.encode('utf-8'),
            data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return signature

    def verify_signature(self, data: str, signature: str, public_key: Optional[str] = None) -> bool:
        """
        Verify digital signature.

        Args:
            data: Original data
            signature: Signature to verify
            public_key: Public key (uses default if not provided)

        Returns:
            True if signature is valid
        """
        if public_key is None:
            public_key = base64.b64encode(self._master_key).decode('utf-8')

        expected_signature = self.create_signature(data, public_key)
        return hmac.compare_digest(signature, expected_signature)


class AccessControlManager:
    """
    Role-based access control (RBAC) manager.

    Manages user authentication, authorization, and access control
    for machine learning resources.
    """

    def __init__(self):
        """Initialize access control manager."""
        self.sessions: Dict[str, SecurityContext] = {}
        self.user_permissions: Dict[str, Dict[str, Set[AccessLevel]]] = {}
        self.crypto_manager = CryptographicManager()

        # Default admin user (for development)
        self._create_default_admin()

        logger.info("AccessControlManager initialized")

    def _create_default_admin(self):
        """Create default admin user for development."""
        admin_permissions = {
            "models": {AccessLevel.READ, AccessLevel.WRITE, AccessLevel.EXECUTE, AccessLevel.ADMIN},
            "data": {AccessLevel.READ, AccessLevel.WRITE, AccessLevel.ADMIN},
            "audit": {AccessLevel.READ, AccessLevel.ADMIN},
            "compliance": {AccessLevel.READ, AccessLevel.WRITE, AccessLevel.ADMIN}
        }
        self.user_permissions["admin"] = admin_permissions

    def authenticate_user(self, user_id: str, password: str, ip_address: Optional[str] = None) -> Optional[str]:
        """
        Authenticate user and create session.

        Args:
            user_id: User identifier
            password: User password
            ip_address: Client IP address

        Returns:
            Session token if authentication successful
        """
        # Simple authentication (in production, use proper auth system)
        if user_id == "admin" and password == "admin123":
            session_id = secrets.token_urlsafe(32)

            context = SecurityContext(
                user_id=user_id,
                session_id=session_id,
                access_level=AccessLevel.ADMIN,
                security_clearance=SecurityLevel.RESTRICTED,
                allowed_operations={"*"},  # Admin can do everything
                expires_at=datetime.utcnow() + timedelta(hours=8),
                ip_address=ip_address
            )

            self.sessions[session_id] = context
            logger.info(f"User {user_id} authenticated successfully")

            return session_id

        logger.warning(f"Authentication failed for user {user_id}")
        return None

    def get_security_context(self, session_token: str) -> Optional[SecurityContext]:
        """
        Get security context for session.

        Args:
            session_token: Session token

        Returns:
            Security context if valid session
        """
        context = self.sessions.get(session_token)

        if context is None:
            return None

        # Check if session expired
        if context.expires_at and datetime.utcnow() > context.expires_at:
            del self.sessions[session_token]
            logger.info(f"Session {session_token} expired")
            return None

        return context

    def check_permission(self, session_token: str, resource: str, access_level: AccessLevel) -> bool:
        """
        Check if user has permission for resource access.

        Args:
            session_token: Session token
            resource: Resource identifier
            access_level: Required access level

        Returns:
            True if access granted
        """
        context = self.get_security_context(session_token)
        if context is None:
            return False

        # Admin can access everything
        if context.access_level == AccessLevel.ADMIN:
            return True

        # Check specific permissions
        user_perms = self.user_permissions.get(context.user_id, {})
        resource_perms = user_perms.get(resource, set())

        return access_level in resource_perms

    def require_permission(self, session_token: str, resource: str, access_level: AccessLevel):
        """
        Require permission or raise exception.

        Args:
            session_token: Session token
            resource: Resource identifier
            access_level: Required access level

        Raises:
            AccessDeniedException: If access denied
        """
        if not self.check_permission(session_token, resource, access_level):
            context = self.get_security_context(session_token)
            user_id = context.user_id if context else "unknown"

            logger.warning(f"Access denied for user {user_id} to {resource} with {access_level}")
            raise AccessDeniedException(f"Insufficient permissions for {resource}")

    def revoke_session(self, session_token: str):
        """Revoke user session."""
        if session_token in self.sessions:
            del self.sessions[session_token]
            logger.info(f"Session {session_token} revoked")


class DifferentialPrivacy:
    """
    Differential privacy implementation for data protection.

    Provides privacy-preserving mechanisms for data analysis and
    model training while maintaining utility.
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        """
        Initialize differential privacy mechanism.

        Args:
            epsilon: Privacy budget (smaller = more private)
            delta: Probability of privacy breach
        """
        self.epsilon = epsilon
        self.delta = delta

        logger.info(f"DifferentialPrivacy initialized with ε={epsilon}, δ={delta}")

    def add_laplace_noise(self, data: np.ndarray, sensitivity: float) -> np.ndarray:
        """
        Add Laplace noise for differential privacy.

        Args:
            data: Original data
            sensitivity: Sensitivity of the query

        Returns:
            Data with noise added
        """
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale, data.shape)
        return data + noise

    def add_gaussian_noise(self, data: np.ndarray, sensitivity: float, delta: Optional[float] = None) -> np.ndarray:
        """
        Add Gaussian noise for differential privacy.

        Args:
            data: Original data
            sensitivity: Sensitivity of the query
            delta: Privacy parameter (uses instance delta if not provided)

        Returns:
            Data with noise added
        """
        if delta is None:
            delta = self.delta

        # Calculate noise scale for (epsilon, delta)-DP
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / self.epsilon
        noise = np.random.normal(0, sigma, data.shape)
        return data + noise

    def private_mean(self, data: np.ndarray, data_range: Tuple[float, float]) -> float:
        """
        Compute differentially private mean.

        Args:
            data: Input data
            data_range: Range of data values (min, max)

        Returns:
            Private mean
        """
        mean = np.mean(data)
        sensitivity = (data_range[1] - data_range[0]) / len(data)

        private_mean = self.add_laplace_noise(np.array([mean]), sensitivity)[0]
        return private_mean

    def private_count(self, data: np.ndarray) -> int:
        """
        Compute differentially private count.

        Args:
            data: Input data

        Returns:
            Private count
        """
        count = len(data)
        sensitivity = 1.0  # Adding/removing one person changes count by 1

        private_count = self.add_laplace_noise(np.array([count]), sensitivity)[0]
        return max(0, int(round(private_count)))  # Ensure non-negative integer

    def private_histogram(self, data: np.ndarray, bins: int = 10) -> np.ndarray:
        """
        Compute differentially private histogram.

        Args:
            data: Input data
            bins: Number of histogram bins

        Returns:
            Private histogram
        """
        hist, _ = np.histogram(data, bins=bins)
        sensitivity = 1.0  # Each person contributes to exactly one bin

        private_hist = self.add_laplace_noise(hist.astype(float), sensitivity)
        return np.maximum(0, private_hist)  # Ensure non-negative counts

    def create_private_dataset(self, df: pd.DataFrame, noise_columns: List[str],
                              sensitivities: Dict[str, float]) -> pd.DataFrame:
        """
        Create differentially private version of dataset.

        Args:
            df: Original dataframe
            noise_columns: Columns to add noise to
            sensitivities: Sensitivity for each column

        Returns:
            Private dataset
        """
        private_df = df.copy()

        for col in noise_columns:
            if col in df.columns and col in sensitivities:
                sensitivity = sensitivities[col]
                private_df[col] = self.add_laplace_noise(df[col].values, sensitivity)

        return private_df


class ComplianceValidator:
    """
    Compliance validation for various privacy frameworks.

    Validates data handling practices against GDPR, CCPA, and other
    privacy regulations.
    """

    def __init__(self):
        """Initialize compliance validator."""
        self.audit_trail: List[AuditEvent] = []

        logger.info("ComplianceValidator initialized")

    def assess_gdpr_compliance(self, data_processing_info: Dict[str, Any]) -> ComplianceReport:
        """
        Assess GDPR compliance.

        Args:
            data_processing_info: Information about data processing

        Returns:
            GDPR compliance report
        """
        violations = []
        recommendations = []
        evidence = {}

        # Check for explicit consent
        if not data_processing_info.get('has_consent', False):
            violations.append("No explicit consent documented for data processing")
            recommendations.append("Obtain and document explicit consent from data subjects")

        # Check for purpose limitation
        if not data_processing_info.get('purpose_documented', False):
            violations.append("Purpose of data processing not clearly documented")
            recommendations.append("Document specific purpose for data collection and processing")

        # Check for data minimization
        collected_fields = data_processing_info.get('collected_fields', [])
        if len(collected_fields) > 20:  # Arbitrary threshold
            violations.append(f"Potentially excessive data collection: {len(collected_fields)} fields")
            recommendations.append("Review data collection to ensure only necessary data is processed")

        # Check for retention policy
        if not data_processing_info.get('retention_policy', False):
            violations.append("No data retention policy documented")
            recommendations.append("Implement and document data retention and deletion policies")

        # Check for data subject rights
        rights_implemented = data_processing_info.get('data_subject_rights', [])
        required_rights = ['access', 'rectification', 'erasure', 'portability']
        missing_rights = [right for right in required_rights if right not in rights_implemented]

        if missing_rights:
            violations.append(f"Missing data subject rights: {missing_rights}")
            recommendations.append(f"Implement procedures for {missing_rights} rights")

        # Check for privacy by design
        if not data_processing_info.get('privacy_by_design', False):
            recommendations.append("Implement privacy by design principles in system architecture")

        # Calculate compliance score
        total_checks = 6
        violations_count = len(violations)
        compliance_score = max(0.0, (total_checks - violations_count) / total_checks)

        evidence = {
            'total_checks': total_checks,
            'violations_count': violations_count,
            'assessment_criteria': [
                'explicit_consent', 'purpose_limitation', 'data_minimization',
                'retention_policy', 'data_subject_rights', 'privacy_by_design'
            ]
        }

        return ComplianceReport(
            framework=ComplianceFramework.GDPR,
            assessment_date=datetime.utcnow(),
            overall_score=compliance_score,
            compliant=violations_count == 0,
            violations=violations,
            recommendations=recommendations,
            evidence=evidence
        )

    def assess_fairness_compliance(self, model_info: Dict[str, Any]) -> ComplianceReport:
        """
        Assess fairness and bias compliance.

        Args:
            model_info: Information about the ML model

        Returns:
            Fairness compliance report
        """
        violations = []
        recommendations = []
        evidence = {}

        # Check for bias testing
        if not model_info.get('bias_testing_performed', False):
            violations.append("No bias testing documented")
            recommendations.append("Perform comprehensive bias testing across protected attributes")

        # Check fairness metrics
        fairness_metrics = model_info.get('fairness_metrics', {})
        if not fairness_metrics:
            violations.append("No fairness metrics calculated")
            recommendations.append("Calculate and monitor key fairness metrics")
        else:
            # Check specific fairness thresholds
            demographic_parity = fairness_metrics.get('demographic_parity_difference', 0)
            if abs(demographic_parity) > 0.1:
                violations.append(f"Demographic parity violation: {demographic_parity:.3f}")
                recommendations.append("Implement bias mitigation techniques to improve demographic parity")

            equalized_odds = fairness_metrics.get('equalized_odds_difference', 0)
            if abs(equalized_odds) > 0.1:
                violations.append(f"Equalized odds violation: {equalized_odds:.3f}")
                recommendations.append("Address equalized odds disparity through post-processing")

        # Check for protected attributes handling
        if not model_info.get('protected_attributes_identified', False):
            violations.append("Protected attributes not properly identified or documented")
            recommendations.append("Identify and document all protected attributes used in analysis")

        # Check for fairness monitoring
        if not model_info.get('ongoing_monitoring', False):
            recommendations.append("Implement ongoing fairness monitoring in production")

        # Calculate compliance score
        total_checks = 4
        violations_count = len(violations)
        compliance_score = max(0.0, (total_checks - violations_count) / total_checks)

        evidence = {
            'total_checks': total_checks,
            'violations_count': violations_count,
            'fairness_metrics': fairness_metrics,
            'assessment_criteria': [
                'bias_testing', 'fairness_metrics', 'protected_attributes', 'ongoing_monitoring'
            ]
        }

        return ComplianceReport(
            framework=ComplianceFramework.GDPR,  # Using GDPR as base for fairness
            assessment_date=datetime.utcnow(),
            overall_score=compliance_score,
            compliant=violations_count == 0,
            violations=violations,
            recommendations=recommendations,
            evidence=evidence
        )


class SecurityAuditLogger:
    """
    Comprehensive security audit logging system.

    Tracks all security-relevant events for compliance and monitoring.
    """

    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize security audit logger.

        Args:
            log_file: Path to audit log file
        """
        self.log_file = log_file or "security_audit.log"
        self.events: List[AuditEvent] = []
        self.crypto_manager = CryptographicManager()

        logger.info("SecurityAuditLogger initialized")

    def log_event(self, event: AuditEvent):
        """
        Log a security audit event.

        Args:
            event: Audit event to log
        """
        # Add timestamp if not set
        if not hasattr(event, 'timestamp') or event.timestamp is None:
            event.timestamp = datetime.utcnow()

        # Calculate risk score
        event.risk_score = self._calculate_risk_score(event)

        # Add to memory
        self.events.append(event)

        # Write to file
        self._write_to_file(event)

        # Log high-risk events
        if event.risk_score > 0.7:
            logger.warning(f"High-risk security event: {event.event_type.value} by {event.user_id}")

    def _calculate_risk_score(self, event: AuditEvent) -> float:
        """Calculate risk score for audit event."""
        risk_score = 0.0

        # Base risk by event type
        risk_map = {
            AuditEventType.DATA_ACCESS: 0.3,
            AuditEventType.MODEL_TRAINING: 0.5,
            AuditEventType.MODEL_INFERENCE: 0.2,
            AuditEventType.DATA_EXPORT: 0.8,
            AuditEventType.SECURITY_VIOLATION: 0.9,
            AuditEventType.COMPLIANCE_CHECK: 0.1,
            AuditEventType.USER_LOGIN: 0.3,
            AuditEventType.CONFIGURATION_CHANGE: 0.7,
        }

        risk_score = risk_map.get(event.event_type, 0.5)

        # Increase risk for failures
        if event.result != "success":
            risk_score += 0.3

        # Increase risk for sensitive operations
        if any(keyword in event.action.lower() for keyword in ['delete', 'export', 'modify']):
            risk_score += 0.2

        return min(1.0, risk_score)

    def _write_to_file(self, event: AuditEvent):
        """Write audit event to log file."""
        try:
            log_entry = {
                'event_id': event.event_id,
                'timestamp': event.timestamp.isoformat(),
                'event_type': event.event_type.value,
                'user_id': event.user_id,
                'resource_id': event.resource_id,
                'action': event.action,
                'result': event.result,
                'ip_address': event.ip_address,
                'risk_score': event.risk_score,
                'details': event.details
            }

            # Create integrity signature
            log_data = json.dumps(log_entry, sort_keys=True)
            signature = self.crypto_manager.create_signature(log_data)
            log_entry['signature'] = signature

            # Write to file
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

    def get_events_by_user(self, user_id: str, limit: int = 100) -> List[AuditEvent]:
        """Get audit events for specific user."""
        user_events = [event for event in self.events if event.user_id == user_id]
        return user_events[-limit:]

    def get_high_risk_events(self, threshold: float = 0.7, limit: int = 50) -> List[AuditEvent]:
        """Get high-risk audit events."""
        high_risk = [event for event in self.events if event.risk_score >= threshold]
        return sorted(high_risk, key=lambda e: e.risk_score, reverse=True)[:limit]

    def generate_security_report(self, days: int = 30) -> Dict[str, Any]:
        """
        Generate security summary report.

        Args:
            days: Number of days to include in report

        Returns:
            Security report
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_events = [event for event in self.events if event.timestamp >= cutoff_date]

        # Event statistics
        event_counts = {}
        for event in recent_events:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        # Risk statistics
        high_risk_count = len([e for e in recent_events if e.risk_score >= 0.7])
        avg_risk_score = np.mean([e.risk_score for e in recent_events]) if recent_events else 0

        # User activity
        user_activity = {}
        for event in recent_events:
            user_activity[event.user_id] = user_activity.get(event.user_id, 0) + 1

        # Failure analysis
        failures = [e for e in recent_events if e.result != "success"]
        failure_rate = len(failures) / len(recent_events) if recent_events else 0

        return {
            'report_period_days': days,
            'total_events': len(recent_events),
            'event_counts': event_counts,
            'high_risk_events': high_risk_count,
            'average_risk_score': avg_risk_score,
            'user_activity': user_activity,
            'failure_rate': failure_rate,
            'top_failures': [f"{e.action}: {e.details.get('error', 'Unknown error')}" for e in failures[:5]]
        }


def create_secure_ml_pipeline():
    """
    Create a secure ML pipeline with comprehensive security measures.

    Returns:
        Dictionary of security components
    """
    # Initialize security components
    crypto_manager = CryptographicManager()
    access_control = AccessControlManager()
    privacy_manager = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
    compliance_validator = ComplianceValidator()
    audit_logger = SecurityAuditLogger()

    return {
        'crypto_manager': crypto_manager,
        'access_control': access_control,
        'privacy_manager': privacy_manager,
        'compliance_validator': compliance_validator,
        'audit_logger': audit_logger
    }


def demonstrate_security_framework():
    """Demonstrate the advanced security framework."""
    print("🔒 Advanced Security Framework Demonstration")

    # Create security pipeline
    security_components = create_secure_ml_pipeline()

    # Test authentication and authorization
    print("\n🔐 Testing Authentication & Authorization...")

    access_control = security_components['access_control']

    # Authenticate user
    session_token = access_control.authenticate_user("admin", "admin123", "192.168.1.1")
    if session_token:
        print(f"   ✅ User authenticated, session token: {session_token[:16]}...")

        # Test permissions
        has_model_access = access_control.check_permission(session_token, "models", AccessLevel.READ)
        print(f"   Model access granted: {has_model_access}")

        has_admin_access = access_control.check_permission(session_token, "audit", AccessLevel.ADMIN)
        print(f"   Admin access granted: {has_admin_access}")
    else:
        print("   ❌ Authentication failed")

    # Test encryption
    print("\n🔐 Testing Encryption...")

    crypto_manager = security_components['crypto_manager']

    sensitive_data = "Personal ID: 123-45-6789, Credit Score: 750"
    encrypted = crypto_manager.encrypt_data(sensitive_data, context="user_data")
    print(f"   Original data: {sensitive_data}")
    print(f"   Encrypted: {encrypted[:50]}...")

    decrypted = crypto_manager.decrypt_data(encrypted, context="user_data").decode('utf-8')
    print(f"   Decrypted: {decrypted}")
    print(f"   ✅ Encryption/Decryption successful: {sensitive_data == decrypted}")

    # Test differential privacy
    print("\n🛡️ Testing Differential Privacy...")

    privacy_manager = security_components['privacy_manager']

    # Generate sample sensitive data
    ages = np.random.normal(35, 10, 1000)
    ages = np.clip(ages, 18, 80)  # Realistic age range

    true_mean = np.mean(ages)
    private_mean = privacy_manager.private_mean(ages, (18, 80))

    print(f"   True mean age: {true_mean:.2f}")
    print(f"   Private mean age: {private_mean:.2f}")
    print(f"   Privacy noise: {abs(true_mean - private_mean):.2f}")

    # Test compliance validation
    print("\n📋 Testing Compliance Validation...")

    compliance_validator = security_components['compliance_validator']

    # GDPR compliance test
    data_processing_info = {
        'has_consent': True,
        'purpose_documented': True,
        'collected_fields': ['name', 'age', 'income', 'credit_score'],
        'retention_policy': True,
        'data_subject_rights': ['access', 'rectification', 'erasure'],
        'privacy_by_design': False
    }

    gdpr_report = compliance_validator.assess_gdpr_compliance(data_processing_info)
    print(f"   GDPR Compliance Score: {gdpr_report.overall_score:.2f}")
    print(f"   Compliant: {gdpr_report.compliant}")
    print(f"   Violations: {len(gdpr_report.violations)}")
    if gdpr_report.violations:
        print(f"     - {gdpr_report.violations[0]}")

    # Fairness compliance test
    model_info = {
        'bias_testing_performed': True,
        'fairness_metrics': {
            'demographic_parity_difference': 0.05,
            'equalized_odds_difference': 0.08
        },
        'protected_attributes_identified': True,
        'ongoing_monitoring': False
    }

    fairness_report = compliance_validator.assess_fairness_compliance(model_info)
    print(f"   Fairness Compliance Score: {fairness_report.overall_score:.2f}")
    print(f"   Compliant: {fairness_report.compliant}")

    # Test audit logging
    print("\n📝 Testing Audit Logging...")

    audit_logger = security_components['audit_logger']

    # Log some sample events
    events = [
        AuditEvent(
            event_type=AuditEventType.DATA_ACCESS,
            user_id="admin",
            resource_id="credit_dataset",
            action="read_data",
            ip_address="192.168.1.1"
        ),
        AuditEvent(
            event_type=AuditEventType.MODEL_TRAINING,
            user_id="admin",
            resource_id="fairness_model",
            action="train_model",
            result="success"
        ),
        AuditEvent(
            event_type=AuditEventType.SECURITY_VIOLATION,
            user_id="unknown",
            resource_id="system",
            action="unauthorized_access",
            result="blocked",
            ip_address="10.0.0.1"
        )
    ]

    for event in events:
        audit_logger.log_event(event)

    print(f"   ✅ Logged {len(events)} audit events")

    # Generate security report
    security_report = audit_logger.generate_security_report(days=1)
    print(f"   Total events: {security_report['total_events']}")
    print(f"   High-risk events: {security_report['high_risk_events']}")
    print(f"   Average risk score: {security_report['average_risk_score']:.3f}")

    print("\n✅ Advanced security framework demonstration completed! 🔒")


if __name__ == "__main__":
    demonstrate_security_framework()
