"""
Security audit logging and monitoring system.

Comprehensive audit logging for security events, compliance monitoring,
and forensic analysis in the fair credit scoring system.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from ..logging_config import get_logger

logger = get_logger(__name__)


class EventType(Enum):
    """Types of security events."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    MODEL_OPERATION = "model_operation"
    SYSTEM_CHANGE = "system_change"
    COMPLIANCE = "compliance"
    SECURITY_VIOLATION = "security_violation"
    ADMIN_ACTION = "admin_action"


class Severity(Enum):
    """Event severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event record."""
    event_id: str
    event_type: EventType
    severity: Severity
    user_id: Optional[str]
    action: str
    resource: str
    outcome: str  # success, failure, warning
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Generate event ID if not provided."""
        if not self.event_id:
            self.event_id = self._generate_event_id()

    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        content = f"{self.timestamp.isoformat()}{self.user_id}{self.action}{self.resource}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'severity': self.severity.value,
            'user_id': self.user_id,
            'action': self.action,
            'resource': self.resource,
            'outcome': self.outcome,
            'details': self.details,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'timestamp': self.timestamp.isoformat()
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class AuditLog:
    """Audit log entry with integrity verification."""
    log_id: str
    events: List[SecurityEvent]
    checksum: str
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Calculate checksum after initialization."""
        if not self.checksum:
            self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate integrity checksum."""
        content = ""
        for event in sorted(self.events, key=lambda x: x.timestamp):
            content += event.event_id + event.timestamp.isoformat()

        return hashlib.sha256(content.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify log integrity."""
        return self.checksum == self._calculate_checksum()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'log_id': self.log_id,
            'events': [event.to_dict() for event in self.events],
            'checksum': self.checksum,
            'created_at': self.created_at.isoformat(),
            'integrity_verified': self.verify_integrity()
        }


class SecurityAuditor:
    """
    Security audit logging and monitoring system.
    
    Provides comprehensive audit logging, compliance monitoring,
    and security event analysis capabilities.
    """

    def __init__(
        self,
        retention_days: int = 365,
        max_events_per_log: int = 1000,
        enable_real_time_alerts: bool = True
    ):
        """
        Initialize security auditor.
        
        Args:
            retention_days: How long to retain audit logs
            max_events_per_log: Maximum events per audit log
            enable_real_time_alerts: Enable real-time security alerts
        """
        self.retention_days = retention_days
        self.max_events_per_log = max_events_per_log
        self.enable_real_time_alerts = enable_real_time_alerts

        # Event storage
        self.events: List[SecurityEvent] = []
        self.audit_logs: List[AuditLog] = []

        # Alert thresholds
        self.alert_thresholds = {
            'failed_logins_per_hour': 10,
            'unauthorized_access_per_hour': 5,
            'model_operations_per_minute': 100,
            'data_access_volume_mb_per_hour': 1000
        }

        # Compliance tracking
        self.compliance_violations: List[Dict[str, Any]] = []

        logger.info("SecurityAuditor initialized")

    def log_event(
        self,
        event_type: EventType,
        severity: Severity,
        action: str,
        resource: str,
        outcome: str,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> SecurityEvent:
        """
        Log a security event.
        
        Args:
            event_type: Type of security event
            severity: Event severity
            action: Action performed
            resource: Resource affected
            outcome: Operation outcome
            user_id: User ID (if applicable)
            details: Additional event details
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Created SecurityEvent
        """
        event = SecurityEvent(
            event_id="",  # Will be generated
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            action=action,
            resource=resource,
            outcome=outcome,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent
        )

        self.events.append(event)

        # Check for real-time alerts
        if self.enable_real_time_alerts:
            self._check_real_time_alerts(event)

        # Create audit log if needed
        if len(self.events) >= self.max_events_per_log:
            self._create_audit_log()

        logger.info(f"Security event logged: {event.action} on {event.resource}")
        return event

    def log_authentication_event(
        self,
        action: str,
        outcome: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> SecurityEvent:
        """Log authentication-related event."""
        severity = Severity.WARNING if outcome == "failure" else Severity.INFO

        return self.log_event(
            event_type=EventType.AUTHENTICATION,
            severity=severity,
            action=action,
            resource="authentication_system",
            outcome=outcome,
            user_id=user_id,
            details=details,
            ip_address=ip_address
        )

    def log_authorization_event(
        self,
        action: str,
        resource: str,
        outcome: str,
        user_id: str,
        details: Optional[Dict[str, Any]] = None
    ) -> SecurityEvent:
        """Log authorization-related event."""
        severity = Severity.WARNING if outcome == "denied" else Severity.INFO

        return self.log_event(
            event_type=EventType.AUTHORIZATION,
            severity=severity,
            action=action,
            resource=resource,
            outcome=outcome,
            user_id=user_id,
            details=details
        )

    def log_data_access_event(
        self,
        action: str,
        resource: str,
        outcome: str,
        user_id: str,
        data_size_mb: Optional[float] = None,
        sensitive_data: bool = False
    ) -> SecurityEvent:
        """Log data access event."""
        severity = Severity.WARNING if sensitive_data else Severity.INFO

        details = {}
        if data_size_mb is not None:
            details['data_size_mb'] = data_size_mb
        if sensitive_data:
            details['sensitive_data'] = True

        return self.log_event(
            event_type=EventType.DATA_ACCESS,
            severity=severity,
            action=action,
            resource=resource,
            outcome=outcome,
            user_id=user_id,
            details=details
        )

    def log_model_operation_event(
        self,
        action: str,
        model_id: str,
        outcome: str,
        user_id: str,
        details: Optional[Dict[str, Any]] = None
    ) -> SecurityEvent:
        """Log model operation event."""
        return self.log_event(
            event_type=EventType.MODEL_OPERATION,
            severity=Severity.INFO,
            action=action,
            resource=f"model:{model_id}",
            outcome=outcome,
            user_id=user_id,
            details=details
        )

    def log_compliance_violation(
        self,
        violation_type: str,
        description: str,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        severity: Severity = Severity.ERROR
    ) -> SecurityEvent:
        """Log compliance violation."""
        details = {
            'violation_type': violation_type,
            'description': description
        }

        event = self.log_event(
            event_type=EventType.COMPLIANCE,
            severity=severity,
            action="compliance_violation",
            resource=resource or "system",
            outcome="violation",
            user_id=user_id,
            details=details
        )

        # Track compliance violations separately
        violation_record = {
            'event_id': event.event_id,
            'violation_type': violation_type,
            'description': description,
            'user_id': user_id,
            'resource': resource,
            'severity': severity.value,
            'timestamp': datetime.utcnow().isoformat()
        }

        self.compliance_violations.append(violation_record)

        return event

    def search_events(
        self,
        event_type: Optional[EventType] = None,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource: Optional[str] = None,
        outcome: Optional[str] = None,
        severity: Optional[Severity] = None,
        hours: int = 24
    ) -> List[SecurityEvent]:
        """
        Search security events.
        
        Args:
            event_type: Filter by event type
            user_id: Filter by user ID
            action: Filter by action
            resource: Filter by resource
            outcome: Filter by outcome
            severity: Filter by severity
            hours: Hours of history to search
            
        Returns:
            List of matching events
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        filtered_events = []
        for event in self.events:
            # Time filter
            if event.timestamp < cutoff_time:
                continue

            # Apply filters
            if event_type and event.event_type != event_type:
                continue
            if user_id and event.user_id != user_id:
                continue
            if action and action.lower() not in event.action.lower():
                continue
            if resource and resource.lower() not in event.resource.lower():
                continue
            if outcome and event.outcome != outcome:
                continue
            if severity and event.severity != severity:
                continue

            filtered_events.append(event)

        return filtered_events

    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get security summary for the specified time period.
        
        Args:
            hours: Hours of history to analyze
            
        Returns:
            Security summary statistics
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_events = [e for e in self.events if e.timestamp >= cutoff_time]

        # Count by type
        event_type_counts = {}
        for event_type in EventType:
            count = len([e for e in recent_events if e.event_type == event_type])
            event_type_counts[event_type.value] = count

        # Count by severity
        severity_counts = {}
        for severity in Severity:
            count = len([e for e in recent_events if e.severity == severity])
            severity_counts[severity.value] = count

        # Count by outcome
        outcome_counts = {}
        for event in recent_events:
            outcome_counts[event.outcome] = outcome_counts.get(event.outcome, 0) + 1

        # Failed authentication attempts
        failed_auth = len([
            e for e in recent_events
            if e.event_type == EventType.AUTHENTICATION and e.outcome == "failure"
        ])

        # Unauthorized access attempts
        unauthorized_access = len([
            e for e in recent_events
            if e.event_type == EventType.AUTHORIZATION and e.outcome == "denied"
        ])

        # Recent compliance violations
        recent_violations = [
            v for v in self.compliance_violations
            if datetime.fromisoformat(v['timestamp']) >= cutoff_time
        ]

        summary = {
            'time_period_hours': hours,
            'total_events': len(recent_events),
            'events_by_type': event_type_counts,
            'events_by_severity': severity_counts,
            'events_by_outcome': outcome_counts,
            'failed_authentications': failed_auth,
            'unauthorized_access_attempts': unauthorized_access,
            'compliance_violations': len(recent_violations),
            'unique_users': len(set(e.user_id for e in recent_events if e.user_id)),
            'unique_resources': len(set(e.resource for e in recent_events))
        }

        return summary

    def get_compliance_report(self, days: int = 30) -> Dict[str, Any]:
        """
        Generate compliance report.
        
        Args:
            days: Days of history to include
            
        Returns:
            Compliance report
        """
        cutoff_time = datetime.utcnow() - timedelta(days=days)

        # Recent violations
        recent_violations = [
            v for v in self.compliance_violations
            if datetime.fromisoformat(v['timestamp']) >= cutoff_time
        ]

        # Group by violation type
        violations_by_type = {}
        for violation in recent_violations:
            vtype = violation['violation_type']
            violations_by_type[vtype] = violations_by_type.get(vtype, 0) + 1

        # Compliance events
        compliance_events = self.search_events(
            event_type=EventType.COMPLIANCE,
            hours=days * 24
        )

        report = {
            'report_period_days': days,
            'total_violations': len(recent_violations),
            'violations_by_type': violations_by_type,
            'compliance_events': len(compliance_events),
            'violation_trend': self._calculate_violation_trend(days),
            'critical_violations': len([
                v for v in recent_violations
                if v['severity'] == 'critical'
            ]),
            'recommendations': self._generate_compliance_recommendations(recent_violations)
        }

        return report

    def export_audit_trail(self, format: str = "json") -> str:
        """
        Export audit trail for external analysis.
        
        Args:
            format: Export format (json, csv)
            
        Returns:
            Exported audit data
        """
        if format == "json":
            audit_data = {
                'export_timestamp': datetime.utcnow().isoformat(),
                'events_count': len(self.events),
                'audit_logs_count': len(self.audit_logs),
                'events': [event.to_dict() for event in self.events[-1000:]],  # Last 1000 events
                'compliance_violations': self.compliance_violations,
                'summary': self.get_security_summary()
            }
            return json.dumps(audit_data, indent=2)

        elif format == "csv":
            import csv
            import io

            output = io.StringIO()
            writer = csv.writer(output)

            # Write header
            writer.writerow([
                'event_id', 'timestamp', 'event_type', 'severity', 'user_id',
                'action', 'resource', 'outcome', 'ip_address'
            ])

            # Write events
            for event in self.events[-1000:]:  # Last 1000 events
                writer.writerow([
                    event.event_id,
                    event.timestamp.isoformat(),
                    event.event_type.value,
                    event.severity.value,
                    event.user_id,
                    event.action,
                    event.resource,
                    event.outcome,
                    event.ip_address
                ])

            return output.getvalue()

        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _create_audit_log(self):
        """Create audit log from current events."""
        if not self.events:
            return

        log_id = f"audit_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        audit_log = AuditLog(
            log_id=log_id,
            events=self.events.copy(),
            checksum=""  # Will be calculated
        )

        self.audit_logs.append(audit_log)
        self.events.clear()

        logger.info(f"Audit log created: {log_id}")

    def _check_real_time_alerts(self, event: SecurityEvent):
        """Check for real-time security alerts."""
        current_time = datetime.utcnow()

        # Failed login threshold
        if (event.event_type == EventType.AUTHENTICATION and
            event.outcome == "failure"):

            recent_failures = len([
                e for e in self.events[-100:]  # Check last 100 events
                if (e.event_type == EventType.AUTHENTICATION and
                    e.outcome == "failure" and
                    e.user_id == event.user_id and
                    (current_time - e.timestamp).total_seconds() < 3600)  # Last hour
            ])

            if recent_failures >= self.alert_thresholds['failed_logins_per_hour']:
                self._trigger_security_alert(
                    "excessive_failed_logins",
                    f"User {event.user_id} has {recent_failures} failed login attempts in the last hour",
                    event
                )

        # Unauthorized access threshold
        if (event.event_type == EventType.AUTHORIZATION and
            event.outcome == "denied"):

            recent_denials = len([
                e for e in self.events[-100:]
                if (e.event_type == EventType.AUTHORIZATION and
                    e.outcome == "denied" and
                    e.user_id == event.user_id and
                    (current_time - e.timestamp).total_seconds() < 3600)
            ])

            if recent_denials >= self.alert_thresholds['unauthorized_access_per_hour']:
                self._trigger_security_alert(
                    "excessive_unauthorized_access",
                    f"User {event.user_id} has {recent_denials} unauthorized access attempts in the last hour",
                    event
                )

    def _trigger_security_alert(self, alert_type: str, message: str, trigger_event: SecurityEvent):
        """Trigger security alert."""
        alert_event = self.log_event(
            event_type=EventType.SECURITY_VIOLATION,
            severity=Severity.CRITICAL,
            action=f"security_alert:{alert_type}",
            resource="security_system",
            outcome="alert_triggered",
            user_id=trigger_event.user_id,
            details={
                'alert_type': alert_type,
                'message': message,
                'trigger_event_id': trigger_event.event_id
            }
        )

        logger.critical(f"Security alert triggered: {alert_type} - {message}")

    def _calculate_violation_trend(self, days: int) -> str:
        """Calculate compliance violation trend."""
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        recent_violations = [
            v for v in self.compliance_violations
            if datetime.fromisoformat(v['timestamp']) >= cutoff_time
        ]

        if len(recent_violations) < 2:
            return "insufficient_data"

        # Simple trend calculation
        mid_point = days // 2
        first_half_cutoff = datetime.utcnow() - timedelta(days=mid_point)

        first_half = [
            v for v in recent_violations
            if datetime.fromisoformat(v['timestamp']) < first_half_cutoff
        ]
        second_half = [
            v for v in recent_violations
            if datetime.fromisoformat(v['timestamp']) >= first_half_cutoff
        ]

        if len(second_half) > len(first_half) * 1.2:
            return "increasing"
        elif len(second_half) < len(first_half) * 0.8:
            return "decreasing"
        else:
            return "stable"

    def _generate_compliance_recommendations(self, violations: List[Dict[str, Any]]) -> List[str]:
        """Generate compliance recommendations based on violations."""
        recommendations = []

        if not violations:
            recommendations.append("No compliance violations detected - maintain current practices")
            return recommendations

        # Count violation types
        violation_types = {}
        for violation in violations:
            vtype = violation['violation_type']
            violation_types[vtype] = violation_types.get(vtype, 0) + 1

        # Generate specific recommendations
        if violation_types.get('data_access', 0) > 0:
            recommendations.append("Review data access controls and implement stricter access policies")

        if violation_types.get('authentication', 0) > 0:
            recommendations.append("Strengthen authentication requirements and monitor login patterns")

        if violation_types.get('authorization', 0) > 0:
            recommendations.append("Review user roles and permissions for principle of least privilege")

        if len(violations) > 10:
            recommendations.append("Consider implementing automated compliance monitoring")

        recommendations.append("Conduct regular security training for users")
        recommendations.append("Review and update security policies based on violation patterns")

        return recommendations


# CLI interface
def main():
    """CLI interface for audit testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Security Audit CLI")
    parser.add_argument("command", choices=["log", "search", "summary", "export"])
    parser.add_argument("--event-type", help="Event type")
    parser.add_argument("--user-id", help="User ID")
    parser.add_argument("--action", help="Action")
    parser.add_argument("--resource", help="Resource")
    parser.add_argument("--outcome", help="Outcome")
    parser.add_argument("--hours", type=int, default=24, help="Hours of history")
    parser.add_argument("--format", choices=["json", "csv"], default="json", help="Export format")
    parser.add_argument("--output", help="Output file path")

    args = parser.parse_args()

    # Initialize auditor
    auditor = SecurityAuditor()

    if args.command == "log":
        # Log demo events
        auditor.log_authentication_event("login", "success", "user123", "192.168.1.100")
        auditor.log_authentication_event("login", "failure", "user456", "192.168.1.101")
        auditor.log_data_access_event("read", "customer_data", "success", "user123", 10.5, True)
        auditor.log_compliance_violation("unauthorized_access", "User accessed restricted data")

        print("Demo security events logged")

    elif args.command == "search":
        event_type = EventType(args.event_type) if args.event_type else None

        events = auditor.search_events(
            event_type=event_type,
            user_id=args.user_id,
            action=args.action,
            resource=args.resource,
            outcome=args.outcome,
            hours=args.hours
        )

        print(f"Found {len(events)} matching events:")
        for event in events[-10:]:  # Show last 10
            print(f"  {event.timestamp}: {event.action} on {event.resource} - {event.outcome}")

    elif args.command == "summary":
        summary = auditor.get_security_summary(args.hours)

        print(f"Security Summary ({args.hours} hours):")
        print(f"  Total events: {summary['total_events']}")
        print(f"  Failed authentications: {summary['failed_authentications']}")
        print(f"  Unauthorized access: {summary['unauthorized_access_attempts']}")
        print(f"  Compliance violations: {summary['compliance_violations']}")
        print(f"  Events by severity: {summary['events_by_severity']}")

    elif args.command == "export":
        exported_data = auditor.export_audit_trail(args.format)

        if args.output:
            with open(args.output, 'w') as f:
                f.write(exported_data)
            print(f"Audit trail exported to {args.output}")
        else:
            print(exported_data[:500] + "..." if len(exported_data) > 500 else exported_data)


if __name__ == "__main__":
    main()
