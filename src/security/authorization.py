"""
Role-Based Access Control (RBAC) and authorization system.

Comprehensive authorization framework for controlling access to
resources and operations in the fair credit scoring system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from ..logging_config import get_logger

logger = get_logger(__name__)


class ResourceType(Enum):
    """Types of resources that can be protected."""
    MODEL = "model"
    DATA = "data"
    REPORT = "report"
    AUDIT = "audit"
    SYSTEM = "system"
    USER = "user"
    API = "api"


class Action(Enum):
    """Actions that can be performed on resources."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    APPROVE = "approve"
    AUDIT = "audit"
    ADMIN = "admin"


@dataclass
class Permission:
    """Permission definition for specific resource and action."""
    resource_type: ResourceType
    resource_id: Optional[str]  # None for all resources of type
    action: Action
    conditions: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """String representation."""
        resource = f"{self.resource_type.value}"
        if self.resource_id:
            resource += f":{self.resource_id}"
        return f"{resource}:{self.action.value}"

    def matches(self, resource_type: ResourceType, resource_id: str, action: Action) -> bool:
        """Check if this permission matches the given resource and action."""
        # Check resource type
        if self.resource_type != resource_type:
            return False

        # Check action
        if self.action != action:
            return False

        # Check resource ID (None means all resources of this type)
        if self.resource_id is not None and self.resource_id != resource_id:
            return False

        return True


@dataclass
class Role:
    """Role definition with associated permissions."""
    name: str
    description: str
    permissions: List[Permission] = field(default_factory=list)
    inherits_from: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True

    def __str__(self) -> str:
        """String representation."""
        return f"Role({self.name})"

    def has_permission(self, resource_type: ResourceType, resource_id: str, action: Action) -> bool:
        """Check if role has specific permission."""
        for permission in self.permissions:
            if permission.matches(resource_type, resource_id, action):
                return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'permissions': [str(p) for p in self.permissions],
            'inherits_from': self.inherits_from,
            'created_at': self.created_at.isoformat(),
            'is_active': self.is_active
        }


@dataclass
class AccessRequest:
    """Access request for authorization check."""
    user_id: str
    resource_type: ResourceType
    resource_id: str
    action: Action
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AccessResult:
    """Result of authorization check."""
    granted: bool
    reason: str
    matched_permissions: List[str] = field(default_factory=list)
    required_permissions: List[str] = field(default_factory=list)
    user_roles: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'granted': self.granted,
            'reason': self.reason,
            'matched_permissions': self.matched_permissions,
            'required_permissions': self.required_permissions,
            'user_roles': self.user_roles,
            'timestamp': self.timestamp.isoformat()
        }


class RBACManager:
    """
    Role-Based Access Control Manager.
    
    Manages roles, permissions, and access control decisions
    for the fair credit scoring system.
    """

    def __init__(self):
        """Initialize RBAC manager."""
        self.roles: Dict[str, Role] = {}
        self.user_roles: Dict[str, List[str]] = {}
        self.access_logs: List[Dict[str, Any]] = []

        # Initialize default roles
        self._setup_default_roles()

        logger.info("RBACManager initialized")

    def _setup_default_roles(self):
        """Setup default system roles."""
        # Admin role - full access
        admin_permissions = [
            Permission(ResourceType.MODEL, None, Action.ADMIN),
            Permission(ResourceType.DATA, None, Action.ADMIN),
            Permission(ResourceType.REPORT, None, Action.ADMIN),
            Permission(ResourceType.AUDIT, None, Action.ADMIN),
            Permission(ResourceType.SYSTEM, None, Action.ADMIN),
            Permission(ResourceType.USER, None, Action.ADMIN),
            Permission(ResourceType.API, None, Action.ADMIN),
        ]

        admin_role = Role(
            name="admin",
            description="System administrator with full access",
            permissions=admin_permissions
        )

        # Data Scientist role - model and data access
        data_scientist_permissions = [
            Permission(ResourceType.MODEL, None, Action.CREATE),
            Permission(ResourceType.MODEL, None, Action.READ),
            Permission(ResourceType.MODEL, None, Action.UPDATE),
            Permission(ResourceType.MODEL, None, Action.EXECUTE),
            Permission(ResourceType.DATA, None, Action.READ),
            Permission(ResourceType.DATA, None, Action.CREATE),
            Permission(ResourceType.REPORT, None, Action.CREATE),
            Permission(ResourceType.REPORT, None, Action.READ),
        ]

        data_scientist_role = Role(
            name="data_scientist",
            description="Data scientist with model development access",
            permissions=data_scientist_permissions
        )

        # Analyst role - read access to models and reports
        analyst_permissions = [
            Permission(ResourceType.MODEL, None, Action.READ),
            Permission(ResourceType.MODEL, None, Action.EXECUTE),
            Permission(ResourceType.REPORT, None, Action.READ),
            Permission(ResourceType.REPORT, None, Action.CREATE),
            Permission(ResourceType.DATA, None, Action.READ),
        ]

        analyst_role = Role(
            name="analyst",
            description="Business analyst with read access",
            permissions=analyst_permissions
        )

        # Auditor role - audit access
        auditor_permissions = [
            Permission(ResourceType.AUDIT, None, Action.READ),
            Permission(ResourceType.AUDIT, None, Action.CREATE),
            Permission(ResourceType.MODEL, None, Action.READ),
            Permission(ResourceType.REPORT, None, Action.READ),
            Permission(ResourceType.SYSTEM, None, Action.READ),
        ]

        auditor_role = Role(
            name="auditor",
            description="Auditor with compliance monitoring access",
            permissions=auditor_permissions
        )

        # User role - basic API access
        user_permissions = [
            Permission(ResourceType.API, None, Action.READ),
            Permission(ResourceType.MODEL, None, Action.EXECUTE),
        ]

        user_role = Role(
            name="user",
            description="Basic user with API access",
            permissions=user_permissions
        )

        # Add roles to manager
        for role in [admin_role, data_scientist_role, analyst_role, auditor_role, user_role]:
            self.roles[role.name] = role

        logger.info("Default roles created")

    def create_role(self, role: Role) -> bool:
        """
        Create a new role.
        
        Args:
            role: Role to create
            
        Returns:
            True if successful
        """
        if role.name in self.roles:
            logger.error(f"Role already exists: {role.name}")
            return False

        # Validate inherited roles exist
        for parent_role in role.inherits_from:
            if parent_role not in self.roles:
                logger.error(f"Parent role does not exist: {parent_role}")
                return False

        self.roles[role.name] = role
        logger.info(f"Role created: {role.name}")
        return True

    def delete_role(self, role_name: str) -> bool:
        """
        Delete a role.
        
        Args:
            role_name: Name of role to delete
            
        Returns:
            True if successful
        """
        if role_name not in self.roles:
            logger.error(f"Role does not exist: {role_name}")
            return False

        # Check if role is assigned to any users
        users_with_role = [
            user_id for user_id, roles in self.user_roles.items()
            if role_name in roles
        ]

        if users_with_role:
            logger.error(f"Cannot delete role {role_name}: assigned to {len(users_with_role)} users")
            return False

        del self.roles[role_name]
        logger.info(f"Role deleted: {role_name}")
        return True

    def assign_role_to_user(self, user_id: str, role_name: str) -> bool:
        """
        Assign role to user.
        
        Args:
            user_id: User ID
            role_name: Role name to assign
            
        Returns:
            True if successful
        """
        if role_name not in self.roles:
            logger.error(f"Role does not exist: {role_name}")
            return False

        if user_id not in self.user_roles:
            self.user_roles[user_id] = []

        if role_name not in self.user_roles[user_id]:
            self.user_roles[user_id].append(role_name)
            logger.info(f"Role {role_name} assigned to user {user_id}")
            return True

        logger.warning(f"User {user_id} already has role {role_name}")
        return False

    def remove_role_from_user(self, user_id: str, role_name: str) -> bool:
        """
        Remove role from user.
        
        Args:
            user_id: User ID
            role_name: Role name to remove
            
        Returns:
            True if successful
        """
        if user_id not in self.user_roles:
            return False

        if role_name in self.user_roles[user_id]:
            self.user_roles[user_id].remove(role_name)
            logger.info(f"Role {role_name} removed from user {user_id}")
            return True

        return False

    def check_access(self, request: AccessRequest) -> AccessResult:
        """
        Check if user has access to perform action on resource.
        
        Args:
            request: Access request to evaluate
            
        Returns:
            AccessResult with decision and details
        """
        user_roles = self.user_roles.get(request.user_id, [])
        matched_permissions = []
        required_permission = f"{request.resource_type.value}:{request.resource_id}:{request.action.value}"

        # Check permissions for each user role
        for role_name in user_roles:
            if role_name not in self.roles:
                continue

            role = self.roles[role_name]
            if not role.is_active:
                continue

            # Get all permissions including inherited
            all_permissions = self._get_effective_permissions(role)

            for permission in all_permissions:
                if permission.matches(request.resource_type, request.resource_id, request.action):
                    # Check additional conditions
                    if self._check_permission_conditions(permission, request.context):
                        matched_permissions.append(str(permission))

        # Determine access result
        granted = len(matched_permissions) > 0

        if granted:
            reason = f"Access granted via permissions: {', '.join(matched_permissions)}"
        else:
            reason = f"Access denied: no matching permissions for {required_permission}"

        result = AccessResult(
            granted=granted,
            reason=reason,
            matched_permissions=matched_permissions,
            required_permissions=[required_permission],
            user_roles=user_roles
        )

        # Log access attempt
        self._log_access_attempt(request, result)

        return result

    def get_user_permissions(self, user_id: str) -> List[str]:
        """
        Get all permissions for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of permission strings
        """
        user_roles = self.user_roles.get(user_id, [])
        all_permissions = set()

        for role_name in user_roles:
            if role_name in self.roles:
                role = self.roles[role_name]
                effective_permissions = self._get_effective_permissions(role)
                for permission in effective_permissions:
                    all_permissions.add(str(permission))

        return list(all_permissions)

    def get_users_with_role(self, role_name: str) -> List[str]:
        """Get all users with a specific role."""
        return [
            user_id for user_id, roles in self.user_roles.items()
            if role_name in roles
        ]

    def get_access_logs(self, user_id: Optional[str] = None, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get access logs.
        
        Args:
            user_id: Filter by user ID (optional)
            hours: Hours of history to return
            
        Returns:
            List of access log entries
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        filtered_logs = []
        for log in self.access_logs:
            if log['timestamp'] < cutoff_time.isoformat():
                continue

            if user_id and log.get('user_id') != user_id:
                continue

            filtered_logs.append(log)

        return filtered_logs

    def _get_effective_permissions(self, role: Role) -> List[Permission]:
        """Get all permissions including inherited ones."""
        permissions = role.permissions.copy()

        # Add inherited permissions
        for parent_role_name in role.inherits_from:
            if parent_role_name in self.roles:
                parent_role = self.roles[parent_role_name]
                parent_permissions = self._get_effective_permissions(parent_role)
                permissions.extend(parent_permissions)

        return permissions

    def _check_permission_conditions(self, permission: Permission, context: Dict[str, Any]) -> bool:
        """Check if permission conditions are met."""
        if not permission.conditions:
            return True

        # Simple condition checking (extend as needed)
        for condition_key, condition_value in permission.conditions.items():
            if condition_key not in context:
                return False

            if context[condition_key] != condition_value:
                return False

        return True

    def _log_access_attempt(self, request: AccessRequest, result: AccessResult):
        """Log access attempt for auditing."""
        log_entry = {
            'timestamp': request.timestamp.isoformat(),
            'user_id': request.user_id,
            'resource_type': request.resource_type.value,
            'resource_id': request.resource_id,
            'action': request.action.value,
            'granted': result.granted,
            'reason': result.reason,
            'user_roles': result.user_roles,
            'matched_permissions': result.matched_permissions
        }

        self.access_logs.append(log_entry)

        # Keep only recent logs (last 10000 entries)
        if len(self.access_logs) > 10000:
            self.access_logs = self.access_logs[-10000:]

        if not result.granted:
            logger.warning(f"Access denied: {request.user_id} -> {request.resource_type.value}:{request.resource_id}:{request.action.value}")


def create_permission(resource_type: str, resource_id: Optional[str], action: str) -> Permission:
    """
    Helper function to create permission from strings.
    
    Args:
        resource_type: Resource type string
        resource_id: Resource ID (optional)
        action: Action string
        
    Returns:
        Permission object
    """
    return Permission(
        resource_type=ResourceType(resource_type),
        resource_id=resource_id,
        action=Action(action)
    )


def create_access_request(user_id: str, resource_type: str, resource_id: str, action: str) -> AccessRequest:
    """
    Helper function to create access request from strings.
    
    Args:
        user_id: User ID
        resource_type: Resource type string
        resource_id: Resource ID
        action: Action string
        
    Returns:
        AccessRequest object
    """
    return AccessRequest(
        user_id=user_id,
        resource_type=ResourceType(resource_type),
        resource_id=resource_id,
        action=Action(action)
    )


# CLI interface
def main():
    """CLI interface for RBAC testing."""
    import argparse

    parser = argparse.ArgumentParser(description="RBAC Authorization CLI")
    parser.add_argument("command", choices=["check", "assign", "list"])
    parser.add_argument("--user-id", help="User ID")
    parser.add_argument("--role", help="Role name")
    parser.add_argument("--resource-type", help="Resource type")
    parser.add_argument("--resource-id", help="Resource ID")
    parser.add_argument("--action", help="Action")

    args = parser.parse_args()

    # Initialize RBAC manager
    rbac = RBACManager()

    if args.command == "check":
        if not all([args.user_id, args.resource_type, args.resource_id, args.action]):
            print("Error: user-id, resource-type, resource-id, and action required")
            return

        request = create_access_request(
            args.user_id, args.resource_type, args.resource_id, args.action
        )

        result = rbac.check_access(request)

        print("Access check result:")
        print(f"  Granted: {result.granted}")
        print(f"  Reason: {result.reason}")
        print(f"  User roles: {result.user_roles}")
        print(f"  Matched permissions: {result.matched_permissions}")

    elif args.command == "assign":
        if not args.user_id or not args.role:
            print("Error: user-id and role required")
            return

        success = rbac.assign_role_to_user(args.user_id, args.role)
        if success:
            print(f"Role {args.role} assigned to user {args.user_id}")
        else:
            print("Role assignment failed")

    elif args.command == "list":
        if args.user_id:
            # List user permissions
            permissions = rbac.get_user_permissions(args.user_id)
            print(f"Permissions for user {args.user_id}:")
            for permission in permissions:
                print(f"  - {permission}")
        else:
            # List all roles
            print("Available roles:")
            for role_name, role in rbac.roles.items():
                print(f"  - {role_name}: {role.description}")


if __name__ == "__main__":
    main()
