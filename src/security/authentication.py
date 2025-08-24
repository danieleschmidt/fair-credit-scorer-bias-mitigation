"""
Authentication and session management for secure access.

Provides secure user authentication, session management, and token-based
access control for the fair credit scoring system.
"""

import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import bcrypt
import jwt

from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class User:
    """User account information."""
    user_id: str
    username: str
    email: str
    password_hash: str
    roles: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    is_active: bool = True
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None

    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            'user_id': self.user_id,
            'username': self.username,
            'email': self.email,
            'roles': self.roles,
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active,
            'failed_login_attempts': self.failed_login_attempts,
            'locked_until': self.locked_until.isoformat() if self.locked_until else None
        }

        if include_sensitive:
            data['password_hash'] = self.password_hash

        return data


@dataclass
class Token:
    """Authentication token."""
    token_id: str
    user_id: str
    token_type: str  # access, refresh, api_key
    token_value: str
    expires_at: datetime
    scopes: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None
    is_revoked: bool = False

    def is_expired(self) -> bool:
        """Check if token is expired."""
        return datetime.utcnow() > self.expires_at

    def is_valid(self) -> bool:
        """Check if token is valid."""
        return not self.is_revoked and not self.is_expired()

    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            'token_id': self.token_id,
            'user_id': self.user_id,
            'token_type': self.token_type,
            'expires_at': self.expires_at.isoformat(),
            'scopes': self.scopes,
            'created_at': self.created_at.isoformat(),
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'is_revoked': self.is_revoked,
            'is_expired': self.is_expired(),
            'is_valid': self.is_valid()
        }

        if include_sensitive:
            data['token_value'] = self.token_value

        return data


@dataclass
class Session:
    """User session information."""
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True

    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.utcnow() > self.expires_at

    def is_valid(self) -> bool:
        """Check if session is valid."""
        return self.is_active and not self.is_expired()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'is_active': self.is_active,
            'is_expired': self.is_expired(),
            'is_valid': self.is_valid()
        }


class AuthenticationManager:
    """
    Secure authentication management system.

    Handles user registration, login, password management, and
    multi-factor authentication with security best practices.
    """

    def __init__(
        self,
        secret_key: str,
        token_expiry_hours: int = 24,
        session_expiry_hours: int = 8,
        max_failed_attempts: int = 5,
        lockout_duration_minutes: int = 30
    ):
        """
        Initialize authentication manager.

        Args:
            secret_key: Secret key for token signing
            token_expiry_hours: Token expiration time
            session_expiry_hours: Session expiration time
            max_failed_attempts: Maximum failed login attempts
            lockout_duration_minutes: Account lockout duration
        """
        self.secret_key = secret_key
        self.token_expiry_hours = token_expiry_hours
        self.session_expiry_hours = session_expiry_hours
        self.max_failed_attempts = max_failed_attempts
        self.lockout_duration_minutes = lockout_duration_minutes

        # In-memory storage (replace with database in production)
        self.users: Dict[str, User] = {}
        self.tokens: Dict[str, Token] = {}
        self.sessions: Dict[str, Session] = {}

        # Password requirements
        self.password_min_length = 8
        self.password_require_uppercase = True
        self.password_require_lowercase = True
        self.password_require_digits = True
        self.password_require_special = True

        logger.info("AuthenticationManager initialized")

    def register_user(
        self,
        username: str,
        email: str,
        password: str,
        roles: Optional[List[str]] = None
    ) -> Optional[User]:
        """
        Register a new user.

        Args:
            username: Unique username
            email: User email address
            password: Plain text password
            roles: List of user roles

        Returns:
            User object if successful, None otherwise
        """
        # Validate inputs
        if not self._validate_username(username):
            logger.error("Invalid username format")
            return None

        if not self._validate_email(email):
            logger.error("Invalid email format")
            return None

        if not self._validate_password(password):
            logger.error("Password does not meet requirements")
            return None

        # Check if user already exists
        if self._user_exists(username, email):
            logger.error("User already exists")
            return None

        # Hash password
        password_hash = self._hash_password(password)

        # Create user
        user_id = self._generate_user_id()
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            roles=roles or ['user']
        )

        self.users[user_id] = user

        logger.info(f"User registered: {username}")
        return user

    def authenticate_user(
        self,
        username: str,
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Authenticate user credentials.

        Args:
            username: Username or email
            password: Plain text password
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            Authentication result with user and tokens
        """
        # Find user
        user = self._find_user_by_credentials(username)
        if not user:
            logger.warning(f"Authentication failed: user not found - {username}")
            return None

        # Check if account is locked
        if self._is_account_locked(user):
            logger.warning(f"Authentication failed: account locked - {username}")
            return None

        # Verify password
        if not self._verify_password(password, user.password_hash):
            self._handle_failed_login(user)
            logger.warning(f"Authentication failed: invalid password - {username}")
            return None

        # Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.utcnow()

        # Generate tokens
        access_token = self._generate_access_token(user)
        refresh_token = self._generate_refresh_token(user)

        # Create session
        session = self._create_session(user, ip_address, user_agent)

        logger.info(f"User authenticated: {username}")

        return {
            'user': user.to_dict(),
            'access_token': access_token.token_value,
            'refresh_token': refresh_token.token_value,
            'session_id': session.session_id,
            'expires_at': access_token.expires_at.isoformat()
        }

    def validate_token(self, token_value: str) -> Optional[Dict[str, Any]]:
        """
        Validate authentication token.

        Args:
            token_value: Token to validate

        Returns:
            Token validation result
        """
        try:
            # Decode JWT token
            payload = jwt.decode(token_value, self.secret_key, algorithms=['HS256'])

            token_id = payload.get('token_id')
            if not token_id or token_id not in self.tokens:
                return None

            token = self.tokens[token_id]

            # Check token validity
            if not token.is_valid():
                return None

            # Update last used
            token.last_used = datetime.utcnow()

            # Get user
            user = self.users.get(token.user_id)
            if not user or not user.is_active:
                return None

            return {
                'token': token.to_dict(),
                'user': user.to_dict(),
                'scopes': token.scopes
            }

        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return None

    def refresh_token(self, refresh_token_value: str) -> Optional[Dict[str, Any]]:
        """
        Refresh access token using refresh token.

        Args:
            refresh_token_value: Refresh token

        Returns:
            New access token or None
        """
        # Validate refresh token
        token_data = self.validate_token(refresh_token_value)
        if not token_data:
            return None

        token = self.tokens.get(token_data['token']['token_id'])
        if not token or token.token_type != 'refresh':
            return None

        # Get user
        user = self.users.get(token.user_id)
        if not user:
            return None

        # Generate new access token
        new_access_token = self._generate_access_token(user)

        logger.info(f"Token refreshed for user: {user.username}")

        return {
            'access_token': new_access_token.token_value,
            'expires_at': new_access_token.expires_at.isoformat()
        }

    def revoke_token(self, token_value: str) -> bool:
        """
        Revoke authentication token.

        Args:
            token_value: Token to revoke

        Returns:
            True if successful
        """
        try:
            payload = jwt.decode(token_value, self.secret_key, algorithms=['HS256'])
            token_id = payload.get('token_id')

            if token_id and token_id in self.tokens:
                self.tokens[token_id].is_revoked = True
                logger.info(f"Token revoked: {token_id}")
                return True

        except Exception as e:
            logger.error(f"Token revocation error: {e}")

        return False

    def change_password(
        self,
        user_id: str,
        current_password: str,
        new_password: str
    ) -> bool:
        """
        Change user password.

        Args:
            user_id: User ID
            current_password: Current password
            new_password: New password

        Returns:
            True if successful
        """
        user = self.users.get(user_id)
        if not user:
            return False

        # Verify current password
        if not self._verify_password(current_password, user.password_hash):
            logger.warning(f"Password change failed: invalid current password - {user.username}")
            return False

        # Validate new password
        if not self._validate_password(new_password):
            logger.error("New password does not meet requirements")
            return False

        # Update password
        user.password_hash = self._hash_password(new_password)

        # Revoke all existing tokens
        self._revoke_user_tokens(user_id)

        logger.info(f"Password changed for user: {user.username}")
        return True

    def _validate_username(self, username: str) -> bool:
        """Validate username format."""
        if not username or len(username) < 3 or len(username) > 50:
            return False

        # Allow alphanumeric, underscore, and hyphen
        import re
        return bool(re.match(r'^[a-zA-Z0-9_-]+$', username))

    def _validate_email(self, email: str) -> bool:
        """Validate email format."""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    def _validate_password(self, password: str) -> bool:
        """Validate password strength."""
        if len(password) < self.password_min_length:
            return False

        has_upper = any(c.isupper() for c in password) if self.password_require_uppercase else True
        has_lower = any(c.islower() for c in password) if self.password_require_lowercase else True
        has_digit = any(c.isdigit() for c in password) if self.password_require_digits else True
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password) if self.password_require_special else True

        return has_upper and has_lower and has_digit and has_special

    def _user_exists(self, username: str, email: str) -> bool:
        """Check if user already exists."""
        for user in self.users.values():
            if user.username == username or user.email == email:
                return True
        return False

    def _find_user_by_credentials(self, username_or_email: str) -> Optional[User]:
        """Find user by username or email."""
        for user in self.users.values():
            if user.username == username_or_email or user.email == username_or_email:
                return user
        return None

    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
        except Exception:
            return False

    def _is_account_locked(self, user: User) -> bool:
        """Check if user account is locked."""
        if user.locked_until and datetime.utcnow() < user.locked_until:
            return True

        # Reset lock if expired
        if user.locked_until and datetime.utcnow() >= user.locked_until:
            user.locked_until = None
            user.failed_login_attempts = 0

        return False

    def _handle_failed_login(self, user: User):
        """Handle failed login attempt."""
        user.failed_login_attempts += 1

        if user.failed_login_attempts >= self.max_failed_attempts:
            user.locked_until = datetime.utcnow() + timedelta(minutes=self.lockout_duration_minutes)
            logger.warning(f"Account locked due to failed attempts: {user.username}")

    def _generate_user_id(self) -> str:
        """Generate unique user ID."""
        return f"user_{secrets.token_hex(16)}"

    def _generate_token_id(self) -> str:
        """Generate unique token ID."""
        return f"token_{secrets.token_hex(16)}"

    def _generate_access_token(self, user: User) -> Token:
        """Generate access token for user."""
        token_id = self._generate_token_id()
        expires_at = datetime.utcnow() + timedelta(hours=self.token_expiry_hours)

        payload = {
            'token_id': token_id,
            'user_id': user.user_id,
            'username': user.username,
            'roles': user.roles,
            'token_type': 'access',
            'exp': expires_at.timestamp(),
            'iat': datetime.utcnow().timestamp()
        }

        token_value = jwt.encode(payload, self.secret_key, algorithm='HS256')

        token = Token(
            token_id=token_id,
            user_id=user.user_id,
            token_type='access',
            token_value=token_value,
            expires_at=expires_at,
            scopes=['read', 'write']
        )

        self.tokens[token_id] = token
        return token

    def _generate_refresh_token(self, user: User) -> Token:
        """Generate refresh token for user."""
        token_id = self._generate_token_id()
        expires_at = datetime.utcnow() + timedelta(days=30)  # Longer expiry for refresh

        payload = {
            'token_id': token_id,
            'user_id': user.user_id,
            'token_type': 'refresh',
            'exp': expires_at.timestamp(),
            'iat': datetime.utcnow().timestamp()
        }

        token_value = jwt.encode(payload, self.secret_key, algorithm='HS256')

        token = Token(
            token_id=token_id,
            user_id=user.user_id,
            token_type='refresh',
            token_value=token_value,
            expires_at=expires_at,
            scopes=['refresh']
        )

        self.tokens[token_id] = token
        return token

    def _create_session(
        self,
        user: User,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Session:
        """Create user session."""
        session_id = f"session_{secrets.token_hex(16)}"
        expires_at = datetime.utcnow() + timedelta(hours=self.session_expiry_hours)

        session = Session(
            session_id=session_id,
            user_id=user.user_id,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            last_activity=datetime.utcnow(),
            ip_address=ip_address,
            user_agent=user_agent
        )

        self.sessions[session_id] = session
        return session

    def _revoke_user_tokens(self, user_id: str):
        """Revoke all tokens for a user."""
        for token in self.tokens.values():
            if token.user_id == user_id:
                token.is_revoked = True


class SessionManager:
    """
    Session management system.

    Manages user sessions with automatic cleanup and security monitoring.
    """

    def __init__(self, cleanup_interval_minutes: int = 60):
        """
        Initialize session manager.

        Args:
            cleanup_interval_minutes: Interval for session cleanup
        """
        self.cleanup_interval_minutes = cleanup_interval_minutes
        self.sessions: Dict[str, Session] = {}

        logger.info("SessionManager initialized")

    def validate_session(self, session_id: str) -> Optional[Session]:
        """
        Validate session.

        Args:
            session_id: Session ID to validate

        Returns:
            Session if valid, None otherwise
        """
        session = self.sessions.get(session_id)
        if not session or not session.is_valid():
            return None

        # Update last activity
        session.last_activity = datetime.utcnow()

        return session

    def end_session(self, session_id: str) -> bool:
        """
        End user session.

        Args:
            session_id: Session ID to end

        Returns:
            True if successful
        """
        if session_id in self.sessions:
            self.sessions[session_id].is_active = False
            logger.info(f"Session ended: {session_id}")
            return True

        return False

    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if session.is_expired()
        ]

        for session_id in expired_sessions:
            del self.sessions[session_id]

        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

        return len(expired_sessions)

    def get_active_sessions(self, user_id: str) -> List[Session]:
        """Get active sessions for user."""
        return [
            session for session in self.sessions.values()
            if session.user_id == user_id and session.is_valid()
        ]


# CLI interface
def main():
    """CLI interface for authentication testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Authentication CLI")
    parser.add_argument("command", choices=["register", "login", "validate"])
    parser.add_argument("--username", help="Username")
    parser.add_argument("--email", help="Email address")
    parser.add_argument("--password", help="Password")
    parser.add_argument("--token", help="Token to validate")

    args = parser.parse_args()

    # Initialize authentication manager
    auth_manager = AuthenticationManager(secret_key="demo_secret_key_change_in_production")

    if args.command == "register":
        if not args.username or not args.email or not args.password:
            print("Error: username, email, and password required for registration")
            return

        user = auth_manager.register_user(args.username, args.email, args.password)
        if user:
            print(f"User registered successfully: {user.username}")
            print(f"User ID: {user.user_id}")
        else:
            print("Registration failed")

    elif args.command == "login":
        if not args.username or not args.password:
            print("Error: username and password required for login")
            return

        result = auth_manager.authenticate_user(args.username, args.password)
        if result:
            print("Login successful!")
            print(f"Access token: {result['access_token'][:50]}...")
            print(f"Expires at: {result['expires_at']}")
        else:
            print("Login failed")

    elif args.command == "validate":
        if not args.token:
            print("Error: token required for validation")
            return

        result = auth_manager.validate_token(args.token)
        if result:
            print("Token is valid!")
            print(f"User: {result['user']['username']}")
            print(f"Roles: {result['user']['roles']}")
        else:
            print("Token is invalid or expired")


if __name__ == "__main__":
    main()
