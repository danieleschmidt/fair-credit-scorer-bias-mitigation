"""
Input validation and data sanitization for security.

Comprehensive input validation system to prevent injection attacks,
data corruption, and ensure data integrity in ML pipelines.
"""

import html
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from email_validator import EmailNotValidError, validate_email

from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationRule:
    """Input validation rule definition."""
    name: str
    validator: Callable[[Any], bool]
    error_message: str
    severity: str = "error"  # error, warning, info

    def validate(self, value: Any) -> tuple[bool, str]:
        """
        Validate a value against this rule.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            is_valid = self.validator(value)
            return is_valid, "" if is_valid else self.error_message
        except Exception as e:
            return False, f"Validation error: {str(e)}"


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sanitized_data: Optional[Any] = None
    validation_timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'has_sanitized_data': self.sanitized_data is not None,
            'validation_timestamp': self.validation_timestamp.isoformat()
        }


class InputValidator:
    """
    Comprehensive input validation system.
    
    Validates user inputs, API requests, and data uploads to prevent
    security vulnerabilities and ensure data quality.
    """

    def __init__(self):
        """Initialize input validator."""
        self.rules: Dict[str, List[ValidationRule]] = {}
        self.global_rules: List[ValidationRule] = []

        # Initialize common validation rules
        self._setup_common_rules()

        logger.info("InputValidator initialized")

    def _setup_common_rules(self):
        """Setup common validation rules."""
        # SQL injection prevention
        sql_injection_rule = ValidationRule(
            name="sql_injection",
            validator=lambda x: not self._contains_sql_injection(str(x)),
            error_message="Input contains potential SQL injection patterns",
            severity="error"
        )

        # XSS prevention
        xss_rule = ValidationRule(
            name="xss_prevention",
            validator=lambda x: not self._contains_xss(str(x)),
            error_message="Input contains potential XSS patterns",
            severity="error"
        )

        # Path traversal prevention
        path_traversal_rule = ValidationRule(
            name="path_traversal",
            validator=lambda x: not self._contains_path_traversal(str(x)),
            error_message="Input contains potential path traversal patterns",
            severity="error"
        )

        # Command injection prevention
        command_injection_rule = ValidationRule(
            name="command_injection",
            validator=lambda x: not self._contains_command_injection(str(x)),
            error_message="Input contains potential command injection patterns",
            severity="error"
        )

        self.global_rules.extend([
            sql_injection_rule,
            xss_rule,
            path_traversal_rule,
            command_injection_rule
        ])

    def add_rule(self, field_name: str, rule: ValidationRule):
        """
        Add validation rule for a specific field.
        
        Args:
            field_name: Name of the field to validate
            rule: Validation rule to add
        """
        if field_name not in self.rules:
            self.rules[field_name] = []

        self.rules[field_name].append(rule)
        logger.info(f"Added validation rule '{rule.name}' for field '{field_name}'")

    def add_global_rule(self, rule: ValidationRule):
        """
        Add global validation rule that applies to all inputs.
        
        Args:
            rule: Validation rule to add
        """
        self.global_rules.append(rule)
        logger.info(f"Added global validation rule '{rule.name}'")

    def validate_input(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate input data against defined rules.
        
        Args:
            data: Input data to validate
            
        Returns:
            ValidationResult with validation status and messages
        """
        errors = []
        warnings = []

        # Validate each field
        for field_name, value in data.items():
            # Apply field-specific rules
            if field_name in self.rules:
                for rule in self.rules[field_name]:
                    is_valid, error_msg = rule.validate(value)
                    if not is_valid:
                        if rule.severity == "error":
                            errors.append(f"{field_name}: {error_msg}")
                        elif rule.severity == "warning":
                            warnings.append(f"{field_name}: {error_msg}")

            # Apply global rules
            for rule in self.global_rules:
                is_valid, error_msg = rule.validate(value)
                if not is_valid:
                    if rule.severity == "error":
                        errors.append(f"{field_name}: {error_msg}")
                    elif rule.severity == "warning":
                        warnings.append(f"{field_name}: {error_msg}")

        is_valid = len(errors) == 0

        result = ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings
        )

        if not is_valid:
            logger.warning(f"Input validation failed: {len(errors)} errors, {len(warnings)} warnings")

        return result

    def validate_dataframe(self, df: pd.DataFrame, schema: Optional[Dict[str, str]] = None) -> ValidationResult:
        """
        Validate pandas DataFrame for ML pipeline.
        
        Args:
            df: DataFrame to validate
            schema: Optional schema definition
            
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        # Basic DataFrame validation
        if df.empty:
            errors.append("DataFrame is empty")

        if df.isnull().any().any():
            null_columns = df.columns[df.isnull().any()].tolist()
            warnings.append(f"DataFrame contains null values in columns: {null_columns}")

        # Check for duplicate rows
        if df.duplicated().any():
            duplicate_count = df.duplicated().sum()
            warnings.append(f"DataFrame contains {duplicate_count} duplicate rows")

        # Schema validation if provided
        if schema:
            for column, expected_type in schema.items():
                if column not in df.columns:
                    errors.append(f"Required column '{column}' is missing")
                else:
                    actual_type = str(df[column].dtype)
                    if not self._types_compatible(actual_type, expected_type):
                        errors.append(f"Column '{column}' has type '{actual_type}', expected '{expected_type}'")

        # Check for suspicious patterns in string columns
        string_columns = df.select_dtypes(include=['object']).columns
        for column in string_columns:
            for _, value in df[column].items():
                if pd.notna(value):
                    for rule in self.global_rules:
                        is_valid, error_msg = rule.validate(str(value))
                        if not is_valid:
                            errors.append(f"Column '{column}': {error_msg}")
                            break  # Only report first violation per cell

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings
        )

    def validate_email(self, email: str) -> ValidationResult:
        """Validate email address."""
        errors = []

        try:
            # Use email-validator library
            valid = validate_email(email)
            normalized_email = valid.email
        except EmailNotValidError as e:
            errors.append(f"Invalid email address: {str(e)}")
            normalized_email = None

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            sanitized_data=normalized_email
        )

    def validate_phone_number(self, phone: str) -> ValidationResult:
        """Validate phone number format."""
        errors = []
        sanitized_phone = None

        # Remove all non-digit characters
        digits_only = re.sub(r'\D', '', phone)

        # Check length and format
        if len(digits_only) < 10:
            errors.append("Phone number too short")
        elif len(digits_only) > 15:
            errors.append("Phone number too long")
        else:
            # Format as +1 (XXX) XXX-XXXX for US numbers
            if len(digits_only) == 10:
                sanitized_phone = f"+1 ({digits_only[:3]}) {digits_only[3:6]}-{digits_only[6:]}"
            elif len(digits_only) == 11 and digits_only[0] == '1':
                sanitized_phone = f"+1 ({digits_only[1:4]}) {digits_only[4:7]}-{digits_only[7:]}"
            else:
                sanitized_phone = f"+{digits_only}"

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            sanitized_data=sanitized_phone
        )

    def validate_ssn(self, ssn: str) -> ValidationResult:
        """Validate Social Security Number format."""
        errors = []

        # Remove all non-digit characters
        digits_only = re.sub(r'\D', '', ssn)

        # Check format
        if len(digits_only) != 9:
            errors.append("SSN must be exactly 9 digits")
        elif digits_only.startswith('000') or digits_only[3:5] == '00' or digits_only[5:] == '0000':
            errors.append("Invalid SSN format")
        else:
            # Additional validation for known invalid SSN patterns
            invalid_patterns = ['123456789', '111111111', '222222222', '333333333']
            if digits_only in invalid_patterns:
                errors.append("SSN appears to be invalid or test data")

        # Don't return sanitized SSN for security reasons
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors
        )

    def _contains_sql_injection(self, text: str) -> bool:
        """Check for SQL injection patterns."""
        sql_patterns = [
            r"(\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b)",
            r"('|(\\))(|(\\\\))('|(\\\")|$)",
            r"(\b(or|and)\b\s+\b\d+\s*[=<>]\s*\d+)",
            r"(--|#|/\*|\*/)",
        ]

        text_lower = text.lower()
        for pattern in sql_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True

        return False

    def _contains_xss(self, text: str) -> bool:
        """Check for XSS patterns."""
        xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"vbscript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>",
        ]

        text_lower = text.lower()
        for pattern in xss_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True

        return False

    def _contains_path_traversal(self, text: str) -> bool:
        """Check for path traversal patterns."""
        path_patterns = [
            r"\.\./",
            r"\.\.\\",
            r"~",
            r"/etc/passwd",
            r"/proc/",
            r"\\windows\\",
        ]

        text_lower = text.lower()
        for pattern in path_patterns:
            if pattern in text_lower:
                return True

        return False

    def _contains_command_injection(self, text: str) -> bool:
        """Check for command injection patterns."""
        command_patterns = [
            r"[;&|`]",
            r"\$\(",
            r">\s*/",
            r"<\s*/",
            r"rm\s+-",
            r"curl\s+",
            r"wget\s+",
        ]

        for pattern in command_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def _types_compatible(self, actual_type: str, expected_type: str) -> bool:
        """Check if data types are compatible."""
        type_mappings = {
            'int64': ['int', 'integer', 'number'],
            'float64': ['float', 'number', 'decimal'],
            'object': ['str', 'string', 'text'],
            'bool': ['boolean', 'bool'],
            'datetime64[ns]': ['datetime', 'timestamp']
        }

        if actual_type == expected_type:
            return True

        if actual_type in type_mappings:
            return expected_type.lower() in type_mappings[actual_type]

        return False


class DataSanitizer:
    """
    Data sanitization and cleaning utilities.
    
    Cleans and sanitizes data to prevent security vulnerabilities
    and ensure data quality.
    """

    def __init__(self):
        """Initialize data sanitizer."""
        logger.info("DataSanitizer initialized")

    def sanitize_string(self, text: str, allow_html: bool = False) -> str:
        """
        Sanitize string input.
        
        Args:
            text: Input text to sanitize
            allow_html: Whether to allow HTML tags
            
        Returns:
            Sanitized string
        """
        if not isinstance(text, str):
            text = str(text)

        # Remove null bytes
        text = text.replace('\x00', '')

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        if not allow_html:
            # Escape HTML entities
            text = html.escape(text)

        # Remove control characters except tabs and newlines
        text = re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

        return text

    def sanitize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sanitize pandas DataFrame.
        
        Args:
            df: DataFrame to sanitize
            
        Returns:
            Sanitized DataFrame
        """
        df_clean = df.copy()

        # Sanitize string columns
        string_columns = df_clean.select_dtypes(include=['object']).columns
        for column in string_columns:
            df_clean[column] = df_clean[column].apply(
                lambda x: self.sanitize_string(str(x)) if pd.notna(x) else x
            )

        # Remove rows with all NaN values
        df_clean = df_clean.dropna(how='all')

        # Handle infinite values
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_columns] = df_clean[numeric_columns].replace([np.inf, -np.inf], np.nan)

        logger.info(f"Sanitized DataFrame: {len(df)} -> {len(df_clean)} rows")

        return df_clean

    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for safe storage.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove path components
        filename = filename.split('/')[-1].split('\\')[-1]

        # Remove dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)

        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')

        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            max_name_length = 255 - len(ext) - 1 if ext else 255
            filename = name[:max_name_length] + ('.' + ext if ext else '')

        # Ensure not empty
        if not filename:
            filename = 'sanitized_file'

        return filename

    def remove_pii(self, text: str) -> str:
        """
        Remove common PII patterns from text.
        
        Args:
            text: Text that may contain PII
            
        Returns:
            Text with PII removed/masked
        """
        # SSN pattern
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
        text = re.sub(r'\b\d{9}\b', '[SSN]', text)

        # Credit card patterns
        text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CREDIT_CARD]', text)

        # Phone number patterns
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)

        # Email patterns
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)

        return text

    def mask_sensitive_data(self, data: Dict[str, Any], sensitive_fields: List[str]) -> Dict[str, Any]:
        """
        Mask sensitive fields in data dictionary.
        
        Args:
            data: Data dictionary
            sensitive_fields: List of field names to mask
            
        Returns:
            Data with sensitive fields masked
        """
        masked_data = data.copy()

        for field in sensitive_fields:
            if field in masked_data:
                value = masked_data[field]
                if isinstance(value, str) and len(value) > 0:
                    # Show only first and last character for strings
                    if len(value) > 2:
                        masked_data[field] = value[0] + '*' * (len(value) - 2) + value[-1]
                    else:
                        masked_data[field] = '*' * len(value)
                else:
                    masked_data[field] = '[MASKED]'

        return masked_data


def create_credit_score_validator() -> InputValidator:
    """
    Create validator specifically for credit scoring data.
    
    Returns:
        Configured InputValidator for credit scoring
    """
    validator = InputValidator()

    # Credit score validation
    credit_score_rule = ValidationRule(
        name="credit_score_range",
        validator=lambda x: isinstance(x, (int, float)) and 300 <= x <= 850,
        error_message="Credit score must be between 300 and 850"
    )
    validator.add_rule("credit_score", credit_score_rule)

    # Income validation
    income_rule = ValidationRule(
        name="income_positive",
        validator=lambda x: isinstance(x, (int, float)) and x >= 0,
        error_message="Income must be non-negative"
    )
    validator.add_rule("income", income_rule)
    validator.add_rule("annual_income", income_rule)

    # Age validation
    age_rule = ValidationRule(
        name="age_range",
        validator=lambda x: isinstance(x, (int, float)) and 18 <= x <= 120,
        error_message="Age must be between 18 and 120"
    )
    validator.add_rule("age", age_rule)

    # Loan amount validation
    loan_amount_rule = ValidationRule(
        name="loan_amount_positive",
        validator=lambda x: isinstance(x, (int, float)) and x > 0,
        error_message="Loan amount must be positive"
    )
    validator.add_rule("loan_amount", loan_amount_rule)
    validator.add_rule("requested_amount", loan_amount_rule)

    # Employment status validation
    employment_status_rule = ValidationRule(
        name="employment_status_valid",
        validator=lambda x: str(x).lower() in ['employed', 'unemployed', 'self-employed', 'retired', 'student'],
        error_message="Employment status must be one of: employed, unemployed, self-employed, retired, student"
    )
    validator.add_rule("employment_status", employment_status_rule)

    logger.info("Credit score validator created")
    return validator


# CLI interface
def main():
    """CLI interface for validation testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Input Validation CLI")
    parser.add_argument("command", choices=["validate", "sanitize", "demo"])
    parser.add_argument("--input", help="Input string to validate/sanitize")
    parser.add_argument("--type", choices=["email", "phone", "ssn"], help="Validation type")

    args = parser.parse_args()

    if args.command == "validate":
        validator = InputValidator()

        if args.type == "email" and args.input:
            result = validator.validate_email(args.input)
            print(f"Email validation: {result.is_valid}")
            if result.errors:
                print(f"Errors: {result.errors}")
            if result.sanitized_data:
                print(f"Sanitized: {result.sanitized_data}")

        elif args.type == "phone" and args.input:
            result = validator.validate_phone_number(args.input)
            print(f"Phone validation: {result.is_valid}")
            if result.errors:
                print(f"Errors: {result.errors}")
            if result.sanitized_data:
                print(f"Sanitized: {result.sanitized_data}")

        else:
            # General validation
            test_data = {"input": args.input} if args.input else {"test": "sample data"}
            result = validator.validate_input(test_data)
            print(f"Validation result: {result.is_valid}")
            if result.errors:
                print(f"Errors: {result.errors}")
            if result.warnings:
                print(f"Warnings: {result.warnings}")

    elif args.command == "sanitize":
        sanitizer = DataSanitizer()

        if args.input:
            sanitized = sanitizer.sanitize_string(args.input)
            print(f"Original: {args.input}")
            print(f"Sanitized: {sanitized}")
        else:
            print("Please provide --input for sanitization")

    elif args.command == "demo":
        # Demo validation
        print("Running validation demo...")

        validator = create_credit_score_validator()

        # Test data
        test_cases = [
            {"credit_score": 750, "income": 50000, "age": 30},
            {"credit_score": 200, "income": -1000, "age": 15},  # Invalid
            {"credit_score": "750", "income": "50000", "age": "30"},  # String inputs
        ]

        for i, test_data in enumerate(test_cases, 1):
            print(f"\nTest case {i}: {test_data}")
            result = validator.validate_input(test_data)
            print(f"  Valid: {result.is_valid}")
            if result.errors:
                print(f"  Errors: {result.errors}")
            if result.warnings:
                print(f"  Warnings: {result.warnings}")


if __name__ == "__main__":
    main()
