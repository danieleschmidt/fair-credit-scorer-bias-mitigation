"""Test runner with comprehensive development environment setup.

This module provides a complete test execution pipeline that automatically
sets up the development environment, installs dependencies, runs linting and
security checks, and executes the test suite with coverage reporting.

Features:
- Automatic dependency installation from requirements.txt
- Environment setup for local module imports without package installation
- Code quality checks using ruff linter
- Security scanning with bandit
- Test execution with pytest and coverage reporting
- Subprocess security hardening with shell=False

The runner ensures a clean testing environment by:
1. Installing project dependencies in editable mode
2. Installing development tools (ruff, bandit)
3. Running static analysis and security checks
4. Executing tests with coverage measurement

Usage:
    python -m src.run_tests

Environment Variables:
    PYTHONPATH: Automatically configured to include src directory

Exit Codes:
    0: All checks and tests passed
    Non-zero: Linting, security, or test failures occurred

The module follows security best practices by using subprocess.check_call
with shell=False and explicit command arrays to prevent shell injection attacks.
All external tool invocations are properly secured with nosec comments where
subprocess usage is intentional and validated.
"""

import os
import subprocess  # nosec B404
import sys

import pytest

# Ensure local src modules are importable without installation and for subprocesses
SRC_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, SRC_DIR)
os.environ["PYTHONPATH"] = os.pathsep.join(
    [SRC_DIR, os.environ.get("PYTHONPATH", "")]
)


def main() -> None:
    """Run the project's test suite with coverage."""
    project_root = os.path.abspath(os.path.join(SRC_DIR, os.pardir))
    requirements = os.path.join(project_root, "requirements.txt")
    if os.path.exists(requirements):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", requirements],
            shell=False,  # nosec B603
        )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", project_root],
        shell=False,  # nosec B603
    )

    # install dev tools for lint and security scanning
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "ruff==0.11.13", "bandit==1.8.5"],
        shell=False,  # nosec B603
    )
    subprocess.check_call(
        [sys.executable, "-m", "ruff", "check", "src", "--quiet"],
        shell=False,  # nosec B603 B607
    )
    subprocess.check_call(
        [sys.executable, "-m", "bandit", "-r", "src"],
        shell=False,  # nosec B603 B607
    )

    raise SystemExit(
        pytest.main(["-ra", "--cov=src", "--cov-report=term-missing"])
    )


if __name__ == "__main__":
    main()
