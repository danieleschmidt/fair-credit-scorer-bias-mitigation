import os
import sys
import subprocess  # nosec B404
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
