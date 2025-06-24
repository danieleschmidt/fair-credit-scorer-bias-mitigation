import pytest


def main() -> None:
    """Run the project's test suite with coverage."""
    raise SystemExit(pytest.main(["-ra", "--cov=src", "--cov-report=term-missing"]))


if __name__ == "__main__":
    main()
