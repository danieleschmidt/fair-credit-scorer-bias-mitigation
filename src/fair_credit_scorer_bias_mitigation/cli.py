"""Command-line interface for running model evaluation."""
from src.evaluate_fairness import main as _main


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
