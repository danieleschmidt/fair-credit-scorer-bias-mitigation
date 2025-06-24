import pathlib


def test_summary_exists():
    assert pathlib.Path("architecture/architecture_review.md").exists()
