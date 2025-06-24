import pathlib


def test_architecture_review_generated():
    assert pathlib.Path("architecture/architecture_review.md").exists()
    assert pathlib.Path("architecture/diagram.svg").exists()
