import pathlib


def test_diagram_exists():
    assert pathlib.Path("architecture/diagram.svg").exists()
