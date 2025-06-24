import pathlib


def test_plan_updated():
    plan = pathlib.Path("DEVELOPMENT_PLAN.md").read_text()
    assert "Phase 3" in plan
