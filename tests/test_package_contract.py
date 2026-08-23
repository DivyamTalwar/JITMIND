from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_setup_is_metadata_free() -> None:
    setup_text = (ROOT / "setup.py").read_text()
    assert "requirements.txt" not in setup_text
    assert "install_requires" not in setup_text


def test_eval_documentation_matches_repository() -> None:
    readme = (ROOT / "eval" / "README.md").read_text()
    for task in ("hotpotqa", "locomo", "narrativeqa", "ruler"):
        assert (ROOT / "eval" / f"{task}_test.py").is_file()
        assert f"eval/run.py {task}" in readme
