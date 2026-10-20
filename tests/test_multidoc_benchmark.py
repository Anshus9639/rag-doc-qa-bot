"""Static checks for the synthetic multi-document evaluation dataset."""

import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
FIXTURES_DIR = BASE_DIR / "evals" / "fixtures"
CASES_PATH = BASE_DIR / "evals" / "multidoc_cases.json"


def test_multidoc_benchmark_has_multiple_synthetic_documents():
    fixtures = sorted(FIXTURES_DIR.glob("*.txt"))
    assert len(fixtures) >= 6
    assert len({path.name for path in fixtures}) == len(fixtures)


def test_multidoc_cases_have_answerable_and_unanswerable_questions():
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    assert len(cases) >= 10
    assert len({case["id"] for case in cases}) == len(cases)
    assert any(case.get("should_answer") for case in cases)
    assert any(not case.get("should_answer") for case in cases)


def test_answerable_cases_point_to_real_fixture_headings():
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    fixtures = {
        path.name: path.read_text(encoding="utf-8")
        for path in FIXTURES_DIR.glob("*.txt")
    }

    for case in cases:
        if not case.get("should_answer"):
            continue

        assert case["expected_source"] in fixtures, case["id"]
        assert case["expected_section_contains"] in fixtures[case["expected_source"]], case["id"]
        assert case["required_groups"], case["id"]
        assert case["evidence_groups"], case["id"]


def test_unanswerable_cases_do_not_claim_expected_evidence():
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))

    for case in cases:
        if case.get("should_answer"):
            continue
        assert not case.get("required_groups")
        assert not case.get("evidence_groups")
