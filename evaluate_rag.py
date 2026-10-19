import json
from pathlib import Path

from src.pipeline import RAGPipeline
from src.prompts import NOT_FOUND_MESSAGE


BASE_DIR = Path(__file__).resolve().parent
CASES_PATH = BASE_DIR / "evals" / "cases.json"


def matches_required_groups(answer: str, groups: list[list[str]]) -> bool:
    """Every group must have at least one matching alternative."""
    answer = answer.casefold()
    return all(
        any(term.casefold() in answer for term in group)
        for group in groups
    )


def main() -> None:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    pipeline = RAGPipeline()
    documents = pipeline.documents()

    if not documents:
        raise SystemExit(
            "No indexed documents found. Upload and index your test "
            "documents in the application first."
        )

    passed = 0
    answer_checks_passed = 0
    citation_checks_passed = 0

    print(f"Evaluation cases: {len(cases)}")
    print(f"Indexed documents: {documents}\n")

    for case in cases:
        case_id = case["id"]
        source = case["source"]

        if source not in documents:
            print(f"SKIP {case_id}: source not indexed: {source}")
            continue

        try:
            result = pipeline.ask(
                case["question"],
                sources={source},
            )
        except Exception as exc:
            print(f"FAIL {case_id}: pipeline error: {exc}")
            continue

        if case["should_answer"]:
            answer_ok = (
                result.answered
                and matches_required_groups(
                    result.text,
                    case["required_groups"],
                )
            )

            citation_ok = any(
                citation.source.casefold() == source.casefold()
                for citation in result.citations
            )
        else:
            answer_ok = (
                not result.answered
                and result.text.strip() == NOT_FOUND_MESSAGE
            )
            citation_ok = True  # No citation required for a refusal.

        answer_checks_passed += int(answer_ok)
        citation_checks_passed += int(citation_ok)
        case_passed = answer_ok and citation_ok
        passed += int(case_passed)

        status = "PASS" if case_passed else "FAIL"
        print(f"{status} {case_id}")
        print(f"  Answer check:   {'PASS' if answer_ok else 'FAIL'}")
        print(f"  Citation check: {'PASS' if citation_ok else 'FAIL'}")
        print(f"  Answer: {result.text[:250]}\n")

    total = len(cases)

    print("=" * 48)
    print(f"Overall cases passed: {passed}/{total}")
    print(f"Answer checks passed: {answer_checks_passed}/{total}")
    print(f"Citation/refusal checks passed: {citation_checks_passed}/{total}")

    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()