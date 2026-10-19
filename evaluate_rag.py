import json
from pathlib import Path

from src.pipeline import RAGPipeline
from src.prompts import NOT_FOUND_MESSAGE


BASE_DIR = Path(__file__).resolve().parent
CASES_PATH = BASE_DIR / "evals" / "cases.json"


def matches_required_groups(
    text: str,
    groups: list[list[str]],
) -> bool:
    """Every group must have at least one matching alternative."""
    text = text.casefold()

    return all(
        any(term.casefold() in text for term in group)
        for group in groups
    )


def main() -> None:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    pipeline = RAGPipeline()
    documents = pipeline.documents()

    if not documents:
        raise SystemExit(
            "No indexed documents found. Index your test document first."
        )

    passed = 0
    evaluated = 0
    skipped = 0
    answer_checks_passed = 0
    citation_checks_passed = 0
    evidence_checks_passed = 0

    print(f"Evaluation cases: {len(cases)}")
    print(f"Indexed documents: {documents}\n")

    for case in cases:
        case_id = case["id"]
        source = case["source"]

        if source not in documents:
            skipped += 1
            print(f"SKIP {case_id}: source not indexed: {source}")
            continue

        evaluated += 1

        try:
            result = pipeline.ask(
                case["question"],
                sources={source},
            )
        except Exception as exc:
            print(f"FAIL {case_id}: pipeline error: {exc}\n")
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

            # Inspect the text of the chunks actually shown as citations.
            cited_evidence = " ".join(
                citation.snippet
                for citation in result.citations
            )

            evidence_ok = matches_required_groups(
                cited_evidence,
                case.get("evidence_groups", []),
            )

        else:
            answer_ok = (
                not result.answered
                and result.text.strip() == NOT_FOUND_MESSAGE
            )

            # A refusal does not need a supporting citation.
            citation_ok = True
            evidence_ok = True

        answer_checks_passed += int(answer_ok)
        citation_checks_passed += int(citation_ok)
        evidence_checks_passed += int(evidence_ok)

        case_passed = answer_ok and citation_ok and evidence_ok
        passed += int(case_passed)

        print(f"{'PASS' if case_passed else 'FAIL'} {case_id}")
        print(f"  Answer check:   {'PASS' if answer_ok else 'FAIL'}")
        print(f"  Citation check: {'PASS' if citation_ok else 'FAIL'}")
        print(f"  Evidence check: {'PASS' if evidence_ok else 'FAIL'}")
        print(f"  Answer: {result.text[:250]}")

        for citation in result.citations:
            print(
                f"  Cited source: {citation.source}, "
                f"page={citation.page}, section={citation.section}"
            )
            print(f"  Evidence: {citation.snippet}")

        print()

    print("=" * 48)
    print(f"Cases configured: {len(cases)}")
    print(f"Cases evaluated: {evaluated}")
    print(f"Cases skipped: {skipped}")
    print(f"Cases passed: {passed}/{evaluated}")
    print(f"Answer checks passed: {answer_checks_passed}/{evaluated}")
    print(f"Citation checks passed: {citation_checks_passed}/{evaluated}")
    print(f"Evidence checks passed: {evidence_checks_passed}/{evaluated}")

    if evaluated == 0 or skipped > 0 or passed != evaluated:
        raise SystemExit(1)


if __name__ == "__main__":
    main()