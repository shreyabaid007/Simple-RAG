"""Sanity-check script. Not a formal eval — just a smoke test.

Runs a handful of hardcoded questions: some answerable from AR6, some not.
Checks that out-of-scope questions correctly return the refusal string.

    python eval.py
"""

from config import REFUSAL_STRING
from rag import RAGPipeline

# (question, answerable_from_IPCC_AR6)
CASES: list[tuple[str, bool]] = [
    ("What has caused the observed warming of the climate system since 1850?", True),
    (
        "By how much has global surface temperature increased from 1850-1900 "
        "to 2011-2020?",
        True,
    ),
    ("What are options for reducing greenhouse gas emissions in the energy sector?", True),
    ("What is the current price of Bitcoin?", False),
    ("Who won the 2022 FIFA World Cup?", False),
]


def main() -> None:
    pipeline = RAGPipeline()
    passed = 0
    for question, answerable in CASES:
        print("=" * 80)
        print(f"Q: {question}")
        answer, _ = pipeline.answer(question)
        print(f"A: {answer}")

        is_refusal = answer.strip() == REFUSAL_STRING
        if answerable:
            ok = not is_refusal
            label = "PASS (answered)" if ok else "FAIL (refused on answerable Q)"
        else:
            ok = is_refusal
            label = "PASS (refused)" if ok else "FAIL (should have refused)"
        print(f"-> {label}")
        passed += int(ok)
        print()

    print(f"{passed}/{len(CASES)} cases passed.")


if __name__ == "__main__":
    main()
