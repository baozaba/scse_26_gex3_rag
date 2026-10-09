"""
rag.py
------
Entry point for the RAG part of Graded Exercise #3.

Run:
    python rag.py

The agent must answer the Service Desk phone-number question by stating that
it does not know the number (the single knowledge-base file contains no
verified phone number), without inventing one or giving an irrelevant reply.
"""

from agent import run_agent

# The graded challenge question.
QUESTION = (
    "What exact phone number should I call for the "
    "University IT Service Desk?"
)

# A few ordinary questions are included to show the RAG agent still answers
# questions that ARE covered by the knowledge base.
SANITY_QUESTIONS = [
    "How long is an unreleased print job kept before it is deleted?",
    "My account got locked after several failed logins. What should I do?",
]


def main():
    print("=" * 70)
    print("CHALLENGE QUESTION")
    print("=" * 70)
    answer = run_agent(QUESTION)
    print("\nFINAL ANSWER:")
    print(answer)

    for question in SANITY_QUESTIONS:
        print("\n" + "=" * 70)
        print("SANITY QUESTION:", question)
        print("=" * 70)
        answer = run_agent(question)
        print("\nFINAL ANSWER:")
        print(answer)


if __name__ == "__main__":
    main()
