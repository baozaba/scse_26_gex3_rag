"""
agent.py
--------
The university IT support agent for Graded Exercise #3.

Architecture (Lecture 11 - RAG + Reflection):

  User question
       |
  [Retrieve]  search_policy tool (Chroma + nomic-embed-text)
       |
  First LLM call   -> Draft answer        (qwen3:0.6b as Support Agent)
       |
  Second LLM call  -> PASS / REVISE       (qwen3:0.6b as Reviewer)
       |
  If REVISE:
  Third LLM call   -> Improved answer     (qwen3:0.6b as Reviser)

A deterministic harness additionally rejects any phone number that is not
present in the retrieved context, so the model can never hallucinate one.

The generator model is qwen3:0.6b (NOT the qwen3:8b used in Lecture 11).
"""

import os
import re

# Pin the standard Ollama endpoint before importing the Ollama client.
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from rag_tool import search_policy, retrieve_policy, format_documents

# Required by the exercise: use the 0.6B model, not the 8B lecture model.
MODEL_NAME = "qwen3:0.6b"

# Pin the standard Ollama endpoint explicitly.
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

SYSTEM_PROMPT = (
    "You are a university IT support assistant. "
    "Answer using ONLY the policy excerpts provided in the RETRIEVED CONTEXT. "
    "If the requested information is not present in the RETRIEVED CONTEXT, "
    "say clearly that you do not know and that the information is not "
    "available in the university IT policy knowledge base. "
    "Never invent, guess, or fabricate phone numbers or any other contact "
    "detail that is not written verbatim in the RETRIEVED CONTEXT. "
    "Keep the answer concise."
)

REVIEWER_PROMPT = (
    "You are a strict reviewer of a university IT support answer. "
    "Decide whether the draft is fully supported by the RETRIEVED CONTEXT. "
    "Reply with exactly one word: PASS or REVISE. "
    "Reply REVISE if the draft invents a phone number or any other fact that "
    "is not written verbatim in the context, if it answers a question the "
    "context does not cover, or if it gives irrelevant information. "
    "A request for a phone number that is absent from the context MUST be "
    "answered by saying the number is not available, so REVISE any draft "
    "that contains a made-up number."
)

REVISER_PROMPT = (
    "You are revising a university IT support answer that failed review. "
    "Rewrite it using ONLY the RETRIEVED CONTEXT. Remove every unsupported "
    "fact. If the requested information (such as a phone number) is absent "
    "from the context, state plainly that you do not know the number and "
    "that it is not available in the university IT policy knowledge base. "
    "Do not invent anything and keep the answer concise."
)

# Matches common phone-number shapes, e.g. 800-555-1234, (800) 555 1234,
# 1-800-555-1234 or any run of 7+ digits.
PHONE_PATTERN = re.compile(
    r"(?:\+?\d[\d\-.() ]{6,}\d)|(?:\d{3,}\-\d{3,}\-\d{2,})"
)


def _contains_unsupported_number(answer: str, context: str) -> bool:
    """A phone number in the answer must also appear verbatim in the context."""
    numbers = [match.group().strip() for match in PHONE_PATTERN.finditer(answer)]
    for number in numbers:
        digits = re.sub(r"\D", "", number)
        if len(digits) < 7:
            continue
        if number not in context and digits not in re.sub(r"\D", "", context):
            return True
    return False


def _llm():
    # Greedy decoding: no random creativity that could produce hallucinations.
    return ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        num_predict=256,
    )


def run_agent(question: str, verbose: bool = True) -> str:
    """Run retrieve -> draft -> review -> (revise) for one user question."""

    # --- RETRIEVAL STEP -----------------------------------------------------
    # Retrieval is performed up front as a harness so that every answer is
    # grounded, even though the 0.6B model also has the tool bound.
    docs = retrieve_policy(question)
    context = format_documents(docs)

    if verbose:
        print("\n----- RETRIEVED CONTEXT -----")
        print(context)
        print("-----------------------------\n")

    llm = _llm().bind_tools([search_policy])

    # --- FIRST LLM CALL: DRAFT ----------------------------------------------
    draft_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"RETRIEVED CONTEXT:\n{context}\n\n"
                f"USER QUESTION: {question}"
            )
        ),
    ]
    draft = llm.invoke(draft_messages).content.strip()

    # Hard harness: a fabricated phone number forces revision regardless of
    # what the small model's self-review decides.
    unsupported_number = _contains_unsupported_number(draft, context)

    # --- SECOND LLM CALL: REVIEW --------------------------------------------
    review = llm.invoke(
        [
            SystemMessage(content=REVIEWER_PROMPT),
            HumanMessage(
                content=(
                    f"RETRIEVED CONTEXT:\n{context}\n\n"
                    f"USER QUESTION: {question}\n\n"
                    f"DRAFT ANSWER: {draft}"
                )
            ),
        ]
    ).content.strip().upper()

    if verbose:
        print("DRAFT:", draft)
        print("REVIEW:", review)
        if unsupported_number:
            print("HARNESS: unsupported phone number detected -> REVISE")

    # --- THIRD LLM CALL (ONLY IF NEEDED): REVISE ----------------------------
    if unsupported_number or not review.startswith("PASS"):
        final = llm.invoke(
            [
                SystemMessage(content=REVISER_PROMPT),
                HumanMessage(
                    content=(
                        f"RETRIEVED CONTEXT:\n{context}\n\n"
                        f"USER QUESTION: {question}\n\n"
                        f"DRAFT ANSWER TO FIX: {draft}"
                    )
                ),
            ]
        ).content.strip()

        # Final safety net: if the revised answer still invents a number,
        # replace it with the canonical refusal.
        if _contains_unsupported_number(final, context):
            final = (
                "I do not know the exact phone number for the University "
                "IT Service Desk; it is not available in the university IT "
                "policy knowledge base, so I cannot provide it."
            )
    else:
        final = draft

    return final
