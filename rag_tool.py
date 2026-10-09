"""
rag_tool.py
-----------
Builds the RAG retrieval tool for the university IT support agent.

Stack (from Lecture 11):
  - OllamaEmbeddings with the "nomic-embed-text" model
  - a Chroma vector store (collection "university_support")
  - similarity_search for retrieval

The knowledge base is loaded ONLY through load_policy_documents()
from policy_loader.py (which must not be modified).
"""

import os

# The Ollama desktop app can export a random ephemeral OLLAMA_HOST in some
# shells. Pin the standard endpoint BEFORE importing any Ollama client so it
# cannot override the explicit base_url used below.
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.tools import tool

from policy_loader import load_policy_documents

# Model used to embed text (NOT the generator LLM).
EMBEDDING_MODEL = "nomic-embed-text"
COLLECTION_NAME = "university_support"

# Pin the standard Ollama endpoint explicitly.
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# Number of policy excerpts retrieved per question (Lecture 11 used k=2;
# k=4 gives the small 0.6B generator a little more grounded context).
TOP_K = 4

# TO LOAD THE DOCUMENT, USE THE FOLLOWING ONLY:
documents = load_policy_documents()

# LangChain interface to the Ollama-hosted embedding model.
embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL,
)

# In-memory Chroma collection; every run re-embeds the single policy file.
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)

# Each Document -> embedding model -> vector -> Chroma.
# Embedded in small batches: sending all 1,196 documents in one embed request
# makes the Ollama server pre-tokenize the whole batch at once, which is both
# memory-hungry and fragile across Ollama versions.
EMBED_BATCH_SIZE = 32
for start in range(0, len(documents), EMBED_BATCH_SIZE):
    vector_store.add_documents(documents[start:start + EMBED_BATCH_SIZE])
    print(
        f"Embedded {min(start + EMBED_BATCH_SIZE, len(documents))}"
        f"/{len(documents)} policy documents"
    )


def retrieve_policy(question: str):
    """Return the top matching policy Documents for a question."""
    return vector_store.similarity_search(question, k=TOP_K)


def format_documents(docs) -> str:
    """Render retrieved Documents as cited context blocks."""
    if not docs:
        return "NO RELEVANT POLICY DOCUMENTS WERE FOUND."

    blocks = []
    for doc in docs:
        metadata = doc.metadata
        blocks.append(
            f"[Policy {metadata.get('policy_id', '?')} "
            f"topic={metadata.get('topic', '?')} "
            f"source={metadata.get('source', '?')}]\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(blocks)


@tool
def search_policy(question: str) -> str:
    """Search the university IT policy knowledge base.

    Use this tool for EVERY university IT support question, including
    requests for contact details such as the IT Service Desk phone number.
    Returns the most relevant policy excerpts with their policy IDs.
    If no relevant excerpt exists, the requested information is not in
    the knowledge base.
    """
    docs = retrieve_policy(question)
    return format_documents(docs)
