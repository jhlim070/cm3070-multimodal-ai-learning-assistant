"""
Integration tests for per-user document retrieval isolation.

This test uses the real embedding pipeline and a temporary ChromaDB instance
to verify that retrieval is isolated between users. A mock vector store would
not test the actual collection and retrieval behaviour used by the application.

Requirements:
    - sentence-transformers, with access to BAAI/bge-small-en
    - chromadb

The ChromaDB client is redirected to a temporary directory so that the test
does not read from or modify the application's persistent vector store.
"""
import pytest
import rag


@pytest.fixture
def temp_chroma(tmp_path, monkeypatch):
    import chromadb
    test_client = chromadb.PersistentClient(path=str(tmp_path / "chroma_test"))
    monkeypatch.setattr(rag, "client", test_client)
    yield test_client


@pytest.mark.checklist
def test_documents_are_isolated_per_user(temp_chroma):
    """
    Ensure that users cannot retrieve each other's documents.
    """
    rag.add_documents(
        "userA_bio",
        ["Photosynthesis converts light energy into chemical energy stored in glucose."],
    )
    rag.add_documents(
        "userB_physics",
        ["Newton's second law states that force equals mass times acceleration."],
    )

    results_a = rag.retrieve_documents("userA_bio", "What is photosynthesis?", top_k=3)
    results_b = rag.retrieve_documents("userB_physics", "What is Newton's second law?", top_k=3)

    assert any("Photosynthesis" in chunk for chunk in results_a)
    assert all("Newton" not in chunk for chunk in results_a)

    assert any("Newton" in chunk for chunk in results_b)
    assert all("Photosynthesis" not in chunk for chunk in results_b)


@pytest.mark.checklist
def test_new_user_collection_starts_empty(temp_chroma):
    # A new user should have no retrievable documents, even when other 
    # users already have data in the vector store.
    rag.add_documents("userC_existing", ["Some unrelated content about databases."])
    results = rag.retrieve_documents("userD_brand_new", "anything at all", top_k=3)
    assert results == []


@pytest.mark.checklist
def test_same_user_can_retrieve_own_multiple_documents(temp_chroma):
    rag.add_documents("userE", ["Binary search runs in O(log n) time."])
    rag.add_documents("userE", ["Bubble sort runs in O(n^2) time in the worst case."])
    results = rag.retrieve_documents("userE", "What is the time complexity of binary search?", top_k=3)
    assert any("Binary search" in chunk for chunk in results)