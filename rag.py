"""
Retrieval-Augmented Generation (RAG) vector-store utilities for the 
multimodal AI learning assistant

This module manages per-user ChromaDB collections, configurable embedding
models, document indexing, semantic retrieval, source-aware retrieval, and
text chunking. The default configuration preserves the application's existing
behaviour, while additional embedding models and chunking strategies support
controlled retrieval experiments.
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer
import uuid

from validation import sanitise_for_collection

# Allow tests and experiments to use an isolated ChromaDB directory while
# preserving the application's existing persistent store by default.
client = chromadb.PersistentClient(path=os.environ.get("CHROMA_DB_PATH", "./chroma_db"))


def _normalize_user_id(user_id):
    """
    Normalise the user ID consistently with db.py so that the same logical user
    maps to the same SQLite profile and ChromaDB collection.
    """
    return (user_id or "").strip().lower()

# ---------------------------------------------------
# Embedding models (used for Experiment 1, Table 1)
# ---------------------------------------------------
"""
Prefix configuration follows the retrieval conventions specified by each
model. Query and document prefixes are stored separately because some models
require different instructions for the two input types.
"""
DEFAULT_EMBEDDING_MODEL = "bge-small"

EMBEDDING_MODELS = {
    "bge-small": {
        "model_name": "BAAI/bge-small-en",
        "trust_remote_code": False,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "document_prefix": "",
    },
    "minilm": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "trust_remote_code": False,
        "query_prefix": "",
        "document_prefix": "",
    },
    "nomic": {
        "model_name": "nomic-ai/nomic-embed-text-v1.5",
        "trust_remote_code": True,
        "query_prefix": "search_query: ",
        "document_prefix": "search_document: ",
    },
}

_embedder_cache = {}


def get_embedder(embedding_model=DEFAULT_EMBEDDING_MODEL):
    """
    Load an embedding model on first use and reuse the cached instance.
    """
    if embedding_model not in _embedder_cache:
        cfg = EMBEDDING_MODELS[embedding_model]
        kwargs = {"trust_remote_code": True} if cfg["trust_remote_code"] else {}
        _embedder_cache[embedding_model] = SentenceTransformer(cfg["model_name"], **kwargs)
    return _embedder_cache[embedding_model]


# Preserve the application's original eager-loading behaviour for the default
# model. Other models remain lazy-loaded until explicitly selected.
get_embedder(DEFAULT_EMBEDDING_MODEL)


def _collection_name(user_id, embedding_model=DEFAULT_EMBEDDING_MODEL):
    """
    Generate the ChromaDB collection name for a user and embedding model.

    The user ID is normalised and sanitised before being incorporated into the
    collection name. Non-default embedding models receive a model-specific suffix
    so that vectors from different embedding spaces are never mixed.
    """
    safe = sanitise_for_collection(_normalize_user_id(user_id))
    if embedding_model == DEFAULT_EMBEDDING_MODEL:
        return f"user_{safe}"
    return f"user_{safe}__{embedding_model}"


def get_collection(user_id, embedding_model=DEFAULT_EMBEDDING_MODEL):
    """
    Return the ChromaDB collection associated with a user and embedding model.

    Each user has an isolated collection. Non-default embedding models use a
    separate collection because vectors from different embedding spaces should
    not be mixed.
    """
    return client.get_or_create_collection(_collection_name(user_id, embedding_model))


def reset_collection(user_id, embedding_model=DEFAULT_EMBEDDING_MODEL):
    """
    Delete a user's vector-store collection for the specified embedding model.

    The function is used both to reset experimental indexes and to support the
    application's material-reset functionality.
    """

    try:
        client.delete_collection(_collection_name(user_id, embedding_model))
    except Exception:
        # The collection does not exist, so there is nothing to reset.
        pass


def add_documents(user_id, texts, metadatas=None, embedding_model=DEFAULT_EMBEDDING_MODEL):
    """
    Embed and add document chunks to the user's ChromaDB collection.
    """
    collection = get_collection(user_id, embedding_model)
    embedder = get_embedder(embedding_model)
    cfg = EMBEDDING_MODELS[embedding_model]
    prefixed = [cfg["document_prefix"] + t for t in texts]
    embeddings = embedder.encode(prefixed).tolist()
    ids = [str(uuid.uuid4()) for _ in texts]
    collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)


def list_sources(user_id, embedding_model=DEFAULT_EMBEDDING_MODEL):
    """
    Return the distinct source filenames currently indexed for the user.

    Only source metadata is retrieved because the caller needs document
    identifiers rather than chunk contents.
    """

    # Use the read-only lookup here so querying available sources does not create
    # an empty collection for users who have not indexed any materials.
    try:
        collection = client.get_collection(_collection_name(user_id, embedding_model))
        got = collection.get(include=["metadatas"])
    except Exception:
        return []  # collection doesn't exist yet -- nothing indexed
    metadatas = got.get("metadatas") or []
    seen = []
    for meta in metadatas:
        src = (meta or {}).get("source")
        if src and src not in seen:
            seen.append(src)
    return seen


def retrieve_documents(user_id, query, top_k=3, embedding_model=DEFAULT_EMBEDDING_MODEL):
    """
    Embed the query and return the top-ranked chunks from the user's collection.
    """
    collection = get_collection(user_id, embedding_model)
    embedder = get_embedder(embedding_model)
    cfg = EMBEDDING_MODELS[embedding_model]
    query_emb = embedder.encode([cfg["query_prefix"] + query]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=top_k)
    return results['documents'][0] if results['documents'] else []


def retrieve_with_sources(user_id, query, top_k=5, embedding_model=DEFAULT_EMBEDDING_MODEL,
                          source_filter=None):
    """
    Retrieve ranked chunks together with their source filename and distance.

    If `source_filter` is provided, restrict retrieval to chunks whose `source`
    metadata matches the supplied filename or filenames.
    """
    collection = get_collection(user_id, embedding_model)
    embedder = get_embedder(embedding_model)
    cfg = EMBEDDING_MODELS[embedding_model]
    query_emb = embedder.encode([cfg["query_prefix"] + query]).tolist()

    where = None
    if source_filter:
        names = [source_filter] if isinstance(source_filter, str) else list(source_filter)
        if len(names) == 1:
            where = {"source": names[0]}
        elif len(names) > 1:
            where = {"source": {"$in": names}}

    query_kwargs = {
        "query_embeddings": query_emb,
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    try:
        results = collection.query(**query_kwargs)
    except Exception:
        # Fall back to unfiltered retrieval if the configured ChromaDB 
        # version does not support the requested metadata filter.
        query_kwargs.pop("where", None)
        results = collection.query(**query_kwargs)

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0] or [{}] * len(documents)
    distances = (results.get("distances") or [[]])[0] or [None] * len(documents)

    combined = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        source = (meta or {}).get("source")
        combined.append({"text": doc, "source": source, "distance": dist})
    return combined


# -----------------------------------------------------------------------------
# Chunking strategies (Experiment 1, Table 1: Fixed, Paragraph, Sliding Window)
# -----------------------------------------------------------------------------

def chunk_text_paragraph(text, max_chunk_size=500):
    """
    Split text into approximately fixed-size chunks while preserving paragraph
    boundaries where possible.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Split oversized paragraphs before packing so that individual chunks remain
    # within the configured character limit and are not dominated by a single
    # unusually long paragraph.
    normalised = []
    for para in paragraphs:
        if len(para) <= max_chunk_size:
            normalised.append(para)
        else:
            normalised.extend(chunk_text_fixed(para, max_chunk_size=max_chunk_size))

    chunks = []
    current = ""
    for para in normalised:
        if len(current) + len(para) < max_chunk_size:
            current += para + "\n\n"
        else:
            if current:
                chunks.append(current.strip())
            current = para + "\n\n"
    if current:
        chunks.append(current.strip())
    return chunks


def chunk_text_fixed(text, max_chunk_size=500):
    """
    Split text into approximately fixed-size chunks on whitespace boundaries.

    Paragraph structure is intentionally ignored. This strategy serves as the
    fixed-size baseline for Experiment 1.
    """
    words = text.split()
    chunks = []
    current, current_len = [], 0
    for word in words:
        if current and current_len + len(word) + 1 > max_chunk_size:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(word)
        current_len += len(word) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_text_sliding_window(text, max_chunk_size=500, overlap=100):
    """
    Split text into overlapping fixed-size chunks.

    Approximately `overlap` characters from the preceding chunk are retained to
    provide additional context across chunk boundaries.
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        current, current_len, j = [], 0, i
        while j < len(words) and (not current or current_len + len(words[j]) + 1 <= max_chunk_size):
            current.append(words[j])
            current_len += len(words[j]) + 1
            j += 1
        chunk_str = " ".join(current)

        # Avoid adding a final chunk that contributes no 
        # new text beyond the previous chunk.
        if chunks and chunk_str in chunks[-1]:
            break
        chunks.append(chunk_str)

        # Advance the window while retaining approximately 
        # `overlap` characters from the current chunk.
        overlap_len, keep_from = 0, len(current)
        for w in reversed(current):
            if overlap_len + len(w) + 1 > overlap:
                break
            keep_from -= 1
            overlap_len += len(w) + 1
        # always make forward progress
        step = max(keep_from, 1)  
        i += step
    return chunks


CHUNKING_STRATEGIES = {
    "paragraph": chunk_text_paragraph,
    "fixed": chunk_text_fixed,
    "sliding_window": chunk_text_sliding_window,
}


def chunk_text(text, max_chunk_size=500, strategy="sliding_window", overlap=100):
    """
    Chunk text using the selected strategy.

    The default strategy is set to sliding_window
    """
    if strategy == "sliding_window":
        return chunk_text_sliding_window(text, max_chunk_size=max_chunk_size, overlap=overlap)
    fn = CHUNKING_STRATEGIES.get(strategy, chunk_text_paragraph)
    return fn(text, max_chunk_size=max_chunk_size)