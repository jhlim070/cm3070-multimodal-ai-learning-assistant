import chromadb
from sentence_transformers import SentenceTransformer
import uuid

embedder = SentenceTransformer('BAAI/bge-small-en')   # bge-small, free & local
client = chromadb.PersistentClient(path="./chroma_db")

def get_collection(user_id):
    """Each user gets their own collection (or use metadata filtering)."""
    return client.get_or_create_collection(f"user_{user_id}")

def add_documents(user_id, texts, metadatas=None):
    """
    Embeds each chunk using the embedder.
    Adds them to a ChromaDB collection named user_{user_id} – each user has their own collection, which 
    makes multi‑user support trivial.
    Unique ID for each chunk
    """
    collection = get_collection(user_id)
    embeddings = embedder.encode(texts).tolist()
    ids = [str(uuid.uuid4()) for _ in texts]
    collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)

def retrieve_documents(user_id, query, top_k=3):
    """
    Embeds the user’s question.
    Queries the user’s collection for the top_k most similar chunks.
    Returns the raw text of those chunks.
    The non‑parametric memory of the RAG system, grounds the LLM in the student’s actual 
    course material.


    """
    collection = get_collection(user_id)
    query_emb = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=top_k)
    return results['documents'][0] if results['documents'] else []

# Chuncking Method ---------------------------------------------------------------------------
def chunk_text(text, max_chunk_size=500):
    """
    Simple paragraph/section chunking: 
    Splits the extracted text into chunks of roughly 500 characters, but respects paragraph boundaries
    (double‑newline). 
    Section‑based chunking – it keeps related sentences together, which is vital for educational content 
    where a concept might span several lines.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) < max_chunk_size:
            current += para + "\n\n"
        else:
            if current:
                chunks.append(current.strip())
            current = para + "\n\n"
    if current:
        chunks.append(current.strip())
    return chunks