import ollama   # no API costs and no internet dependency
from rag import retrieve_documents
from db import get_profile, save_profile, log_interaction, get_weak_topics

# Create learning style
def onboard_user(user_id, learning_style, difficulty, topics):
    """
    A thin wrapper that calls save_profile() from db.py. It’s separated because it’s a “command” rather than a query.
    """
    save_profile(user_id, learning_style, difficulty, topics)
    
#  Retrieval‑Augmented Generation (RAG), fomulating of answers based on materials provided
def generate_answer(user_id, query, model="llama3.2"):
    # 1. Retrieve relevant chunks from user's uploaded materials
    context_chunks = retrieve_documents(user_id, query, top_k=3)
    context = "\n\n".join(context_chunks)

    # 2. Load user profile
    profile = get_profile(user_id)
    learning_style = profile.get("learning_style", "clear and detailed") if profile else "clear and detailed"
    difficulty = profile.get("difficulty", "intermediate") if profile else "intermediate"

    # 3. System prompt with personalisation
    system_msg = f"""You are a helpful, adaptive learning assistant.
The student prefers {learning_style} explanations at a {difficulty} level.
Answer the question using ONLY the provided context. If the context does not contain the answer, say so.
Keep your answer educational and supportive."""

    # 4. Call Ollama
    response = ollama.chat(model=model, messages=[
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}], 
        options={'timeout': 60})
    answer = response['message']['content']

    # 5. Log interaction (topic extraction simplified)
    topic = query.split()[0]  # naive – you can improve with keyword extraction later
    log_interaction(user_id, query, answer, topic)
    return answer

# Adaptive recommendations ---------------------------------------------------------------------------
def generate_recommendations(user_id):
    """
    Uses weak topics to suggest revision.
    Calls get_weak_topics() from db.py.
    If weak topics exist, it formats them, queries the LLM then returns the suggestion
    
    """
    weak_topics = get_weak_topics(user_id)
    if not weak_topics:
        return "No weak topics identified yet. Keep studying!"

    # Create a prompt for the LLM
    topic_list = "\n".join([f"- {t[0]} (queries: {t[1]}, avg rating: {t[2]:.1f})" for t in weak_topics])
    prompt = f"""Based on the user's interaction history, the following topics seem challenging:
{topic_list}

Suggest 3 specific study activities or concepts the user should review to strengthen their understanding.
Be encouraging and practical."""
    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

# Multimodal handling ---------------------------------------------------------------------------
def describe_image(image_path):
    # a textual description of an image from LLaVA
    res = ollama.generate(model="llava:7b",
                          prompt="Describe the content of this image in detail, focusing on any text, diagrams, or key concepts.",
                          images=[image_path])
    return res['response']

def process_image_and_text(user_id, text, image_path):
    """
    If an image is present, it appends an image description to the query
    If no text is given, it defaults to “Explain this image”
    Then calls generate_answer
    giving the ChatInterface a single entry point
    """
    if image_path:
        desc = describe_image(image_path)
        text = f"[Image description: {desc}]\nUser query: {text}"
    # If no query but just an image, treat as "explain this image"
    if not text.strip():
        text = "Explain the content and any important concepts shown in this image."
    return generate_answer(user_id, text)
