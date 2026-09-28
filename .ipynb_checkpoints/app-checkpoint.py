import gradio as gr
from engine import process_image_and_text, generate_recommendations, onboard_user
from db import init_db
# OCR + PDF extraction
from material_processor import process_file
# vector storage, chunking
from rag import add_documents, chunk_text        

# Initialise the SQLite database (creates tables if they don't exist)
init_db()

# 1. Document upload & indexing handler ---------------------------------------------------------
def upload_and_index(file, user_id):
    """Extract text from uploaded file, chunk it, and store in ChromaDB."""
    if file is None:
        return "!!! No file selected."
    try:
        # handles PDF and Image files using OCR (from material_processor)
        text = process_file(file.name)   # file.name gives the temp file path
        # splits huge text into sections that make sense to the LLM
        chunks = chunk_text(text)
        if not chunks:
            return "!!! No text could be extracted."
        # embeds each chunk and stores them in vector database tied to the user
        add_documents(user_id, chunks)
        return f"✅ Indexed {len(chunks)} chunks."
    except Exception as e:
        return f"! Error: {str(e)}"

# 2. Chat handler -------------------------------------------------------------------------------
def respond(message, history, user_id, image):
    """
    called each time the user sends message in the ChatInterface
    - if an image is uploaded, it will be described by LLaVA and combined with the query.
    - the answer is generated using RAG, user profile, and interaction memory.
    """
    # On the very first message, we could run a quick onboarding, but here we keep it simple.
    # Convert image to file path (Gradio gives a temp file path or None)
    print("--- respond called ---")
    img_path = image if image else None
    print("Calling process_image_and_text...")
    # process the material uploaded
    answer = process_image_and_text(user_id, message, img_path)
    print("Got answer:", answer[:100] if answer else "None")
    return answer

# 3. Recommendation handler ---------------------------------------------------------------------
def get_recommendations_btn(user_id):
    # Fetch study recommendations based on weak topics
    return generate_recommendations(user_id)

# 4. Build the UI using Gradio -------------------------------------------------------------------
with gr.Blocks(title="Multimodal AI Learning Assistant") as demo:
    gr.Markdown(
        """
        # Multimodal AI Learning Assistant
        Upload your study materials (PDFs, lecture slides, images), then ask questions.
        The assistant remembers what was learned and suggests weak topics for revision.
        """
    )

    # User ID and Profile Section 
    with gr.Row():
        user_id = gr.Textbox(label="Your User ID", value="demo", scale=1)
        with gr.Column(scale=3):
            gr.Markdown("")  # spacer

    with gr.Row():
        with gr.Column(scale=1):
            style = gr.Dropdown(
                ["simple analogies", "detailed technical", "visual examples", "step-by-step"],
                label="Learning Style",
                value="step-by-step"
            )
            difficulty = gr.Radio(
                ["beginner", "intermediate", "advanced"],
                label="Difficulty",
                value="intermediate"
            )
            topics = gr.Textbox(
                label="Topics of interest (comma separated)",
                placeholder="e.g., machine learning, databases, linear algebra"
            )
            onboard_btn = gr.Button("Save Profile")
            onboard_status = gr.Textbox(label="", visible=False)  # hidden feedback
        with gr.Column(scale=2):
            gr.Markdown("### Upload Study Material")
            file_upload = gr.File(label="Select a PDF or image file", file_types=[".pdf", ".png", ".jpg", ".jpeg"])
            upload_btn = gr.Button("Index Material")
            upload_status = gr.Textbox(label="Upload Status", interactive=False)

    # Chat Interface (main area) 
    chatbot = gr.ChatInterface(
        fn=respond,
        additional_inputs=[user_id, gr.Image(type="filepath", label="Upload an image (optional)")],
        title="Ask a question about your materials",
        description="Type your question or upload an image of a diagram/problem.",
    )

    # Recommendations 
    with gr.Row():
        recommend_btn = gr.Button("Get Study Recommendations")
        recommendations_out = gr.Textbox(label="Your Personalised Recommendations", lines=6, interactive=False)

    # link buttons to trigger the respective functions
    onboard_btn.click(
        fn=onboard_user,
        inputs=[user_id, style, difficulty, topics],
        outputs=onboard_status
    )
    upload_btn.click(
        fn=upload_and_index,
        inputs=[file_upload, user_id],
        outputs=upload_status
    )
    recommend_btn.click(
        fn=get_recommendations_btn,
        inputs=user_id,
        outputs=recommendations_out
    )


# 5. Launch the app ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch()