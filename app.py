"""
Multimodal AI Learning Assistant using Gradio Frontend

Provides the web interface, user profile management, document indexing 
workflow, interactive Q&A chat, and comprehension-based rating system 
for personalized study recommendations.
"""

import logging
import time
from functools import partial

import gradio as gr

from engine import (process_image_and_text_detailed, generate_recommendations,
                    onboard_user, DEFAULT_MODEL)
from db import (init_db, get_profile, get_last_rateable_interaction_id, set_feedback,
                is_file_indexed, record_indexed_file, clear_indexed_files,
                list_indexed_files, clear_pending_clarification)
# OCR + PDF extraction
from material_processor import process_file
# vector storage, chunking
from rag import add_documents, chunk_text, reset_collection
import validation

logger = logging.getLogger(__name__)

# Initialise the SQLite database (creates any missing tables)
init_db()

MAX_FILES_PER_UPLOAD = 5


# Document upload & indexing handler ---------------------------------------------------------
def upload_and_index(files, user_id):
    """
    Validate, de-duplicate, extract, chunk, and index uploaded files.

    Each file is processed independently, so a failure in one file does not
    prevent the remaining files from being processed. 
    
    Validation and duplicate checks happen before parsing: 
      1. Upload validation checks the extension, file size, magic bytes, and,
      for .docx/.pptx files, the ZIP structure and expected Office format. 
      2. In-batch duplicate checks prevent the same file or filename from 
      being selected more than once. 
      3. The indexed-file registry is checked to prevent files already indexed 
      for this user from being indexed again.
    
    Duplicates are rejected rather than silently ignored because duplicate chunks 
    can occupy retrieval slots and reduce the amount of distinct material returned 
    by the vector search.
    """
    ok_user, user_id, msg = validation.validate_user_id(user_id)
    if not ok_user:
        return f"!!! {msg}"

    if not files:
        return "!!! No file selected."

    file_list = files if isinstance(files, list) else [files]

    overflow_note = ""
    if len(file_list) > MAX_FILES_PER_UPLOAD:
        overflow_note = (f"\n(Only the first {MAX_FILES_PER_UPLOAD} of the "
                         f"{len(file_list)} selected files were processed -- "
                         f"upload the rest in a second batch.)")
        file_list = file_list[:MAX_FILES_PER_UPLOAD]

    total_chunks = 0
    indexed_count = 0
    per_file_results = []
    seen_hashes = set()
    seen_names = set()

    for f in file_list:
        path = getattr(f, "name", f)

        valid, name, why = validation.validate_upload(path)
        if not valid:
            per_file_results.append(f"❌ {name}: rejected -- {why}")
            continue

        try:
            digest = validation.sha256_file(path)
        except OSError as e:
            per_file_results.append(f"❌ {name}: could not be read ({e})")
            continue

        if digest in seen_hashes:
            per_file_results.append(f"⏭️ {name}: skipped -- selected twice in this batch")
            continue
        if name in seen_names:
            per_file_results.append(f"⏭️ {name}: skipped -- another file in this batch "
                                    f"has the same name")
            continue
        seen_hashes.add(digest)
        seen_names.add(name)

        already = is_file_indexed(user_id, sha256=digest, filename=name)
        if already:
            if already["sha256"] == digest:
                per_file_results.append(
                    f"⏭️ {name}: skipped -- already indexed on {already['indexed_at']} "
                    f"as \"{already['filename']}\" ({already['chunks']} chunks)")
            else:
                per_file_results.append(
                    f"❌ {name}: rejected -- a different file is already indexed under "
                    f"this name. Rename it, or use \"Reset my materials\" and re-index.")
            continue

        try:
            text = process_file(path)
            chunks = chunk_text(text, strategy="sliding_window")
            if not chunks:
                per_file_results.append(f"!!! {name}: no text could be extracted")
                continue
            add_documents(user_id, chunks, metadatas=[{"source": name} for _ in chunks])
            record_indexed_file(user_id, name, digest, len(chunks))
            total_chunks += len(chunks)
            indexed_count += 1
            per_file_results.append(f"✅ {name}: {len(chunks)} chunks")
        except Exception as e:
            per_file_results.append(f"! {name}: Error - {e}")

    summary = (f"Indexed {total_chunks} chunks from {indexed_count} of "
               f"{len(file_list)} file(s):{overflow_note}\n")
    summary += "\n".join(per_file_results)
    return summary


def reset_materials(user_id):
    """
    Clear both the user's vector collection and indexed-file registry. 
    
    These two stores represent the same indexing state, so they must 
    be reset together. Resetting only one could either block valid 
    re-uploads or create duplicate indexed content. 

    This is also the supported way to rebuild the index after changing 
    text extraction or chunking behaviour.
    """
    ok_user, user_id, msg = validation.validate_user_id(user_id)
    if not ok_user:
        return f"!!! {msg}"
    removed = clear_indexed_files(user_id)
    reset_collection(user_id)
    return (f"🧹 Cleared {removed} indexed file record(s) and emptied the index for "
            f"\"{user_id}\". Upload materials again to re-index them.")


def show_indexed_files(user_id):
    ok_user, user_id, msg = validation.validate_user_id(user_id)
    if not ok_user:
        return f"!!! {msg}"
    rows = list_indexed_files(user_id)
    if not rows:
        return "Nothing indexed yet."
    lines = [f"• {r['filename']} -- {r['chunks']} chunks (indexed {r['indexed_at']})"
             for r in rows]
    return "Currently indexed:\n" + "\n".join(lines)


# Chat handler -------------------------------------------------------------------------------
def respond(message, history, user_id, image, awaiting_rating):
    """
    Handle a message submitted through the ChatInterface.

    The rating gate is controlled by the structured `rateable` flag 
    returned by the engine rather than by inspecting the generated 
    answer for specific words or formatting.

    Only completed, rateable answers trigger the rating gate. 
    Clarification requests, "not in materials" responses, errors, 
    rejected input, and continuations are not currently rateable
    because they do not represent a completed answer for the 
    learner to evaluate.
    
    Keeping this decision separate from the answer text also
    prevents non-answer interactions from being recorded as 
    weak-topic signals.
    """

    ok_user, user_id, user_msg = validation.validate_user_id(user_id)
    if not ok_user:
        return (f"⚠️ {user_msg}", gr.update(), gr.update(visible=False), False)

    ok_q, message, q_msg = validation.validate_question(message)
    if not ok_q:
        return (f"⚠️ {q_msg}", gr.update(), gr.update(visible=bool(awaiting_rating)),
                bool(awaiting_rating))

    ok_img, img_msg = validation.validate_chat_image(image)
    if not ok_img:
        return (f"⚠️ {img_msg}", gr.update(), gr.update(visible=bool(awaiting_rating)),
                bool(awaiting_rating))

    if awaiting_rating:
        return (
            "⏸️ Please rate your understanding of the previous answer (1-5) below before "
            "asking another question. (Or use Clear chat to start over.)",
            gr.update(),                 # leave the model/time status line unchanged
            gr.update(visible=True),     # keep the rating panel shown
            True,                        # awaiting a rating
        )

    start_time = time.time()
    result = process_image_and_text_detailed(user_id, message, image if image else None)
    elapsed = time.time() - start_time

    status_text = (f"**Model:** {result.get('model', DEFAULT_MODEL)} &nbsp;|&nbsp; "
                   f"**Last response time:** {elapsed:.2f}s")
    sources = result.get("sources") or []
    if sources and result.get("kind") in ("answer", "continuation"):
        status_text += " &nbsp;|&nbsp; **From:** " + ", ".join(sources)

    needs_rating = bool(result.get("rateable"))
    return (
        result["answer"],
        status_text,
        gr.update(visible=needs_rating),
        needs_rating,
    )


# Recommendation handler ---------------------------------------------------------------------
def get_recommendations_btn(user_id):
    ok_user, user_id, msg = validation.validate_user_id(user_id)
    if not ok_user:
        return f"!!! {msg}"
    return generate_recommendations(user_id)


def _clear_pending_clarification(user_id):
    """
    Clears any pending document-clarification state for the user.

    A clarification request belongs to the conversation in which 
    it was created. Clearing it when that conversation ends prevents 
    a message in a new session from being interpreted as an answer 
    to an old clarification.

    This cleanup is best-effort: failure here should not prevent 
    the UI action that triggered it from completing.
    """
    try:
        ok, uid, _ = validation.validate_user_id(user_id)
        if ok:
            clear_pending_clarification(uid)
    except Exception:
        logger.exception("Failed to clear pending clarification")


# Login / landing gate handlers --------------------------------------------------------------
def handle_login(user_id):
    """
    Handle the Continue button on the landing screen.

    This is an identification gate, not authentication: no password is checked. 
    The user ID is validated because it is used as a SQLite key 
    and as part of the ChromaDB collection naming.

    Existing users have their saved profile loaded and locked. 
    New users start with an editable profile.
    """
    ok, user_id, message = validation.validate_user_id(user_id)
    if not ok:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(), gr.update(), gr.update(),  # leave style/difficulty/topics unchanged
            f"!!! {message}",
            "",
            gr.update(),                # profile button unchanged
            False,                      # not locked
            "",
        )

    _clear_pending_clarification(user_id)   # a new session starts with an empty chat

    profile = get_profile(user_id)
    if profile:
        style_val = profile["learning_style"]
        difficulty_val = profile["difficulty"]
        topics_val = ", ".join(profile["topics"])
        locked = True
        button_update = gr.update(value="✏️ Edit Profile", variant="secondary")
        profile_note = "🔒 Profile loaded and locked. Click Edit Profile to change it."
        message = f"Welcome back, {user_id}! Your saved preferences have been loaded."
    else:
        style_val, difficulty_val, topics_val = "step-by-step", "intermediate", ""
        locked = False
        button_update = gr.update(value="💾 Save Profile", variant="primary")
        profile_note = "Set your preferences, then click Save Profile."
        message = f"Welcome, {user_id}! This is a new profile -- set your preferences and save it."

    return (
        gr.update(visible=False),   
        gr.update(visible=True),   
        gr.update(value=style_val, interactive=not locked),
        gr.update(value=difficulty_val, interactive=not locked),
        gr.update(value=topics_val, interactive=not locked),
        message,
        user_id,
        button_update,
        locked,
        profile_note,
    )

# Switch account -----------------------------------------------------------------------
def go_back_to_login(user_id=None):
    """
    Return to the landing screen when the user switches accounts.
    
    The per-session rating gate and any pending clarification are cleared so
    the next user does not inherit state from the previous session.
    """
    _clear_pending_clarification(user_id)
    return (gr.update(visible=True), gr.update(visible=False), "",
            gr.update(visible=False), False, "")


MODEL_STATUS_IDLE = f"**Model:** {DEFAULT_MODEL} &nbsp;|&nbsp; **Last response time:** -"


def blank_session_view():
    """
    Return the values used to clear all per-user UI state.

    A session switch must clear more than the visible chat. Gradio's
    ChatInterface maintains its own copies of the conversation, so those
    internal history values must also be reset or the previous conversation
    can reappear when the next message is submitted.
    
    The other fields are cleared to prevent the next user from seeing the
    previous user's recommendations, upload status, attached image, or model
    status. Although this data is not stored under the new user's ID, leaving
    it visible would expose information across sessions.
    """
    return {
        "chat": [], "chat_state": [], "chat_value": [], "chat_text": "", "chat_image": None,
        "recommendations": "", "upload_status": "", "file_upload": None,
        "model_status": MODEL_STATUS_IDLE, "login_status": "",
    }


# Profile save / lock / edit ------------------------------------------------------------
def toggle_profile(user_id, learning_style, difficulty, topics, locked):
    """
    Handle both profile editing and profile saving.
    When the profile is locked, the button switches to edit mode. 
    When the profile is unlocked, the current settings are validated, 
    saved, and locked again.

    Locking makes profile changes explicit. The profile is used when building
    the system prompt, so changing it mid-session can change the conditions
    under which personalised responses are generated. Keeping the profile
    fixed during a session also supports reproducible experiment runs.
    """
    if locked:
        return (
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(value="💾 Save Profile", variant="primary"),
            False,
            "✏️ Editing -- change what you need, then click Save Profile.",
        )

    ok_user, user_id, msg = validation.validate_user_id(user_id)
    if not ok_user:
        return (gr.update(), gr.update(), gr.update(), gr.update(), False, f"!!! {msg}")

    ok_topics, clean_topics, topic_msg = validation.validate_topics(topics)
    if not ok_topics:
        return (gr.update(), gr.update(), gr.update(), gr.update(), False, f"!!! {topic_msg}")

    onboard_user(user_id, learning_style, difficulty, clean_topics)

    return (
        gr.update(value=learning_style, interactive=False),
        gr.update(value=difficulty, interactive=False),
        gr.update(value=clean_topics, interactive=False),
        gr.update(value="✏️ Edit Profile", variant="secondary"),
        True,
        "🔒 Profile saved and locked.",
    )


# Feedback handlers ----------------------------------------------------------------------
RATING_LABELS = {
    1: "1  😕 Confused",
    2: "2  🤔 Struggling",
    3: "3  😐 Knowing, not fully",
    4: "4  🙂 Mostly understood",
    5: "5  🤩 Fully understood",
}


def submit_feedback(user_id, rating):
    ok_user, user_id, msg = validation.validate_user_id(user_id)
    if not ok_user:
        return f"!!! {msg}"
    if rating not in RATING_LABELS:
        return "!!! Invalid rating."
    
    # Rate the most recent completed answer, not simply the most recent
    # interaction. Clarification or "not in your materials" interactions 
    # may have been logged after the answer being rated.
    interaction_id = get_last_rateable_interaction_id(user_id)
    if interaction_id is None:
        return "!!! No answer to rate yet -- ask a question first."
    set_feedback(interaction_id, rating)
    return f"✅ Logged: {RATING_LABELS[rating].split('  ', 1)[1]}"


def rate_response(user_id, rating):
    """
    Record the rating for the most recent rateable interaction, 
    hide the rating panel, and release the question gate.
    """
    status = submit_feedback(user_id, rating)
    return status, gr.update(visible=False), False


def release_gate(user_id=None):
    """
    Release the rating gate without recording a rating.

    This is used when the conversation is cleared. Treating the abandoned
    conversation as "not rated" avoids inventing a rating and prevents the
    gate from becoming stuck when there is no longer an answer to rate.

    Any pending document clarification is also cleared because it belongs to
    the conversation that was just abandoned.
    """
    _clear_pending_clarification(user_id)
    return gr.update(visible=False), False, ""


def clear_chat(user_id=None):
    """ 
    The explicit Clear chat button: empties the chat AND releases the gate.
    """
    panel, awaiting, status = release_gate(user_id)
    return [], panel, awaiting, status


RATING_PANEL_CSS = """
.rating-prompt {
    background: linear-gradient(90deg, #fff176, #ffb74d);
    border: 3px solid #f57c00;
    border-radius: 10px;
    padding: 14px;
}
.rating-prompt button {
    font-weight: 600;
}
"""

# Build the UI using Gradio -------------------------------------------------------------------
with gr.Blocks(title="Multimodal AI Learning Assistant") as demo:

    profile_locked_state = gr.State(False)

    # Screen 1: Login / landing gate 
    with gr.Group(visible=True) as login_screen:
        gr.Markdown(
            """
            # Multimodal AI Learning Assistant
            Enter a User ID to continue. An existing ID loads your saved
            preferences automatically; a new ID starts a fresh profile.
            """
        )
        login_user_id = gr.Textbox(label="Your User ID", placeholder="e.g. demo",
                                   max_lines=1)
        login_btn = gr.Button("Continue", variant="primary")
        login_status = gr.Markdown("")

    # Screen 2: Main application
    with gr.Group(visible=False) as main_screen:
        with gr.Row():
            current_user_display = gr.Textbox(label="Logged in as", interactive=False, scale=3)
            logout_btn = gr.Button("Switch User", scale=1)

        # How to use accordion 
        with gr.Accordion("ℹ️ How to use this assistant (click to expand)", open=False):
            gr.Markdown(
                """
                **Quick-start guide - 4 steps:**

                **Step 1 - Set your profile** (left panel)
                Choose how you want answers explained (*Learning Style*), your background
                knowledge level (*Difficulty*), and what subjects you're studying (*Topics*).
                Click **Save Profile** to lock it in. You can edit it later at any time.

                **Step 2 - Upload and index your study materials** (right panel)
                Select up to 5 PDF, Word (.docx), PowerPoint (.pptx), or image files and
                click **Index Material**. The assistant reads your documents and builds a
                searchable index - it cannot answer questions until at least one file is indexed.
                Use **What's indexed?** to check what's loaded, and **Reset my materials** to
                start over with a fresh index.

                **Step 3 - Ask questions in the chat**
                Type any question about your uploaded materials. You can also attach an image
                (diagram, photo of a problem) using the image upload button in the chat.
                Type **"continue"** to ask the assistant to expand on its last answer.
                If your materials span multiple documents, the assistant may ask you to
                clarify which document you mean - just reply with the number or file name shown.

                **Step 4 - Rate your understanding after each answer**
                After every answer a coloured rating panel appears.
                **You must rate before asking the next question.**
                Ratings feed the weak-topic detector:
                - Ratings 1-2 (Confused / Struggling) count as low-confidence signals.
                - Once a document accumulates **3 or more** low-confidence ratings it is
                  flagged as a weak area.
                - Click **Get Study Recommendations** to see personalised revision suggestions
                  based on those flags. You need at least 3 low ratings on the same document
                  before recommendations appear.
                """
            )

        gr.Markdown(
            """
            Upload your study materials (PDFs, lecture slides, images), then ask questions.
            The assistant remembers what was learned and suggests weak topics for revision.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                style = gr.Dropdown(
                    ["simple analogies", "detailed technical", "visual examples", "step-by-step"],
                    label="Learning Style",
                    value="step-by-step",
                    info=(
                        "Controls how answers are phrased. "
                        "'Simple analogies' uses everyday comparisons; "
                        "'detailed technical' uses precise vocabulary; "
                        "'visual examples' structures answers as tables or diagrams; "
                        "'step-by-step' breaks answers into numbered steps."
                    ),
                )
                difficulty = gr.Radio(
                    ["beginner", "intermediate", "advanced"],
                    label="Difficulty",
                    value="intermediate",
                    info=(
                        "Sets assumed background knowledge. "
                        "'Beginner' defines every term; "
                        "'intermediate' assumes general familiarity; "
                        "'advanced' skips elementary definitions."
                    ),
                )
                topics = gr.Textbox(
                    label="Topics of interest (comma separated)",
                    placeholder="e.g., machine learning, databases, linear algebra",
                    info=(
                        "Enter the subjects you are studying, separated by commas. "
                        "These are used to improve topic labelling in your interaction history."
                    ),
                )
                onboard_btn = gr.Button("💾 Save Profile", variant="primary")
                profile_note = gr.Markdown("Set your preferences, then click Save Profile.")
            with gr.Column(scale=2):
                gr.Markdown("### Upload Study Material")
                file_upload = gr.File(
                    label="Select up to 5 PDF, Word, PowerPoint, or image files",
                    file_types=[".pdf", ".docx", ".pptx", ".png", ".jpg", ".jpeg"],
                    file_count="multiple",
                )
                with gr.Row():
                    upload_btn = gr.Button(
                        "Index Material", variant="primary",
                    )
                    list_files_btn = gr.Button("What's indexed?")
                    reset_btn = gr.Button("🧹 Reset my materials", variant="stop")
                upload_status = gr.Textbox(
                    label="Upload Status",
                    interactive=False,
                    lines=6,
                    info=(
                        "Shows the result of the last upload or index operation. "
                        "Each file is reported individually - a ✅ means it was indexed "
                        "successfully; ⏭️ means it was skipped (already indexed); "
                        "❌ means it was rejected."
                    ),
                )

        model_status_display = gr.Markdown(
            MODEL_STATUS_IDLE,
            render=False,
        )
        awaiting_rating_state = gr.State(False)

        rating_group = gr.Group(visible=False, elem_classes=["rating-prompt"], render=False)
        with rating_group:
            gr.Markdown(
                "### How well did you understand that answer?\n"
                "Rate 1-2 if you found it confusing - these feed the weak-topic detector. "
                "Rate 3-5 if you understood it. **You must rate before asking the next question.**"
            )
            with gr.Row():
                rating_buttons = [gr.Button(RATING_LABELS[v]) for v in (1, 2, 3, 4, 5)]
        feedback_status = gr.Markdown("", render=False)
        clear_chat_btn = gr.Button("Clear chat", render=False)

        chatbot = gr.ChatInterface(
            fn=respond,
            chatbot=gr.Chatbot(height=480),
            additional_inputs=[current_user_display, 
                               gr.Image(type="filepath", label="Upload an image (optional)",
                                        ),
                               awaiting_rating_state],
            additional_outputs=[model_status_display, rating_group, awaiting_rating_state],
            title="Ask a question about your materials",
            autofocus=False,
            description=(
                "Type your question or upload an image of a diagram/problem. "
                "Type \"continue\" to expand on the last answer. "
                "If the assistant asks which document you mean, reply with the number shown."
            ),
        )
        model_status_display.render()
        rating_group.render()
        feedback_status.render()
        clear_chat_btn.render()

        # Recommendations
        with gr.Row():
            recommend_btn = gr.Button(
                "Get Study Recommendations",
            )
            recommendations_out = gr.Textbox(
                label="Your Personalised Recommendations",
                lines=6,
                interactive=False,
                info=(
                    "Appears once a document has accumulated 3 or more low-confidence ratings "
                    "(scores of 1 or 2). Ask more questions and rate them honestly - "
                    "the more you rate, the more accurate the recommendations."
                ),
            )

    # Wiring
    login_btn.click(
        fn=handle_login,
        inputs=login_user_id,
        outputs=[login_screen, main_screen, style, difficulty, topics, login_status,
                 current_user_display, onboard_btn, profile_locked_state, profile_note],
    )
    """
    Reset per-user display state when switching accounts.
    
    ChatInterface keeps internal copies of the conversation in addition to  
    the visible chat, so resetting only the visible component is insufficient: 
    the old messages can reappear when the next message is submitted
    
    These targets are accessed with getattr because they are not documented
    public Gradio APIs. Missing attributes are ignored so the reset remains
    compatible with different Gradio versions.
    """
    _chat_image = next((c for c in getattr(chatbot, "additional_inputs", []) or []
                        if isinstance(c, gr.Image)), None)
    _chat_history_targets = {
        "chat": getattr(chatbot, "chatbot", None),
        "chat_state": getattr(chatbot, "chatbot_state", None),
        "chat_value": getattr(chatbot, "chatbot_value", None),
    }
    SESSION_RESET_TARGETS = {
        **_chat_history_targets,
        "chat_text": getattr(chatbot, "textbox", None),
        "chat_image": _chat_image,
        "recommendations": recommendations_out,
        "upload_status": upload_status,
        "file_upload": file_upload,
        "model_status": model_status_display,
        "login_status": login_status,
    }
    SESSION_RESET_TARGETS = {k: v for k, v in SESSION_RESET_TARGETS.items() if v is not None}
    CHAT_HISTORY_TARGETS = {k: v for k, v in _chat_history_targets.items() if v is not None}

    def reset_session_view():
        values = blank_session_view()
        return {comp: values[name] for name, comp in SESSION_RESET_TARGETS.items()}
    
    def reset_chat_history():
        """
        Reset only the ChatInterface history components.
        """
        values = blank_session_view()
        return {comp: values[name] for name, comp in CHAT_HISTORY_TARGETS.items()}

    logout_btn.click(
        fn=go_back_to_login,
        inputs=current_user_display,
        outputs=[login_screen, main_screen, login_user_id,
                 rating_group, awaiting_rating_state, feedback_status],
    ).then(
        fn=reset_session_view,
        inputs=None,
        outputs=list(SESSION_RESET_TARGETS.values()),
    )
    onboard_btn.click(
        fn=toggle_profile,
        inputs=[current_user_display, style, difficulty, topics, profile_locked_state],
        outputs=[style, difficulty, topics, onboard_btn, profile_locked_state, profile_note],
    )
    upload_btn.click(
        fn=upload_and_index,
        inputs=[file_upload, current_user_display],
        outputs=upload_status,
    )
    list_files_btn.click(
        fn=show_indexed_files,
        inputs=current_user_display,
        outputs=upload_status,
    )
    reset_btn.click(
        fn=reset_materials,
        inputs=current_user_display,
        outputs=upload_status,
    )
    for rating_value, btn in zip((1, 2, 3, 4, 5), rating_buttons):
        btn.click(
            fn=partial(rate_response, rating=rating_value),
            inputs=current_user_display,
            outputs=[feedback_status, rating_group, awaiting_rating_state],
        )
    recommend_btn.click(
        fn=get_recommendations_btn,
        inputs=current_user_display,
        outputs=recommendations_out,
    )

    """
    The explicit Clear chat button resets both the visible conversation and
    the rating gate.

    The built-in ChatInterface trash/clear control does not call respond(),
    so it must also release the rating gate. The `.clear()` event is used
    when supported by the installed Gradio version; otherwise the explicit
    Clear chat button remains the fallback.
    """
    _chat_display = getattr(chatbot, "chatbot", None)
    if _chat_display is not None:
        clear_chat_btn.click(
            fn=clear_chat,
            inputs=current_user_display,
            outputs=[_chat_display, rating_group, awaiting_rating_state, feedback_status],
        ).then(
            # clear_chat() only empties the VISIBLE chat; ChatInterface's own copy of
            # the history must go too, or the old messages return on the next question.
            fn=reset_chat_history,
            inputs=None,
            outputs=list(CHAT_HISTORY_TARGETS.values()),
        )
        try:
            _chat_display.clear(
                fn=release_gate,
                inputs=current_user_display,
                outputs=[rating_group, awaiting_rating_state, feedback_status],
            )
        except Exception:
            logger.exception("Failed to clear pending clarification")
    else:
        clear_chat_btn.click(
            fn=release_gate,
            inputs=current_user_display,
            outputs=[rating_group, awaiting_rating_state, feedback_status],
        )


"""
Launch the application using only the options supported by the installed
Gradio version.

`build_launch_kwargs()` inspects the `demo.launch()` signature and removes
unsupported options before calling it. This allows the application to retain
compatible features without failing because of a version-specific argument.

Supported options include:
  - `css`: applies the rating-panel stylesheet.
  - `max_file_size`: limits uploads before they reach application-level validation.
  - `show_api`: disables Gradio's generated API page when supported.

Older Gradio versions may store the CSS on the `Blocks` object instead, 
so that case is handled separately below.
"""
WANTED_LAUNCH_OPTIONS = {"css": RATING_PANEL_CSS, "max_file_size": "50mb", "show_api": False}


def build_launch_kwargs(launch_fn, wanted=None):
    """
    Return only the requested launch options accepted by `launch_fn`. 
    Signature inspection keeps the launch code compatible with 
    Gradio versions that expose different sets of parameters.
    """
    import inspect
    wanted = WANTED_LAUNCH_OPTIONS if wanted is None else wanted
    try:
        accepted = set(inspect.signature(launch_fn).parameters)
    except (TypeError, ValueError):
        return {}
    return {k: v for k, v in wanted.items() if k in accepted}


if __name__ == "__main__":
    _kwargs = build_launch_kwargs(demo.launch)
    # Very old Gradio versions may not accept CSS through launch().
    if "css" not in _kwargs:
        demo.css = RATING_PANEL_CSS      
    demo.launch(**_kwargs)