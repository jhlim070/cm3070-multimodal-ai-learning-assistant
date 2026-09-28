"""
End-to-end smoke test for the running Gradio application.

This file is not collected by pytest because its filename does not start with
`test_`. Run it directly (if no file names are changed):

    python tests/e2e_gradio_smoke.py

Set PROJECT_ROOT if the test file is not located in <project>/tests/.

The test launches the real application and exercises it through Gradio's HTTP
API using gradio_client. It covers login, profile locking, file upload and
duplicate detection, the rating gate, continuation, input validation, chat
reset, and document clarification.

Only the LLM response is mocked with canned replies. The remaining application
components are exercised normally, including PDF extraction, chunking,
ChromaDB, embeddings, SQLite, and Gradio session state.

This complements the pytest suite, which calls handler functions directly.
Running the application end-to-end verifies that those handlers are correctly
connected and that state stored in Gradio components persists across requests.

SAFETY: the working directory is changed to a temporary directory before the
application is imported. Relative application paths such as learner.db and
./chroma_db therefore resolve inside the temporary directory rather than the
real project data.

The first run may take longer because the embedding model and PaddleOCR may
need to be loaded.
"""
import os
import sys
import tempfile
import fitz  # PyMuPDF, a project dependency


_here = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.abspath(os.path.join(_here, "..")))
sys.path.insert(0, PROJECT_ROOT)
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
# Run the application in an isolated temporary working directory so that
# relative database and ChromaDB paths cannot affect the real project data.
os.chdir(tempfile.mkdtemp(prefix="e2e_scratch_"))


# Use a real multi-page PDF so the production PDF-processing, chunking,
# embedding, and ChromaDB pipeline is exercised rather than mocked.
pdf = os.path.join(tempfile.mkdtemp(), "CM3020 AI_lecture 1.pdf")
d = fitz.open()
for i in range(6):
    p = d.new_page()
    p.insert_text((72, 72), f"Slide {i}. The four approaches to AI are thinking humanly, thinking rationally, acting humanly, acting rationally. Page {i} unique{i}.")
d.save(pdf); d.close()

import engine
# Replace only the LLM call with a deterministic response. All surrounding
# application and retrieval logic remains real.
CALLS = []
REPLY = {"text": "Here is a real answer about the four approaches."}
def fake_chat(model, messages, **kw):
    CALLS.append({**kw, "messages": messages})
    return {"message": {"content": REPLY["text"]}}
engine._client.chat = fake_chat

import app
from gradio_client import Client, handle_file
_, url, _ = app.demo.launch(prevent_thread_lock=True, quiet=True)
c = Client(url, verbose=False)
# Inspect the registered API endpoints for a basic startup sanity check.
apis = [k for k in c.view_api(return_format="dict")["named_endpoints"]]
print("endpoints:", len(apis), "found")

FAILS = []
def check(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + ("" if cond else f"   <-- {detail}"))
    if not cond: FAILS.append(name)

U = "e2e user"
def ask(msg, user=U): return c.predict(msg, user, None, api_name="/respond")
def rate(n):
    ep = {1:"/partial",2:"/partial_1",3:"/partial_2",4:"/partial_3",5:"/partial_4"}[n]
    return c.predict(U, api_name=ep)

print("\n--- profile lock ---")
r = c.predict(U, api_name="/handle_login")
check("new user logs in", "Welcome" in r[3], r)
r = c.predict(U, "visual examples", "beginner", "robots,  ai ,, ", api_name="/toggle_profile")
check("save locks profile + cleans topics", "locked" in r[3].lower() and r[2]["interactive"] is False and r[2]["value"] == "robots, ai", r)
r = c.predict(U, "visual examples", "beginner", "robots, ai", api_name="/toggle_profile")
check("second click = Edit -> unlocks", "Editing" in r[3], r)
r = c.predict(U, "step-by-step", "beginner", "x" * 400, api_name="/toggle_profile")
check("over-long topics rejected, not saved", r[3].startswith("!!!"), r)
r = c.predict("bad;id", api_name="/handle_login")
check("malformed User ID rejected at login", "!!!" in r[3], r)

print("\n--- upload / duplicate rejection ---")
r = c.predict([handle_file(pdf)], U, api_name="/upload_and_index")
check("real PDF indexed via real chunker+Chroma", "✅" in r and "Indexed" in r, r)
r = c.predict([handle_file(pdf)], U, api_name="/upload_and_index")
check("same file again -> skipped as already indexed", "already indexed" in r and "✅" not in r, r)
copy = os.path.join(tempfile.mkdtemp(), "renamed copy.pdf"); open(copy,"wb").write(open(pdf,"rb").read())
r = c.predict([handle_file(copy)], U, api_name="/upload_and_index")
check("same CONTENT, new name -> skipped by hash", "already indexed" in r, r)
d2 = fitz.open(); pg = d2.new_page(); pg.insert_text((72,72),"totally different text"); other = os.path.join(tempfile.mkdtemp(), os.path.basename(pdf)); d2.save(other); d2.close()
r = c.predict([handle_file(other)], U, api_name="/upload_and_index")
check("different content, same NAME -> rejected", "different file is already indexed" in r, r)
junk = os.path.join(tempfile.mkdtemp(), "evil.pdf"); open(junk,"wb").write(b"MZ\x90\x00 not a pdf at all")
r = c.predict([handle_file(junk)], U, api_name="/upload_and_index")
check("renamed non-PDF rejected by magic-byte check", "rejected" in r and "do not match" in r, r)
r = c.predict(U, api_name="/show_indexed_files")
check("registry lists exactly one file", r.count("•") == 1, r)

print("\n--- rating gate ---")
REPLY["text"] = "Here is a real answer about the four approaches."
a = ask("What are the four approaches to AI in CM3020 lecture 1?")
check("real answer returned", "real answer" in a[0], a)
check("chat() received num_ctx/temperature options", CALLS and CALLS[-1].get("options", {}).get("num_ctx") == 8192, CALLS[-1:] )
a = ask("another question")
check("gate blocks 2nd question until rated", "rate your understanding" in a[0], a)
r = rate(4)
check("rating logged", "Mostly understood" in r, r)
REPLY["text"] = "NOT_IN_MATERIALS"
a = ask("What is 5NF?")
check("refusal token -> clean not-in-materials message", "doesn't appear in the study materials" in a[0], a)
a2 = ask("and what about 6NF?")
check("non-answer does NOT open the gate", "rate your understanding" not in a2[0], a2)
REPLY["text"] = "The model said something with does not contain the answer in the provided context."
a = ask("Q with free-text refusal")
check("free-text refusal caught by fallback detector", "doesn't appear in the study materials" in a[0], a)

print("\n--- 'continue' ---")
REPLY["text"] = "Here is a real answer about the four approaches."
a = ask("Explain acting rationally in CM3020 lecture 1")
check("setup: real answer opens gate", "real answer" in a[0], a)
rate(5)
REPLY["text"] = "More detail continuing the explanation."
a = ask("continue")
check("'continue' answered", "continuing" in a[0], a)
a = ask("Next question please")
check("answer to 'continue' does NOT force a rating", "rate your understanding" not in a[0], a)

print("\n--- validation on chat ---")
a = ask("x" * 2500); check("over-long question rejected", a[0].startswith("⚠️"), a[0][:80])
a = ask("   ");       check("blank question rejected", a[0].startswith("⚠️"), a[0][:80])
a = ask("hello", user="bad;user"); check("bad user id rejected in chat", a[0].startswith("⚠️"), a[0][:80])

print("\n--- clear chat / reset ---")
r = c.predict(U, api_name="/release_gate"); check("release_gate endpoint responds", r is not None, r)
r = c.predict(U, api_name="/reset_materials"); check("reset clears index", "Cleared 1" in r, r)
r = c.predict([handle_file(pdf)], U, api_name="/upload_and_index")
check("after reset the same PDF can be indexed again", "✅" in r, r)


print("\n--- clarification: the reply answers the ORIGINAL question ---")
def make_pdf(name, text, pages=3):
    dd = fitz.open()
    for i in range(pages):
        pg = dd.new_page(); pg.insert_text((72, 72), text + f" Page {i}.")
    path = os.path.join(tempfile.mkdtemp(), name); dd.save(path); dd.close(); return path
SAME = "Benefits of testing include documenting behaviour, enabling safe refactoring and giving faster feedback."
docA = make_pdf("CM3050 Testing deck.pdf", SAME)
# Use identical document text so both documents have the same retrieval
# distance. This intentionally creates an ambiguous result that should trigger
# document clarification.
docB = make_pdf("CM3010 Schema deck.pdf", SAME)
c.predict(U, api_name="/reset_materials")
r = c.predict([handle_file(docA), handle_file(docB)], U, api_name="/upload_and_index")
check("two look-alike documents indexed", r.count("✅") == 2, r)
REPLY["text"] = "Here is a real answer about testing."
QQ = "What are the benefits of testing?"

a = ask(QQ)
check("ambiguous question -> numbered clarification", a[0].startswith("🤔") and '1. "' in a[0] and '2. "' in a[0], a[0][:120])
n_calls = len(CALLS)
a = ask("CM3050 Testing deck.pdf")
check("naming the document -> a real answer", "real answer about testing" in a[0], a[0][:120])
sent = CALLS[-1]["messages"][1]["content"]
check("the LLM was asked the ORIGINAL question, not the file name", sent.rstrip().endswith("Question: " + QQ), sent[-160:])
check("the LLM saw ONLY the chosen document (not the look-alike)", "CM3050 Testing deck.pdf" in sent and "CM3010 Schema deck.pdf" not in sent, sent[:200])
rate(4)

a = ask(QQ)
check("same ambiguity again", a[0].startswith("🤔"), a[0][:80])
a = ask("2")
sent = CALLS[-1]["messages"][1]["content"]
check("replying with a number also answers the original question", sent.rstrip().endswith("Question: " + QQ), sent[-160:])
rate(5)

a = ask("the mobile one")
# Start a fresh ambiguous question after the previous clarification cycle.
a = ask(QQ)
# An unrecognised clarification response should trigger the clarification
# prompt again rather than being treated as a new question.
a = ask("the one about nothing")
check("an unrecognised reply re-asks instead of answering the words", a[0].startswith("🤔") and '1. "' in a[0], a[0][:120])
r = c.predict(U, api_name="/clear_chat")
check("Clear chat endpoint responds", r is not None, r)
# Verify that clearing the chat removes the previous clarification context.
n_before = len(CALLS)
a = ask("CM3050 Testing deck.pdf")
sent = CALLS[-1]["messages"][1]["content"] if len(CALLS) > n_before else ""
check("after Clear chat the old question is absent from the LLM prompt", QQ not in sent, sent[-160:])

print("\n==== e2e:", "ALL PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
app.demo.close()
sys.exit(1 if FAILS else 0)