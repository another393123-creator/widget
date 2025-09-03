# RAG Web App (single-file)
# Run:
#   pip install fastapi uvicorn openai pypdf scikit-learn numpy python-multipart
#   python app.py
# Then open http://localhost:8000

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import uvicorn
import os
import io
import uuid
import re
from typing import List, Dict, Any

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np


# ---------------------- RAG Store ----------------------
class DocumentStore:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def add_document(self, text: str) -> str:
        chunks = self._chunk_text(text)
        if not chunks:
            raise ValueError("No extractable text found.")
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), stop_words="english", max_features=20000
        )
        doc_matrix = vectorizer.fit_transform(chunks)
        sess_id = str(uuid.uuid4())
        self.sessions[sess_id] = {
            "chunks": chunks,
            "vectorizer": vectorizer,
            "doc_matrix": doc_matrix,
        }
        return sess_id

    def retrieve(self, session_id: str, query: str, k: int = 5) -> List[str]:
        if session_id not in self.sessions:
            raise KeyError("Session not found")
        vectorizer = self.sessions[session_id]["vectorizer"]
        doc_matrix = self.sessions[session_id]["doc_matrix"]
        q_vec = vectorizer.transform([query])
        scores = linear_kernel(q_vec, doc_matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:k]
        results = [
            self.sessions[session_id]["chunks"][i]
            for i in top_idx
            if scores[i] > 0.0
        ]
        if not results:
            # Fallback: if nothing scores above 0, return the first k chunks
            chunks = self.sessions[session_id]["chunks"]
            return chunks[: min(k, len(chunks))]
        return results

    def _chunk_text(self, text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        chunks = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + chunk_size, n)
            chunks.append(text[start:end])
            if end == n:
                break
            start = max(0, end - overlap)
        return chunks


def extract_text_from_upload(upload: UploadFile) -> str:
    name = (upload.filename or "document").lower()
    content = upload.file.read()
    if name.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(content))
            pages_text = []
            for page in reader.pages:
                try:
                    pages_text.append(page.extract_text() or "")
                except Exception:
                    pass
            text = "\n".join(pages_text).strip()
            return text
        except Exception:
            return ""
    else:
        # Fallback treat as plain text
        try:
            return content.decode("utf-8", errors="ignore")
        except Exception:
            return ""


# ---------------------- LLM Client ----------------------
def get_client() -> OpenAI:
    # Uses NVIDIA's Integrate API endpoint with an API key from env if available.
    # Falls back to the provided key if env is not set.
    api_key = (
        os.environ.get("NVAPI_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "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"
    )
    return OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)


# ---------------------- FastAPI App ----------------------
app = FastAPI(title="Minimal RAG Chat")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = DocumentStore()


@app.get("/", response_class=HTMLResponse)
def root():
    return HTML_PAGE


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    text = extract_text_from_upload(file)
    if not text or len(text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Could not extract text from the uploaded file.")
    try:
        session_id = store.add_document(text)
        return {"session_id": session_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/chat")
async def chat(request: Request):
    payload = await request.json()
    session_id = payload.get("session_id")
    user_message = (payload.get("message") or "").strip()

    if not session_id or session_id not in store.sessions:
        raise HTTPException(status_code=400, detail="Invalid or expired session_id.")
    if not user_message:
        raise HTTPException(status_code=400, detail="Empty message.")

    # Retrieve context
    context_chunks = store.retrieve(session_id, user_message, k=5)
    context = (
        "\n\n".join([f"[S{i+1}]\n{c}" for i, c in enumerate(context_chunks)])
        if context_chunks
        else "No relevant context retrieved."
    )

    # Build messages
    system_prompt = (
        "You are a helpful assistant for chatting and question answering over a user-provided document. "
        "For document-specific questions, use ONLY the provided context snippets to answer and cite them as [S#] where applicable. "
        "If a document-specific answer cannot be found in the context, reply exactly: \"I don't know based on the document.\" "
        "For greetings, small talk, or meta questions (e.g., \"hello\", \"who are you?\", \"what can you do?\"), reply in a warm, friendly, human-like tone (use natural phrasing and contractions), and keep it concise; do not cite snippets or use the fallback phrase. "
        "Respond with the final answer only (no analysis or step-by-step), in 1-3 sentences."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nUser question: {user_message}",
        },
    ]

    client = get_client()

    def stream():
        try:
            completion = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=messages,
                temperature=0.2,
                top_p=1,
                max_tokens=1024,
                stream=True,
            )
            for chunk in completion:
                delta = chunk.choices[0].delta.content
                if delta is not None:
                    yield delta
        except Exception as e:
            yield f"\n[Stream error: {str(e)}]"

    return StreamingResponse(stream(), media_type="text/plain")


# ---------------------- Inline Frontend ----------------------
HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Minimal RAG Chat</title>
  <style>
    :root {
      --bg: #0b0b0f;
      --panel: #14151a;
      --muted: #9aa0a6;
      --text: #e8eaed;
      --accent: #6ee7ff;
      --accent-2: #22d3ee;
      --border: #2a2b31;
      --good: #22c55e;
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.5 system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
    }
    .container {
      max-width: 960px;
      margin: 0 auto;
      padding: 24px;
    }
    header {
      display: flex; align-items: center; justify-content: space-between;
      margin-bottom: 24px;
    }
    header .brand {
      display: flex; align-items: center; gap: 10px; font-weight: 700; letter-spacing: .3px;
    }
    header .brand .logo {
      width: 24px; height: 24px; border: 2px solid var(--accent); border-radius: 6px;
      display: inline-block; box-shadow: 0 0 18px rgba(109, 218, 255, .15) inset;
    }
    header .tip { color: var(--muted); font-size: 12px; }

    .card {
      background: linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,.01));
      border: 1px solid var(--border);
      border-radius: 14px;
      box-shadow: 0 20px 50px rgba(0,0,0,.35);
    }

    /* Upload view */
    #upload-view { padding: 28px; }
    .dropzone {
      border: 1px dashed var(--border);
      border-radius: 12px;
      background: var(--panel);
      padding: 36px; text-align: center; color: var(--muted);
      transition: border-color .2s ease, background .2s ease;
    }
    .dropzone.dragover { border-color: var(--accent-2); background: #0f1116; }
    .dropzone input { display: none; }
    .btn {
      display: inline-flex; align-items: center; gap: 8px;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
      color: #00202b; font-weight: 700; padding: 10px 16px; border-radius: 10px;
      border: none; cursor: pointer; margin-top: 16px;
      box-shadow: 0 6px 20px rgba(34, 211, 238, .25);
    }
    .btn:disabled { opacity: .6; cursor: not-allowed; }

    /* Chat view */
    #chat-view { display: none; height: calc(100vh - 120px); }
    .chat {
      display: grid; grid-template-rows: 1fr auto; gap: 14px; height: 100%;
    }
    .messages {
      overflow: auto; padding: 18px; border: 1px solid var(--border);
      border-radius: 14px; background: var(--panel);
    }
    .msg-row { display: flex; margin-bottom: 14px; }
    .msg-row.user { justify-content: flex-end; }
    .bubble {
      max-width: min(720px, 92%);
      padding: 12px 14px; border-radius: 12px; border: 1px solid var(--border);
      white-space: pre-wrap; word-wrap: break-word;
    }
    .user .bubble { background: #0c2530; color: #ccf3ff; border-color: #193946; }
    .assistant .bubble { background: #111318; color: var(--text); }

    .composer { display: flex; gap: 10px; }
    .composer input[type="text"]{
      flex: 1; padding: 12px 14px; border-radius: 12px; border: 1px solid var(--border);
      background: #0f1116; color: var(--text);
    }
    .send { background: var(--accent-2); color: #00202b; padding: 8px 12px; border-radius: 10px; border: none; font-weight: 700; cursor: pointer; }
    .muted { color: var(--muted); }

    .footer-tip { text-align: center; margin-top: 10px; color: var(--muted); font-size: 12px; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="brand"><span class="logo"></span>Minimal RAG</div>
      <div class="tip">Upload a document, then chat about it.</div>
    </header>

    <!-- Upload View -->
    <div id="upload-view" class="card">
      <div id="dropzone" class="dropzone">
        <h3>Drop a .pdf or .txt file here</h3>
        <p class="muted">or click to choose a file</p>
        <input id="file-input" type="file" accept=".pdf,.txt" />
        <div>
          <button id="upload-btn" class="btn">Upload</button>
        </div>
        <div id="upload-status" class="muted" style="margin-top:8px"></div>
      </div>
    </div>

    <!-- Chat View -->
    <div id="chat-view" class="card">
      <div class="chat">
        <div id="messages" class="messages"></div>
        <div class="composer">
          <input id="prompt" type="text" placeholder="Ask something about your document..." />
          <button id="send" class="send">Send</button>
        </div>
      </div>
      <div class="footer-tip">Answers are grounded in your uploaded document. If it isn't in the doc, the assistant will say it doesn't know.</div>
    </div>
  </div>

  <script>
    const state = {
      sessionId: null,
      streaming: false,
    };

    const $ = (id) => document.getElementById(id);
    const dropzone = $("dropzone");
    const fileInput = $("file-input");
    const uploadBtn = $("upload-btn");
    const uploadStatus = $("upload-status");

    dropzone.addEventListener('click', () => fileInput.click());

    ;['dragenter','dragover'].forEach(evt => dropzone.addEventListener(evt, e => {
      e.preventDefault(); e.stopPropagation(); dropzone.classList.add('dragover');
    }));
    ;['dragleave','drop'].forEach(evt => dropzone.addEventListener(evt, e => {
      e.preventDefault(); e.stopPropagation(); dropzone.classList.remove('dragover');
    }));

    dropzone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files && files.length) fileInput.files = files;
    });

    uploadBtn.addEventListener('click', async () => {
      if (!fileInput.files || !fileInput.files[0]) { uploadStatus.textContent = 'Choose a .pdf or .txt file first.'; return; }
      uploadBtn.disabled = true; uploadBtn.textContent = 'Uploading...';
      uploadStatus.textContent = '';
      try {
        const form = new FormData();
        form.append('file', fileInput.files[0]);
        const res = await fetch('/upload', { method: 'POST', body: form });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        state.sessionId = data.session_id;
        // Switch UI
        document.getElementById('upload-view').style.display = 'none';
        document.getElementById('chat-view').style.display = 'block';
        pushAssistant("Upload successful. Ask me anything about your document.");
      } catch (err) {
        uploadStatus.textContent = 'Upload failed: ' + (err?.message || err);
      } finally {
        uploadBtn.disabled = false; uploadBtn.textContent = 'Upload';
      }
    });

    const messagesEl = $("messages");
    const promptEl = $("prompt");
    const sendBtn = $("send");

    function pushUser(text) {
      const row = document.createElement('div'); row.className = 'msg-row user';
      const b = document.createElement('div'); b.className = 'bubble'; b.textContent = text;
      row.appendChild(b); messagesEl.appendChild(row); messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function pushAssistant(text = '') {
      const row = document.createElement('div'); row.className = 'msg-row assistant';
      const b = document.createElement('div'); b.className = 'bubble'; b.textContent = text;
      row.appendChild(b); messagesEl.appendChild(row); messagesEl.scrollTop = messagesEl.scrollHeight;
      return b;
    }

    async function ask(question) {
      if (!state.sessionId) return;
      pushUser(question);
      const bubble = pushAssistant('');
      sendBtn.disabled = true; promptEl.disabled = true; state.streaming = true;
      try {
        const res = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: state.sessionId, message: question })
        });
        if (!res.ok) throw new Error(await res.text());
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let acc = '';
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          acc += decoder.decode(value, { stream: true });
          bubble.textContent = acc;
          messagesEl.scrollTop = messagesEl.scrollHeight;
        }
      } catch (err) {
        bubble.textContent += "\n[Error] " + (err?.message || err);
      } finally {
        sendBtn.disabled = false; promptEl.disabled = false; state.streaming = false;
      }
    }

    sendBtn.addEventListener('click', () => {
      const q = promptEl.value.trim(); if (!q) return; promptEl.value = ''; ask(q);
    });
    promptEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendBtn.click(); }
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
