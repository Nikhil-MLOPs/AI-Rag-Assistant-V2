import gradio as gr
import requests
import os

API_URL = os.getenv("API_URL", "http://api:8000/query")

# -------------------------------------------------
# 🎨 CSS
# -------------------------------------------------

CSS = """
html, body, .gradio-container {
    height: 100%;
    margin: 0;
}

#chat {
    height: calc(100vh - 240px);
    overflow-y: auto;
}

/* Footer styling */
.footer {
    position: fixed;
    bottom: 0;
    width: 100%;
    padding: 14px 0;
    text-align: center;
    font-size: 13px;
    color: #6b7280;
    background: linear-gradient(to right, #f9fafb, #ffffff, #f9fafb);
    border-top: 1px solid #e5e7eb;
}

.footer strong {
    color: #111827;
    font-weight: 600;
}

.footer span {
    margin: 0 8px;
    color: #9ca3af;
}
"""

# -------------------------------------------------
# 🔄 Streaming client (Gradio-safe)
# -------------------------------------------------

def ask(question, history):
    history = history or []
    history = history + [{"role": "user", "content": question}]
    assistant = {"role": "assistant", "content": ""}

    with requests.post(
        API_URL,
        json={"question": question},
        stream=True,
    ) as r:
        if r.status_code != 200:
            assistant["content"] = "❌ Backend error. Please try again."
            yield history + [assistant]
            return

        for chunk in r.iter_content(chunk_size=None):
            token = chunk.decode("utf-8")
            assistant["content"] += token
            yield history + [assistant]

# -------------------------------------------------
# 🧠 UI
# -------------------------------------------------

with gr.Blocks() as demo:
    gr.Markdown("## 🧠 AI Medical RAG Assistant")

    chatbot = gr.Chatbot(elem_id="chat")

    with gr.Row():
        txt = gr.Textbox(
            placeholder="Ask a medical question…",
            show_label=False,
            scale=8,
        )
        btn = gr.Button("Send", scale=1)

    spinner = gr.Markdown("⏳ **Thinking…**", visible=False)

    def wrapped_ask(question, history):
        spinner.visible = True
        yielded = False

        for update in ask(question, history):
            yielded = True
            spinner.visible = False
            yield update

        if not yielded:
            spinner.visible = False
            yield history + [
                {"role": "assistant", "content": "⚠️ No response received."}
            ]

    btn.click(
        wrapped_ask,
        inputs=[txt, chatbot],
        outputs=chatbot,
    )

    txt.submit(
        wrapped_ask,
        inputs=[txt, chatbot],
        outputs=chatbot,
    )

    # -------------------------------------------------
    # ℹ️ Footer (clean & professional)
    # -------------------------------------------------

    gr.Markdown(
        """
<div class="footer">
    <strong>AI Medical RAG Assistant</strong>
    <span>•</span>
    Evidence-based document retrieval
    <span>•</span>
    Educational use only — not medical advice
</div>
"""
    )

# -------------------------------------------------
# 🚀 Launch (CSS goes here to avoid warning)
# -------------------------------------------------

demo.queue().launch(
    server_name="0.0.0.0",
    server_port=7860,
    css=CSS,
    share=True
)
