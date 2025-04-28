import streamlit as st
import requests
import base64
from PIL import Image
import io
import json
import speech_recognition as sr
import PyPDF2

# ----------------------- Utility Functions -----------------------

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_ollama_response(image_base64, text, model):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": text,
        "stream": False,
    }
    if image_base64:
        payload["images"] = [image_base64]
    response = requests.post(url, data=json.dumps(payload))
    return json.loads(response.text)["response"]

def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Could not understand audio."
    except sr.RequestError:
        return "API unavailable or request error."

def read_file_content(uploaded_file):
    if uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        return "Unsupported file format."

# ----------------------- Main App -----------------------

def main():
    st.set_page_config(page_title="Multimodal Chatbot", page_icon="🧠", layout="wide")

    st.markdown("""
    <style>
    .stTextInput>div>div>input {
        background-color: #f0f2f6;
        border-radius: 10px;
    }
    .stTextArea>div>textarea {
        background-color: #f0f2f6;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.6em;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        padding: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🧠 Multimodal Chatbot")
    st.caption("Interact with text, images, audio, and files seamlessly.")

    initial_message = "Hello 👋 Upload an image, audio, or file and ask a question!"

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": initial_message}]

    with st.sidebar:
        st.header("⚙️ Settings")
        model_choice = st.selectbox("Choose a model:", ["llava", "llava-phi3"], index=0)
        if st.button("🔄 Clear Chat"):
            st.session_state.messages = [{"role": "assistant", "content": initial_message}]
            st.experimental_rerun()

    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Image", "🎙️ Audio", "📄 File", "💬 Text"])

    prompt = None
    image_base64 = None

    # ---- Image Tab ----
    with tab1:
        uploaded_image = st.file_uploader("Upload an image (JPEG)", type=["jpeg", "jpg"])
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, use_container_width=True)
            image_base64 = image_to_base64(image)
            prompt = st.text_input("Ask a question about the image")

    # ---- Audio Tab ----
    with tab2:
        audio_file = st.file_uploader("Upload an audio file (WAV)", type=["wav"])
        if audio_file:
            with st.spinner("Transcribing..."):
                transcription = transcribe_audio(audio_file)
                st.success("Transcription complete!")
                st.text_area("Transcribed Text:", transcription, height=150)
                prompt = transcription

    # ---- File Tab ----
    with tab3:
        uploaded_file = st.file_uploader("Upload a file (TXT or PDF)", type=["txt", "pdf"])
        if uploaded_file:
            content = read_file_content(uploaded_file)
            st.text_area("Extracted Content:", value=content, height=200)
            if st.button("Ask a question about the file"):
                prompt = content[:3000]

    # ---- Text Tab ----
    with tab4:
        prompt_text = st.text_input("Enter your question")
        if prompt_text:
            prompt = prompt_text

    # ---- Display Chat ----
    st.divider()
    st.subheader("💬 Conversation")
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f"**You:** {msg['content']}")
        else:
            with st.chat_message("assistant"):
                st.markdown(f"**Bot:** {msg['content']}")

    # ---- Response ----
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(f"**You:** {prompt}")

        with st.spinner("Generating response..."):
            response = get_ollama_response(image_base64, prompt, model_choice)
        st.session_state.messages.append({"role": "assistant", "content": response})

        with st.chat_message("assistant"):
            st.markdown(f"**Bot:** {response}")

if __name__ == "__main__":
    main()
