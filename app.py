import streamlit as st
import tempfile
import pandas as pd
import time
from langchain_community.chat_models import ChatOllama
from main import process_pdf, chatting, evaluate_response
from langchain_core.messages import HumanMessage, AIMessage

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="JupiterBrains AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------
# Custom CSS for Minimalist UI & Right Sidebar
# -------------------------------
st.markdown("""
    <style>
    /* Hide default Streamlit top header */
    header {visibility: hidden;}

    /* 1. MOVE SIDEBAR TO THE RIGHT */
    [data-testid="collapsedControl"] {
        right: 15px !important;
        left: auto !important;
        top: 15px !important;
        z-index: 10000 !important;
        color: white !important;
    }
    section[data-testid="stSidebar"] {
        right: 0 !important;
        left: auto !important;
        border-left: 1px solid #30363d;
        border-right: none;
    }

    /* 2. TOP RIBBON STYLING */
    .top-ribbon {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 15px 30px;
        background-color: #0E1117; 
        border-bottom: 1px solid #30363d;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 9999;
    }
    .app-name {
        font-size: 22px;
        font-weight: 600;
        color: #ffffff !important;
        font-family: sans-serif;
        text-decoration: none !important;
    }

    /* Navigation Container */
    .nav-container {
        display: flex;
        align-items: center;
        gap: 25px;
        margin-right: 60px; /* Leave room for the right sidebar toggle */
    }

    /* Individual Links */
    .ribbon-link {
        color: #c9d1d9 !important;
        text-decoration: none !important;
        font-size: 16px;
        font-weight: 500;
        transition: color 0.2s ease;
    }
    .ribbon-link:hover {
        color: #ffffff !important;
        text-decoration: none !important;
    }

    .auth-buttons {
        font-size: 16px;
        font-weight: 500;
        color: #ffffff;
        cursor: pointer;
        padding-left: 15px;
        border-left: 1px solid #30363d;
    }

    /* 3. HERO TEXT STYLING */
    .hero-text {
        text-align: center;
        font-size: 48px;
        font-weight: 500;
        color: #ffffff;
        margin-top: 15vh;
        margin-bottom: 2rem;
        font-family: sans-serif;
        letter-spacing: -1px;
    }

    /* Main container padding adjustments */
    .block-container {
        padding-top: 6rem !important;
    }
    </style>

    <!-- Ribbon HTML Injection -->
    <div class="top-ribbon">
        <a href="/" target="_self" class="app-name">✨ JupiterBrains AI</a>
        <div class="nav-container">
            <a href="/Info" target="_self" class="ribbon-link">Info</a>
            <a href="https://github.com/photodork/RAG" target="_blank" class="ribbon-link">Code</a>
            <a href="/About_Us" target="_self" class="ribbon-link">About Us</a>
            <div class="auth-buttons">Login / Sign up</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# -------------------------------
# Top Right Slider (Sidebar Navigation)
# -------------------------------
with st.sidebar:
    st.title("Navigation")
    menu_selection = st.radio(
        "Menu",
        ["App", "Info", "Code", "About Us"],
        label_visibility="collapsed"
    )

# -------------------------------
# Session State Management
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # For RAG Document Chat
if "basic_history" not in st.session_state:
    st.session_state.basic_history = []  # For General Local LLM Chat
if "chroma" not in st.session_state:
    st.session_state.chroma = None
    st.session_state.bm25 = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama3.2:latest"


# -------------------------------
# Reusable Function to Process PDF Upload
# -------------------------------
def handle_pdf_upload(uploaded_file):
    if uploaded_file:
        with st.status("Initializing document processing pipeline...", expanded=True) as status:
            st.write("⏳ loading the pdf...")
            time.sleep(0.8)
            st.write("✂️ splitting pdf into chunks...")
            time.sleep(0.8)
            st.write("🗄️ storing chunks into vector database...")

            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded_file.read())
                pdf_path = tmp.name

            chroma, bm25 = process_pdf(pdf_path)
            st.session_state.chroma = chroma
            st.session_state.bm25 = bm25

            status.update(label="Processing Complete!", state="complete", expanded=False)
            time.sleep(0.5)

        st.session_state.history.append(AIMessage(content="Where should we begin?"))
        st.rerun()


# -------------------------------
# Routing Based on Slider
# -------------------------------
if menu_selection == "App":

    # ==========================================
    # STATE 1: NO PDF UPLOADED (GENERAL CHAT)
    # ==========================================
    if not st.session_state.chroma:

        if not st.session_state.basic_history:
            st.markdown('<div class="hero-text">What do you want to know?</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.session_state.selected_model = st.selectbox(
                    "🤖 Choose local model for general chat",
                    ["llama3.2:latest", "llama3.1:latest", "gemma3:1b"],
                    index=["llama3.2:latest", "llama3.1:latest", "gemma3:1b"].index(st.session_state.selected_model)
                )
                uploaded_file = st.file_uploader("📄 Or upload a PDF for Document Q&A", type="pdf")
                handle_pdf_upload(uploaded_file)
        else:
            with st.expander("⚙️ General Chat Settings & PDF Upload", expanded=False):
                st.session_state.selected_model = st.selectbox(
                    "🤖 Change local model",
                    ["llama3.2:latest", "llama3.1:latest", "gemma3:1b"],
                    index=["llama3.2:latest", "llama3.1:latest", "gemma3:1b"].index(st.session_state.selected_model)
                )
                uploaded_file = st.file_uploader("📄 Upload a PDF to switch to Document Q&A Mode", type="pdf")
                handle_pdf_upload(uploaded_file)

        for msg in st.session_state.basic_history:
            if isinstance(msg, HumanMessage):
                with st.chat_message("user"):
                    st.write(msg.content)
            elif isinstance(msg, AIMessage):
                with st.chat_message("assistant"):
                    st.write(msg.content)

        user_input = st.chat_input("Ask anything to the local LLM...")
        if user_input:
            with st.chat_message("user"):
                st.write(user_input)
            st.session_state.basic_history.append(HumanMessage(content=user_input))

            with st.spinner(f"Thinking with {st.session_state.selected_model}..."):
                basic_llm = ChatOllama(model=st.session_state.selected_model)
                response = basic_llm.invoke(st.session_state.basic_history)

            with st.chat_message("assistant"):
                st.write(response.content)
            st.session_state.basic_history.append(AIMessage(content=response.content))

    # ==========================================
    # STATE 2: PDF UPLOADED (FIXED RAG MODE)
    # ==========================================
    if st.session_state.chroma:

        # Display RAG chat history
        for msg in st.session_state.history:
            if isinstance(msg, HumanMessage):
                with st.chat_message("user"):
                    st.write(msg.content)
            elif isinstance(msg, AIMessage):
                with st.chat_message("assistant"):
                    st.write(msg.content)

        # RAG Chat Input
        user_input = st.chat_input("Ask anything about your document...")

        if user_input:
            with st.chat_message("user"):
                st.write(user_input)

            # Enhanced visual feedback pipeline
            with st.status("Analyzing document with llama3.2...", expanded=True) as status:
                st.write("🔄 Rewriting user query for optimal retrieval...")
                time.sleep(5)

                st.write("🧠 Understanding context and semantic meaning...")
                time.sleep(4.7)

                st.write("🔍 Performing hybrid search (BM25 + Chroma)...")
                time.sleep(5.9)

                st.write("📑 Retrieving most relevant document chunks...")
                time.sleep(4.9)

                st.write("✍️ Structuring response and adding citations...")

                # Call core charting logic from main.py
                answer, rewritten, contexts = chatting(
                    user_input,
                    st.session_state.history,
                    st.session_state.bm25,
                    st.session_state.chroma
                )

                status.update(label="Response generated successfully!", state="complete", expanded=False)

            with st.chat_message("assistant"):
                st.write(answer)

                # Evaluation Expander
                with st.expander("📊 View Ragas Evaluation Metrics", expanded=False):
                    with st.spinner("Evaluating response quality with Ragas..."):
                        try:
                            eval_df = evaluate_response(user_input, answer, contexts)
                            col1, col2 = st.columns(2)

                            f_score = eval_df['faithfulness'][0]
                            a_score = eval_df['answer_relevancy'][0]

                            f_display = f"{f_score:.2f}" if pd.notna(f_score) else "Failed to parse"
                            a_display = f"{a_score:.2f}" if pd.notna(a_score) else "Failed to parse"

                            col1.metric("Faithfulness (0 to 1)", f_display)
                            col2.metric("Answer Relevancy (0 to 1)", a_display)

                        except Exception as e:
                            st.error(f"Evaluation encountered an error: {e}")

# (Fallback slider routing functions)
elif menu_selection == "Info":
    st.title("ℹ️ About the App")
    st.write("This application is a local Retrieval-Augmented Generation (RAG) system.")

elif menu_selection == "Code":
    st.title("💻 Source Code")
    st.write("The complete source code for this hybrid RAG pipeline and user interface can be found on GitHub.")

elif menu_selection == "About Us":
    st.title("👋 About Us")
    st.write("I am an Artificial Intelligence and Data Science (AI-DS) student at Graphic Era University (GEU).")
