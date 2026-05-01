import streamlit as st

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Project Info - JupiterBrains",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------
# Custom CSS & Top Ribbon
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
        margin-right: 60px; /* Leaves room for the sidebar toggle */
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

    /* Main container padding adjustments so content doesn't hide behind ribbon */
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
# Page Content
# -------------------------------
st.title("📘 Project Information")

st.markdown("""
## 🔍 What is this project?

This is a **RAG-based PDF Chat System** that allows users to:
- Upload PDFs
- Ask questions
- Get citation-backed answers

---

## ⚙️ Architecture

### 📄 PDF Pipeline
- Load PDF
- Chunk text
- Generate embeddings
- Store in vector DB (Chroma)

### 🔎 Retrieval
- Hybrid Search:
  - BM25 (keyword)
  - Chroma (semantic)

### 🧠 LLM Layer
- Query rewriting
- Context injection
- Answer generation

---

## 🧪 Evaluation
- Faithfulness
- Answer Relevancy
(using Ragas)

---

## 🛠️ Tech Stack
- Streamlit
- LangChain
- Ollama (LLMs)
- ChromaDB
- BM25
- Ragas
""")
