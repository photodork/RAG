import streamlit as st

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="About Us - JupiterBrains",
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

    /* 1. MOVE SIDEBAR TO THE RIGHT (If you want the slider here too) */
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
        text-decoration: none;
        font-size: 16px;
        font-weight: 500;
        transition: color 0.2s ease;
        text-decoration: none !important;
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
st.title("👋 About Us")

st.markdown("""
Welcome to **JupiterBrains**! 

I am currently pursuing my specialization in Artificial Intelligence and Data Science (AI-DS) at Graphic Era University (GEU). My work primarily exists at the intersection of AI architecture and modern, minimalist UI/UX design.

### What We Focus On
*   **Privacy-First AI & SLMs:** Developing and deploying local, air-gapped Small Language Models for enterprises. We prioritize data privacy and cost-efficiency over relying solely on cloud-based APIs.
*   **Advanced RAG Pipelines:** Building robust retrieval systems using tools like Ollama, ChromaDB, and hybrid search methods (BM25 + Semantic) for secure document chatting.
*   **Modern Digital Experiences:** Crafting clean web interfaces and CMS structures to bring highly technical backends to the surface seamlessly.

### Select Projects
*   **Cactus Broadcast Media Reconciliation Engine:** Architecting a multi-stage pipeline using deterministic hashing and semantic vector matches to automate the matching of vendor invoices against media buy summaries.
*   **Predictive Tooling:** Developed and deployed machine learning models, including a Password Crack-Time Predictor using Linear Regression.
*   **Local AI Assistants:** Engineering the very RAG application you are currently exploring.

Beyond coding and building AI agents, I have a deep fascination with sports analytics—specifically how data can be leveraged to gain competitive advantages in cricket and combat sports—and I am an avid consumer of diverse music, ranging from Turkish pop to 90s rock.
""")