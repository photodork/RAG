from main import process_pdf, chatting, evaluate_response
import streamlit as st
import tempfile
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import pandas as pd


st.set_page_config(page_title="Data Privacy in GenAI Models", layout="wide")

st.title("📄 Chat with your PDF")

# -------------------------------
# Session State
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "chroma" not in st.session_state:
    st.session_state.chroma = None
    st.session_state.bm25 = None


# -------------------------------
# Upload PDF
# -------------------------------
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    st.success("PDF uploaded. Processing...")

    chroma, bm25 = process_pdf(pdf_path)

    st.session_state.chroma = chroma
    st.session_state.bm25 = bm25

    st.success("Ready to chat!")

# -------------------------------
# Chat UI
# -------------------------------
if st.session_state.chroma:

    # 1. Display existing chat history FIRST
    for msg in st.session_state.history:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.write(msg.content)

    # 2. Accept New Input
    user_input = st.chat_input("Ask a question...")

    if user_input:
        # Show User Message instantly
        with st.chat_message("user"):
            st.write(user_input)

        # Generate Response
        with st.spinner("Thinking..."):
            answer, rewritten, contexts = chatting(
                user_input,
                st.session_state.history,
                st.session_state.bm25,
                st.session_state.chroma
            )

        # Show Assistant Message
        with st.chat_message("assistant"):
            st.write(answer)

            # -------------------------------
            # Run Ragas Evaluation
            # -------------------------------
            with st.expander("📊 View Ragas Evaluation Metrics", expanded=True):
                with st.spinner("Evaluating response quality with Ragas (this may take a minute on local hardware)..."):
                    try:
                        eval_df = evaluate_response(user_input, answer, contexts)

                        col1, col2 = st.columns(2)

                        # FIXED: Use standard pd.isna() or pd.notna()
                        f_score = eval_df['faithfulness'][0]
                        a_score = eval_df['answer_relevancy'][0]

                        # Format the scores safely
                        f_display = f"{f_score:.2f}" if pd.notna(f_score) else "Failed to parse"
                        a_display = f"{a_score:.2f}" if pd.notna(a_score) else "Failed to parse"

                        col1.metric("Faithfulness (0 to 1)", f_display)
                        col2.metric("Answer Relevancy (0 to 1)", a_display)

                        st.caption(
                            "*(Scores closer to 1.0 are better. 'Failed to parse' means the local LLM struggled with the JSON formatting.)*")

                    except Exception as e:
                        st.error(f"Evaluation encountered a critical error: {e}")