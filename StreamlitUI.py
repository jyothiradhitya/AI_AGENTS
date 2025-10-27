# StreamlitUI.py
import streamlit as st
import asyncio
import nest_asyncio
from Coordinator import CoordinatorAgent
from datetime import datetime

nest_asyncio.apply()  # allows asyncio inside Streamlit

# -------------------------------
# Initialize Coordinator
# -------------------------------
if "coordinator" not in st.session_state:
    st.session_state.coordinator = CoordinatorAgent()
    st.session_state.answer = ""
    st.session_state.flow_running = False
    st.session_state.coordinator_task = None

coordinator = st.session_state.coordinator

# -------------------------------
# UI Elements
# -------------------------------
st.title("AI SQL & Document Chatbot")

uploaded_files = st.file_uploader(
    "Upload PDF / DOCX / PPTX / CSV / TXT",
    type=["pdf", "docx", "pptx", "csv", "txt"],
    accept_multiple_files=True
)

query = st.text_input("Enter your query", value="What is in the document?")
run_flow_btn = st.button("Run Flow")

preview_container = st.empty()
answer_container = st.empty()

# -------------------------------
# Async Flow Runner
# -------------------------------
async def run_flow(files, query_text):
    st.session_state.flow_running = True

    # Start coordinator loop if not running
    if st.session_state.coordinator_task is None:
        st.session_state.coordinator_task = asyncio.create_task(coordinator.run())

    # Start the flow
    await coordinator.start_flow(files, query_text)

    # Poll for preview and answer
    preview_shown = False
    while True:
        # Preview
        preview = coordinator.state.get("preview")
        if preview and not preview_shown:
            #preview_container.subheader("Preview (first chunk):")
            #preview_container.write(preview[:500] + "..." if len(preview) > 500 else preview)
            preview_shown = True

        # Final answer
        answer = coordinator.state.get("answer")
        if answer:
            answer_container.subheader("LLM Answer:")
            answer_container.write(answer)
            break

        await asyncio.sleep(0.5)

    st.session_state.flow_running = False

# -------------------------------
# Run Flow Button
# -------------------------------
if run_flow_btn and not st.session_state.flow_running:
    if not uploaded_files:
        st.warning("Please upload at least one file!")
    elif not query:
        st.warning("Please enter a query!")
    else:
        # Run async RAG flow
        asyncio.run(run_flow(uploaded_files, query))
