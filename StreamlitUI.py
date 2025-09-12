# StreamlitUI.py
import streamlit as st
import asyncio
import nest_asyncio
from Coordinator import CoordinatorAgent
from datetime import datetime

nest_asyncio.apply()  # allows asyncio inside Streamlit

# Initialize Coordinator
if "coordinator" not in st.session_state:
    st.session_state.coordinator = CoordinatorAgent()
    st.session_state.answer = ""
    st.session_state.flow_running = False

coordinator = st.session_state.coordinator

st.title("AI SQL & Visualization Chatbot")

uploaded_file = st.file_uploader(
    "Upload PDF / DOCX / PPTX / CSV / TXT",
    type=["pdf", "docx", "pptx", "csv", "txt"]
)
query = st.text_input("Enter your query", value="What is in the document?")
run_flow_btn = st.button("Run Flow")

answer_container = st.empty()

# Async helper
async def run_flow(file, query_text):
    st.session_state.flow_running = True

    # Start coordinator loop if not running
    if "coordinator_task" not in st.session_state:
        st.session_state.coordinator_task = asyncio.create_task(coordinator.run())

    # Start the flow
    await coordinator.start_flow([file], query_text)

    # Wait until LLM_RESPONSE is received
    while "answer" not in coordinator.state or not coordinator.state["answer"]:
        await asyncio.sleep(0.5)

    # Display only the final answer
    st.session_state.answer = coordinator.state["answer"]
    answer_container.subheader("LLM Answer:")
    answer_container.write(st.session_state.answer)

    st.session_state.flow_running = False

# Run flow on button click
if run_flow_btn and not st.session_state.flow_running:
    if uploaded_file is None:
        st.warning("Please upload a file first!")
    elif not query:
        st.warning("Please enter a query!")
    else:
        asyncio.run(run_flow(uploaded_file, query))
