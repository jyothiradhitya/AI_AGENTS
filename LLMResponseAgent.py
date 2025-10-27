from google import genai
from datetime import datetime
import streamlit as st
import asyncio
from mcp_models import MCPMessage
from dotenv import load_dotenv
import os

class LLMResponseAgent:
    def __init__(self, out_queue):
        self.out_queue = out_queue
        self.name = "LLMResponseAgent"
        load_dotenv()
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def log(self, text: str):
        msg = f"[{self.name}] {text}"
        print(msg)
        if "logs" not in st.session_state:
            st.session_state.logs = []
        st.session_state.logs.append(msg)

    async def handle(self, msg: MCPMessage):
        self.log(f"handle: Received {msg.type} from {msg.sender}")

        if msg.type != "LLM_REQUEST":
            self.log("Ignoring non-LLM_REQUEST message")
            return

        context = msg.payload.get("context", "")
        query = msg.payload.get("query", "")
        if not context.strip():
            context = "(No context provided)"

        MAX_CHARS = 1500
        context_short = context[:MAX_CHARS]

        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents = f"""You are an expert AI assistant. Using the context provided below, explain the answer to the query clearly and in detail. 
If the context contains specific data or examples, include them in your explanation.

Context:
{context_short}

Query:
{query}
"""
 )
            )
            final_text = getattr(response, "text", "(No response text)")
        except Exception as e:
            final_text = f"LLM error: {e}"

        out_msg = MCPMessage(
            trace_id=msg.trace_id,
            type="LLM_RESPONSE",
            sender=self.name,
            receiver="CoordinatorAgent",
            timestamp=datetime.utcnow(),
            payload={"answer": final_text}
        )
        await self.out_queue.put(out_msg)
        self.log("Sent final LLM_RESPONSE → CoordinatorAgent")

