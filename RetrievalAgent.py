from datetime import datetime
from mcp_models import MCPMessage

class RetrievalAgentSync:
    def __init__(self, out_queue):
        self.out_queue = out_queue
        self.name = "RetrievalAgent"
        self.docs = []

    def log(self, text: str):
        print(f"[{self.name}] {text}")

    async def handle(self, msg):
        if msg.type == "INGESTION_RESULT":
            self.docs = msg.payload.get("files", [])
            self.log(f"Received INGESTION_RESULT with {len(self.docs)} docs")
        
        elif msg.type == "RETRIEVE":
            # Get docs from payload if self.docs is empty
            if not self.docs:
                self.docs = msg.payload.get("docs", [])
                self.log(f"Received docs from RESTRICT payload, count: {len(self.docs)}")

            query = msg.payload.get("query", "")
            self.log(f"Processing query: {query}")

            # Step 1: Preview response
            preview_text = ""
            if self.docs:
                first_doc = self.docs[0].get("text", "")
                preview_text = first_doc[:500] if isinstance(first_doc, str) else str(first_doc)
                self.log(f"First 200 chars of first doc:\n{first_doc[:200]}")

            await self.out_queue.put(
                MCPMessage(
                    trace_id=msg.trace_id,
                    type="PREVIEW_RESPONSE",
                    sender=self.name,
                    receiver="CoordinatorAgent",
                    timestamp=datetime.utcnow(),
                    payload={"query": query, "preview": preview_text}
                )
            )
            self.log("Sent PREVIEW_RESPONSE → CoordinatorAgent")

            # Step 2: Full context
            full_text = ""
            if self.docs:
                full_text = "\n\n".join([doc.get("text", "") for doc in self.docs])
                self.log(f"Full context length: {len(full_text)} chars")

            await self.out_queue.put(
                MCPMessage(
                    trace_id=msg.trace_id,
                    type="CONTEXT_RESPONSE",
                    sender=self.name,
                    receiver="CoordinatorAgent",
                    timestamp=datetime.utcnow(),
                    payload={"context": full_text}  # ✅ actual text
                )
            )
            self.log("Sent CONTEXT_RESPONSE → CoordinatorAgent")
