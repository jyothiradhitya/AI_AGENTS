from datetime import datetime
from mcp_models import MCPMessage
from sentence_transformers import SentenceTransformer, util
import torch

class RetrievalAgentSync:
    def __init__(self, out_queue, model_name="all-MiniLM-L6-v2", top_k=3):
        self.out_queue = out_queue
        self.name = "RetrievalAgent"
        self.docs = []  # raw docs with chunks
        self.model = SentenceTransformer(model_name)
        self.top_k = top_k

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

            # --- Flatten all chunks ---
            all_chunks = []
            for doc in self.docs:
                chunks = doc.get("chunks", [])
                all_chunks.extend(chunks)
            self.log(f"Total chunks available: {len(all_chunks)}")

            if not all_chunks:
                await self.out_queue.put(
                    MCPMessage(
                        trace_id=msg.trace_id,
                        type="CONTEXT_RESPONSE",
                        sender=self.name,
                        receiver="CoordinatorAgent",
                        timestamp=datetime.utcnow(),
                        payload={"context": ""}
                    )
                )
                return

            # --- Encode chunks and query ---
            chunk_embeddings = self.model.encode(all_chunks, convert_to_tensor=True)
            query_embedding = self.model.encode([query], convert_to_tensor=True)

            # --- Compute cosine similarity ---
            cos_scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
            top_results = torch.topk(cos_scores, k=min(self.top_k, len(all_chunks)))

            # --- Get top-k relevant chunks ---
            relevant_chunks = [all_chunks[idx] for idx in top_results.indices]

            # --- Send preview response (first chunk) ---
            preview_text = relevant_chunks[0][:500] if relevant_chunks else ""
            self.log(f"Preview (first chunk 200 chars):\n{preview_text[:200]}")
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

            # --- Send full context response (top-k chunks joined) ---
            full_context = "\n\n".join(relevant_chunks)
            self.log(f"Full context length: {len(full_context)} chars")
            await self.out_queue.put(
                MCPMessage(
                    trace_id=msg.trace_id,
                    type="CONTEXT_RESPONSE",
                    sender=self.name,
                    receiver="CoordinatorAgent",
                    timestamp=datetime.utcnow(),
                    payload={"context": full_context}
                )
            )
            self.log("Sent CONTEXT_RESPONSE → CoordinatorAgent")
