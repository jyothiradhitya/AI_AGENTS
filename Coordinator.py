import asyncio
from datetime import datetime
from mcp_models import MCPMessage, new_trace_id
from IngestionAgent import IngestionAgentSync
from RetrievalAgent import RetrievalAgentSync
from LLMResponseAgent import LLMResponseAgent

class CoordinatorAgent:
    def __init__(self):
        self.name = "CoordinatorAgent"
        self.state = {}
        self.logs = []

        # Communication queues
        self.in_queue = asyncio.Queue()
        self.out_queue = asyncio.Queue()

        # Agents
        self.ingestion = IngestionAgentSync(out_queue=self.in_queue)
        self.retrieval = RetrievalAgentSync(out_queue=self.in_queue)
        self.llm = LLMResponseAgent(out_queue=self.in_queue)

    def log(self, text: str):
        msg = f"[{self.name}] {text}"
        print(msg)
        self.logs.append(msg)

    async def run(self):
        self.log("run: Coordinator loop started")
        while True:
            msg = await self.in_queue.get()
            await self.handle(msg)

    async def start_flow(self, file_paths, query):
        self.log("start_flow: Flow started")
        trace = new_trace_id()

        # Print query for debugging
        print(f"[DEBUG] User query: {query}")

        # Send files to ingestion
        ingest_msg = MCPMessage(
            trace_id=trace,
            type="INGEST",
            sender=self.name,
            receiver="IngestionAgent",
            timestamp=datetime.utcnow(),
            payload={"files": file_paths}
        )
        await self.ingestion.handle(ingest_msg)

        # Save query for LLM later
        self.state["query"] = query

    async def handle(self, msg: MCPMessage):
        self.log(f"handle: {msg.type} from {msg.sender}")

        if msg.type == "INGESTION_ACK":
            docs = msg.payload.get("files", [])
            self.state["docs"] = docs
            self.log(f"handle: Got {len(docs)} docs from ingestion")

            # Print first document text length for debugging
            if docs:
                first_doc_text = docs[0].get("text", "")
                print(f"[DEBUG] First doc length: {len(first_doc_text)}")
                print(f"[DEBUG] First 200 chars:\n{first_doc_text[:200]}")

            # Send to retrieval with query
            retrieval_msg = MCPMessage(
                trace_id=msg.trace_id,
                type="RETRIEVE",
                sender=self.name,
                receiver="RetrievalAgent",
                timestamp=datetime.utcnow(),
                payload={"docs": docs, "query": self.state["query"]}
            )
            await self.retrieval.handle(retrieval_msg)

        elif msg.type == "PREVIEW_RESPONSE":
            preview = msg.payload.get("preview", "")
            self.state["preview"] = preview
            print("\n>>> PREVIEW RESPONSE <<<")
            print(preview[:200] + "..." if len(preview) > 200 else preview)

        elif msg.type == "CONTEXT_RESPONSE":
            passages = msg.payload.get("context") or ""
            self.state["retrieved"] = passages
            self.log("handle: Got retrieval context")

            # Print context and query before sending to LLM
            print(f"[DEBUG] Sending to LLM, context length: {len(passages)}")
            print(f"[DEBUG] Context first 200 chars:\n{passages[:200]}")
            print(f"[DEBUG] Query: {self.state.get('query')}")

            # Send both query and context to LLM
            llm_msg = MCPMessage(
                trace_id=msg.trace_id,
                type="LLM_REQUEST",
                sender=self.name,
                receiver="LLMResponseAgent",
                timestamp=datetime.utcnow(),
                payload={
                    "query": self.state.get("query"),
                    "context": passages
                }
            )
            await self.llm.handle(llm_msg)

        elif msg.type == "LLM_RESPONSE":
            answer = msg.payload.get("answer", "(No answer received)")
            self.state["answer"] = answer
            self.log("handle: Got final LLM answer")
            print(f"\n>>> FINAL ANSWER: {answer}\n")

        else:
            self.log(f"handle: Unknown message type {msg.type}")
