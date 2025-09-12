from datetime import datetime

class MCPMessage:
    def __init__(self, trace_id, type, sender, receiver, timestamp, payload):
        self.trace_id = trace_id
        self.type = type
        self.sender = sender
        self.receiver = receiver
        self.timestamp = timestamp
        self.payload = payload

def new_trace_id():
    import uuid
    return str(uuid.uuid4())
