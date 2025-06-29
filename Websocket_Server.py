# Install: pip install fastapi[all]
from fastapi import FastAPI, WebSocket
import json

app = FastAPI()
connected_clients = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            print("Received:", data)
            # Save or process the JSON data
            log_entry = json.loads(data)
            with open("network_logs.jsonl", "a") as f:
                f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print("Connection closed:", e)
        connected_clients.remove(websocket)
