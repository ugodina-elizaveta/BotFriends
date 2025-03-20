from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from src.inference import get_response

app = FastAPI()

class ChatRequest(BaseModel):
    text: str

@app.post("/chat")
async def chat(request: ChatRequest):
    response = get_response(request.text)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)