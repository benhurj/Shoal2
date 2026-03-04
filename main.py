# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent_hybrid import run_agent
from models import AgentResponse

app = FastAPI(title="ReAct Agent Prototype")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


@app.post("/agent", response_model=AgentResponse)
async def agent_endpoint(req: QueryRequest):
    return await run_agent(req.query)


@app.get("/health")
def health():
    return {"status": "ok"}