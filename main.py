# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from agent import run_agent
from models import AgentResponse

app = FastAPI(title="ReAct Agent Prototype")


class QueryRequest(BaseModel):
    query: str


@app.post("/agent", response_model=AgentResponse)
def agent_endpoint(req: QueryRequest):
    return run_agent(req.query)


@app.get("/health")
def health():
    return {"status": "ok"}