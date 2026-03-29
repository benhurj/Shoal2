# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent_hybrid import run_agent, run_agent_many_samples
from models import AgentResponse

app = FastAPI(title="Shoal — Multi-Agent Ensemble")

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
    """K branched plans → K parallel executor loops → compile."""
    return await run_agent(req.query)


@app.post("/agent/many-samples", response_model=AgentResponse)
async def agent_many_samples_endpoint(req: QueryRequest):
    """K×N 135M executor samples → 9B best-of-N filter → 9B compile."""
    return await run_agent_many_samples(req.query)


@app.get("/health")
def health():
    return {"status": "ok"}