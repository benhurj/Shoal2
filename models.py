# models.py
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ToolName(str, Enum):
    CALCULATOR = "calculator"
    SEARCH = "search"
    DATETIME = "datetime"


class ToolCall(BaseModel):
    """Parsed from the LLM's Action + Action Input output."""
    tool: ToolName
    input: str = Field(description="The input argument to the tool")


class ToolResult(BaseModel):
    """Returned by tool execution."""
    tool: ToolName
    input: str
    output: str
    success: bool = True
    error: Optional[str] = None


class AgentStep(BaseModel):
    """One iteration of the ReAct loop."""
    iteration: int
    thought: str
    action: Optional[ToolCall] = None
    observation: Optional[str] = None
    is_final: bool = False
    final_answer: Optional[str] = None


class AgentResponse(BaseModel):
    """Full agent response returned to the user."""
    query: str
    answer: str
    steps: list[AgentStep]
    total_tokens: int = 0
    iterations: int = 0
    success: bool = True
    error: Optional[str] = None