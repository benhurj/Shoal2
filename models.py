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


class EvaluationResult(BaseModel):
    """Evaluator's judgement of an Executor's sub-task output."""
    passed: bool
    feedback: str = ""


class EnsembleMemberResult(BaseModel):
    """Result from one member of the ensemble (one config variation)."""
    config_label: str                        # e.g. "conservative"
    llm_params: dict                         # the actual overrides used
    executor_steps: list[AgentStep] = []
    final_answer: str = ""
    evaluation: Optional[EvaluationResult] = None
    tokens_used: int = 0


class SubTaskResult(BaseModel):
    """Result of one Planner sub-task through the Executor/Evaluator cycle."""
    sub_task: str
    executor_steps: list[AgentStep] = []     # Used in non-ensemble (fast path)
    evaluator_result: Optional[EvaluationResult] = None
    ensemble_results: list[EnsembleMemberResult] = []   # Ensemble members
    compiled_answer: str = ""                # Compiler output
    attempts: int = 1
    passed: bool = False


class AgentResponse(BaseModel):
    """Full agent response returned to the user."""
    query: str
    plan: Optional[list[str]] = None         # None if fast-path was used
    sub_task_results: list[SubTaskResult] = []
    answer: str
    steps: list[AgentStep] = []              # Kept for fast-path compatibility
    total_tokens: int = 0
    iterations: int = 0
    success: bool = True
    error: Optional[str] = None