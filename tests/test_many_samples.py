import pytest
from unittest.mock import AsyncMock, patch
from agent_hybrid import run_agent_many_samples
from models import AgentResponse
import config


def _make_roles(k):
    return [
        {"label": f"role_{i}", "executor_addendum": "", "temperature": 0.1, "top_p": 0.8}
        for i in range(k)
    ]


@pytest.mark.asyncio
async def test_run_agent_many_samples_returns_agent_response():
    """run_agent_many_samples returns a valid AgentResponse with expected fields."""
    base_plan = [{"task": "research topic", "tool_hint": "search"}]
    roles = _make_roles(config.ENSEMBLE_K)
    plans = [base_plan] * config.ENSEMBLE_K

    with patch("agent_hybrid.run_planner_and_roles", new=AsyncMock(return_value=(base_plan, roles, 100))), \
         patch("agent_hybrid.run_plan_adjuster", new=AsyncMock(return_value=(plans, 80))), \
         patch("agent_hybrid.run_agent_executor_simple", new=AsyncMock(return_value=("history text", [], 30))), \
         patch("agent_hybrid.run_evaluator_best_of_n", new=AsyncMock(return_value=(0, "GOOD", 40))), \
         patch("agent_hybrid.run_compiler", new=AsyncMock(return_value=("The final answer.", 60))):

        result = await run_agent_many_samples("test query")

    assert isinstance(result, AgentResponse)
    assert result.answer == "The final answer."
    assert result.total_tokens > 0
    assert len(result.sub_task_results) == 1
    assert result.success is True


@pytest.mark.asyncio
async def test_run_agent_many_samples_executor_and_eval_call_counts():
    """K×N executor instances are launched and K evaluators are called."""
    base_plan = [{"task": "do something", "tool_hint": "none"}]
    roles = _make_roles(config.ENSEMBLE_K)
    plans = [base_plan] * config.ENSEMBLE_K

    mock_executor = AsyncMock(return_value=("history", [], 10))
    mock_eval = AsyncMock(return_value=(0, "GOOD", 20))

    with patch("agent_hybrid.run_planner_and_roles", new=AsyncMock(return_value=(base_plan, roles, 50))), \
         patch("agent_hybrid.run_plan_adjuster", new=AsyncMock(return_value=(plans, 40))), \
         patch("agent_hybrid.run_agent_executor_simple", new=mock_executor), \
         patch("agent_hybrid.run_evaluator_best_of_n", new=mock_eval), \
         patch("agent_hybrid.run_compiler", new=AsyncMock(return_value=("answer", 30))):

        await run_agent_many_samples("test query")

    assert mock_executor.call_count == config.ENSEMBLE_K * config.SAMPLES_PER_ROLE
    assert mock_eval.call_count == config.ENSEMBLE_K


@pytest.mark.asyncio
async def test_run_agent_many_samples_planner_failure_fallback():
    """Falls back gracefully when planner raises."""
    base_plan = [{"task": "test", "tool_hint": "none"}]
    plans = [base_plan] * config.ENSEMBLE_K

    with patch("agent_hybrid.run_planner_and_roles", side_effect=Exception("planner error")), \
         patch("agent_hybrid.run_plan_adjuster", new=AsyncMock(return_value=(plans, 40))), \
         patch("agent_hybrid.run_agent_executor_simple", new=AsyncMock(return_value=("history", [], 10))), \
         patch("agent_hybrid.run_evaluator_best_of_n", new=AsyncMock(return_value=(0, "GOOD", 20))), \
         patch("agent_hybrid.run_compiler", new=AsyncMock(return_value=("answer", 30))):

        result = await run_agent_many_samples("test query")

    assert isinstance(result, AgentResponse)
    assert result.success is True


@pytest.mark.asyncio
async def test_run_agent_many_samples_poor_quality_marked_fail():
    """Roles with POOR quality get passed=False in ensemble results."""
    base_plan = [{"task": "task", "tool_hint": "none"}]
    roles = _make_roles(config.ENSEMBLE_K)
    plans = [base_plan] * config.ENSEMBLE_K

    with patch("agent_hybrid.run_planner_and_roles", new=AsyncMock(return_value=(base_plan, roles, 50))), \
         patch("agent_hybrid.run_plan_adjuster", new=AsyncMock(return_value=(plans, 40))), \
         patch("agent_hybrid.run_agent_executor_simple", new=AsyncMock(return_value=("history", [], 10))), \
         patch("agent_hybrid.run_evaluator_best_of_n", new=AsyncMock(return_value=(0, "POOR", 20))), \
         patch("agent_hybrid.run_compiler", new=AsyncMock(return_value=("answer", 30))):

        result = await run_agent_many_samples("test query")

    sub = result.sub_task_results[0]
    assert all(r.evaluation and not r.evaluation.passed for r in sub.ensemble_results)
