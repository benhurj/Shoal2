import pytest
from unittest.mock import AsyncMock, patch
from agent_hybrid import run_evaluator
from models import EvaluationResult


@pytest.mark.asyncio
async def test_evaluator_pass():
    with patch("agent_hybrid.llm_client") as mock:
        mock.call_llm = AsyncMock(return_value=("PASS", 50))
        result, tokens = await run_evaluator("calculate 2+2", "4")
        assert result.passed is True
        assert tokens == 50


@pytest.mark.asyncio
async def test_evaluator_fail_with_reason():
    with patch("agent_hybrid.llm_client") as mock:
        mock.call_llm = AsyncMock(return_value=("FAIL: wrong answer", 50))
        result, tokens = await run_evaluator("calculate 2+2", "5")
        assert result.passed is False
        assert "wrong answer" in result.feedback


@pytest.mark.asyncio
async def test_evaluator_unparseable_defaults_to_fail():
    with patch("agent_hybrid.llm_client") as mock:
        mock.call_llm = AsyncMock(return_value=("The output looks reasonable I think", 50))
        result, tokens = await run_evaluator("calculate 2+2", "4")
        assert result.passed is False
        assert "unparseable" in result.feedback


@pytest.mark.asyncio
async def test_evaluator_pass_case_insensitive():
    with patch("agent_hybrid.llm_client") as mock:
        mock.call_llm = AsyncMock(return_value=("pass", 50))
        result, tokens = await run_evaluator("task", "output")
        assert result.passed is True


@pytest.mark.asyncio
async def test_evaluator_fail_case_insensitive():
    with patch("agent_hybrid.llm_client") as mock:
        mock.call_llm = AsyncMock(return_value=("Fail: bad", 50))
        result, tokens = await run_evaluator("task", "output")
        assert result.passed is False
