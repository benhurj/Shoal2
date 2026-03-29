import pytest
from unittest.mock import AsyncMock, patch
from agent_hybrid import run_evaluator_full, run_evaluator_best_of_n
from models import EvaluationResult


@pytest.mark.asyncio
async def test_evaluator_pass():
    with patch("agent_hybrid.llm_client") as mock:
        mock.call_llm = AsyncMock(return_value=("PASS", 50))
        result, tokens = await run_evaluator_full("calculate 2+2", "4")
        assert result.passed is True
        assert tokens == 50


@pytest.mark.asyncio
async def test_evaluator_fail_with_reason():
    with patch("agent_hybrid.llm_client") as mock:
        mock.call_llm = AsyncMock(return_value=("FAIL: wrong answer", 50))
        result, tokens = await run_evaluator_full("calculate 2+2", "5")
        assert result.passed is False
        assert "wrong answer" in result.feedback


@pytest.mark.asyncio
async def test_evaluator_unparseable_defaults_to_fail():
    with patch("agent_hybrid.llm_client") as mock:
        mock.call_llm = AsyncMock(return_value=("The output looks reasonable I think", 50))
        result, tokens = await run_evaluator_full("calculate 2+2", "4")
        assert result.passed is False
        assert "unparseable" in result.feedback


@pytest.mark.asyncio
async def test_evaluator_pass_case_insensitive():
    with patch("agent_hybrid.llm_client") as mock:
        mock.call_llm = AsyncMock(return_value=("pass", 50))
        result, tokens = await run_evaluator_full("task", "output")
        assert result.passed is True


@pytest.mark.asyncio
async def test_evaluator_fail_case_insensitive():
    with patch("agent_hybrid.llm_client") as mock:
        mock.call_llm = AsyncMock(return_value=("Fail: bad", 50))
        result, tokens = await run_evaluator_full("task", "output")
        assert result.passed is False


# ---------------------------------------------------------------------------
# run_evaluator_best_of_n tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_evaluator_best_of_n_selects_correct():
    with patch("agent_hybrid.llm_client") as mock:
        mock.call_llm = AsyncMock(return_value=("BEST: 2\nQUALITY: GOOD\nREASON: Most complete.", 50))
        idx, quality, tokens = await run_evaluator_best_of_n(
            "test query", "analyst", ["ans 0", "ans 1", "ans 2"]
        )
        assert idx == 2
        assert quality == "GOOD"
        assert tokens == 50


@pytest.mark.asyncio
async def test_evaluator_best_of_n_parse_failure_defaults_zero():
    with patch("agent_hybrid.llm_client") as mock:
        mock.call_llm = AsyncMock(return_value=("The second one looks good.", 50))
        idx, quality, tokens = await run_evaluator_best_of_n(
            "test query", "analyst", ["ans 0", "ans 1", "ans 2"]
        )
        assert idx == 0
        assert quality == "FAIR"


@pytest.mark.asyncio
async def test_evaluator_best_of_n_out_of_range_defaults_zero():
    with patch("agent_hybrid.llm_client") as mock:
        mock.call_llm = AsyncMock(return_value=("BEST: 99\nQUALITY: GOOD\nREASON: x.", 50))
        idx, quality, tokens = await run_evaluator_best_of_n(
            "test query", "analyst", ["ans 0", "ans 1"]
        )
        assert idx == 0


@pytest.mark.asyncio
async def test_evaluator_best_of_n_poor_quality():
    with patch("agent_hybrid.llm_client") as mock:
        mock.call_llm = AsyncMock(return_value=("BEST: 0\nQUALITY: POOR\nREASON: weak.", 50))
        idx, quality, tokens = await run_evaluator_best_of_n(
            "query", "role", ["only one answer"]
        )
        assert quality == "POOR"


@pytest.mark.asyncio
async def test_evaluator_best_of_n_case_insensitive_quality():
    with patch("agent_hybrid.llm_client") as mock:
        mock.call_llm = AsyncMock(return_value=("BEST: 1\nQUALITY: fair\nREASON: ok.", 50))
        idx, quality, tokens = await run_evaluator_best_of_n(
            "query", "role", ["ans 0", "ans 1"]
        )
        assert idx == 1
        assert quality == "FAIR"
