import pytest
from unittest.mock import AsyncMock, patch
from agent_hybrid import run_evaluator_full, run_evaluator_unified
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
# run_evaluator_unified tests
# ---------------------------------------------------------------------------

def _make_candidates(n):
    return [{"history": f"answer {i}", "role_label": f"role_{i}"} for i in range(n)]


@pytest.mark.asyncio
async def test_evaluator_unified_all_pass():
    with patch("agent_hybrid.llm_client") as mock:
        mock.call_llm = AsyncMock(return_value=(
            "VERDICT 0: PASS - correct\nVERDICT 1: PASS - correct\nVERDICT 2: PASS - correct", 80
        ))
        verdicts, tokens = await run_evaluator_unified("query", _make_candidates(3))
        assert all(v["passed"] for v in verdicts)
        assert tokens == 80


@pytest.mark.asyncio
async def test_evaluator_unified_mixed():
    with patch("agent_hybrid.llm_client") as mock:
        mock.call_llm = AsyncMock(return_value=(
            "VERDICT 0: PASS - good\nVERDICT 1: FAIL - wrong\nVERDICT 2: PASS - good", 80
        ))
        verdicts, tokens = await run_evaluator_unified("query", _make_candidates(3))
        assert verdicts[0]["passed"] is True
        assert verdicts[1]["passed"] is False
        assert verdicts[2]["passed"] is True


@pytest.mark.asyncio
async def test_evaluator_unified_unparseable_defaults_fail():
    with patch("agent_hybrid.llm_client") as mock:
        mock.call_llm = AsyncMock(return_value=("Looks okay to me.", 40))
        verdicts, tokens = await run_evaluator_unified("query", _make_candidates(2))
        assert all(not v["passed"] for v in verdicts)


@pytest.mark.asyncio
async def test_evaluator_unified_reason_captured():
    with patch("agent_hybrid.llm_client") as mock:
        mock.call_llm = AsyncMock(return_value=("VERDICT 0: FAIL - missing units", 40))
        verdicts, _ = await run_evaluator_unified("query", _make_candidates(1))
        assert "missing units" in verdicts[0]["reason"]


@pytest.mark.asyncio
async def test_evaluator_unified_returns_one_verdict_per_candidate():
    with patch("agent_hybrid.llm_client") as mock:
        mock.call_llm = AsyncMock(return_value=("VERDICT 0: PASS", 40))
        verdicts, _ = await run_evaluator_unified("query", _make_candidates(4))
        assert len(verdicts) == 4
