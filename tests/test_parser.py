import pytest
from agent_hybrid import parse_llm_output


class TestParseAction:
    def test_action_with_thought(self):
        text = "Thought: I need to calculate\nAction: calculator\nAction Input: 2+2"
        step = parse_llm_output(text)
        assert step.thought == "I need to calculate"
        assert step.action is not None
        assert step.action.tool.value == "calculator"
        assert step.action.input == "2+2"
        assert not step.is_final

    def test_action_without_thought(self):
        text = "Action: search\nAction Input: weather in London"
        step = parse_llm_output(text)
        assert step.action is not None
        assert step.action.tool.value == "search"
        assert step.action.input == "weather in London"

    def test_invalid_tool_name(self):
        text = "Thought: let me try\nAction: nonexistent_tool\nAction Input: something"
        step = parse_llm_output(text)
        # Should fall through to fallback since tool name is invalid
        assert step.action is None


class TestParseFinalAnswer:
    def test_final_answer_with_thought(self):
        text = "Thought: I now know the answer\nFinal Answer: 42"
        step = parse_llm_output(text)
        assert step.is_final
        assert step.final_answer == "42"
        assert step.thought == "I now know the answer"

    def test_final_answer_multiline(self):
        text = "Final Answer: The result is 42.\nIt was calculated using the calculator tool."
        step = parse_llm_output(text)
        assert step.is_final
        assert "42" in step.final_answer


class TestParseFallback:
    def test_unrecognized_format(self):
        text = "I don't know how to respond properly."
        step = parse_llm_output(text)
        assert not step.is_final
        assert step.action is None
        assert step.thought == text

    def test_empty_string(self):
        step = parse_llm_output("")
        assert step.thought == ""
        assert step.action is None
        assert not step.is_final


class TestActionPrecedence:
    def test_action_before_final_answer(self):
        text = "Thought: checking\nAction: calculator\nAction Input: 5*5\nFinal Answer: 25"
        step = parse_llm_output(text)
        # Action should take precedence
        assert step.action is not None
        assert not step.is_final
