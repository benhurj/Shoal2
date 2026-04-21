import pytest
from unittest.mock import AsyncMock, patch


@pytest.fixture
def mock_llm_client():
    """Patch llm_client.call_llm to return canned responses."""
    with patch("llm_client.llm_client") as mock_client:
        mock_client.call_llm = AsyncMock(return_value=("mocked response", 100))
        yield mock_client
