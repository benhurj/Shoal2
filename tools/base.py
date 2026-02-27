# tools/base.py
from abc import ABC, abstractmethod
from models import ToolName


class BaseTool(ABC):
    name: ToolName
    description: str
    usage_example: str

    @abstractmethod
    def execute(self, input_str: str) -> str:
        """Execute the tool and return a string observation."""
        pass