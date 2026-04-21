# tools/datetime_tool.py
from datetime import datetime, timezone
from tools.base import BaseTool
from models import ToolName


class DateTimeTool(BaseTool):
    name = ToolName.DATETIME
    description = "Returns the current date and time in UTC. No input needed."
    usage_example = "datetime: now"

    async def execute(self, input_str: str) -> str:
        now = datetime.now(timezone.utc)
        return now.strftime("%Y-%m-%d %H:%M:%S UTC")