# test_planner.py
import asyncio
from agent_hybrid import run_planner
from config import COMPILER_TEMPERATURE, COMPILER_TOP_P

async def test_planner():
    try:
        plan, tokens = await run_planner("what is 2+2?", temperature=COMPILER_TEMPERATURE, top_p=COMPILER_TOP_P)
        print(f"Success: {plan}")
        print(f"Tokens: {tokens}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_planner())
