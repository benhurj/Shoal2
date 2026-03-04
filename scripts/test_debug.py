# test_debug.py
import asyncio
from agent_hybrid import run_full_agent_loop

async def test_debug():
    try:
        result = await run_full_agent_loop(
            query="what is 2+2?",
            label="test_agent",
            temperature=0.5,
            top_p=0.9
        )
        print(f"Success: {result.final_answer}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_debug())
