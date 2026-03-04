# test_llm.py
import asyncio
from llm_client import llm_client

async def test_llm():
    messages = [{"role": "user", "content": "what is 2+2?"}]
    try:
        result, tokens = await llm_client.call_llm(messages, "smollm2:135m")
        print(f"Success: {result}")
        print(f"Tokens: {tokens}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_llm())
