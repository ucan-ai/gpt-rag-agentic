import asyncio
from orchestration.orchestrator import run_conversation_turn

async def test():
    result = await run_conversation_turn(
        strategy='FINE_TUNED_SLM_MULTIMODAL',
        query='lockpin number cargo ramp support equipment TO 1300i-2-32JG-30-1',
        conversation_id='test-multimodal-123',
        user_id='test-user',
        output_format='TEXT',
        temperature=0.7
    )
    print('Result type:', type(result))
    if hasattr(result, 'content'):
        print('Content type:', type(result.content))
        if hasattr(result.content, '__len__'):
            print('Content length:', len(result.content))
            for i, item in enumerate(result.content):
                print(f'Item {i}: {type(item)}')
    return result

if __name__ == "__main__":
    asyncio.run(test()) 