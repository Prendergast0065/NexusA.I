def test_hosted_prompt_json_mode():
    import os
    import json
    from openai_utils import call_hosted_prompt

    api_key = os.getenv("OPENAI_API_KEY")
    prompt_id = os.getenv("HOSTED_PROMPT_ID")
    if not api_key or api_key.startswith("your_") or not prompt_id:
        import pytest

        pytest.skip("OPENAI_API_KEY and HOSTED_PROMPT_ID not set")
    output = call_hosted_prompt(
        api_key=api_key,
        model="gpt-4o",
        prompt_id=prompt_id,
        variables={
            "strategy_prompt": "ping",
            "data_block": "dummy",
            "user_message": "",
        },
    )
    json.loads(output)
