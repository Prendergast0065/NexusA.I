from openai import OpenAI


def call_hosted_prompt(
    variables: dict,
    *,
    api_key: str,
    model: str,
    prompt_id: str,
    prompt_version: str = "1",
    temperature: float = 0.3,
) -> str:
    """Execute a hosted prompt and return the output text."""
    client = OpenAI(api_key=api_key)
    resp = client.responses.create(
        prompt={"id": prompt_id, "version": prompt_version, "variables": variables},
        model=model,
        temperature=temperature,
    )
    return resp.output_text
