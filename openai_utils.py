"""Utilities for working with OpenAI Hosted Prompts (Responses API)."""

from openai import OpenAI


def call_hosted_prompt(
    *,
    api_key: str,
    model: str,
    prompt_id: str,
    variables: dict,
    prompt_version: str = "1",
    temperature: float = 0.3,
    input_message: str = "Please respond in valid json."
) -> str:
    """
    Execute a Hosted Prompt (Output format = JSON object) and return the
    raw JSON string.

    The single `input_message` lands as the user‑role message — it MUST
    contain the word “json” to satisfy the server‑side guard.
    """
    client = OpenAI(api_key=api_key)

    resp = client.responses.create(
        prompt={
            "id": prompt_id,
            "version": prompt_version,
            "variables": variables,
        },
        model=model,
        input=input_message,
        temperature=temperature,
    )
    return resp.output_text
