"""Utilities for working with OpenAI Hosted Prompts."""

from openai import OpenAI


def call_hosted_prompt(
    *,
    api_key: str,
    model: str,
    prompt_id: str,
    variables: dict,
    prompt_version: str = "1",
    temperature: float = 0.3,
    input_message: str = "Please respond in valid json.",
) -> str:
    """Execute a hosted prompt and return the output text.

    The ``input_message`` is sent as a single user message which must contain the
    word ``"json"`` to satisfy the server side JSON-mode guard.
    """

    client = OpenAI(api_key=api_key)
    resp = client.responses.create(
        prompt={"id": prompt_id, "version": prompt_version, "variables": variables},
        model=model,
        input=input_message,
        temperature=temperature,
    )
    return resp.output_text
