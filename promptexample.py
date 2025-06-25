"""
CLI utility: collects key market variables, sends them to a *hosted* OpenAI Prompt,
returns a strict JSON with `buy` / `sell` / `hold` signal.

Usage (interactive):
    python signal_cli.py --prompt "Focus on RSI and MACD" 

Or one‑liner:
    python signal_cli.py \
        --price 420.10 --volume 18450000 \
        --sma-10 412.30 --sma-50 398.75 --rsi-14 68.2 \
        --atr-14 6.45 --macd 3.15 \
        --bollinger-upper 422.6 --bollinger-lower 402.0 \
        --prompt "Generate signal"

Environment variables required:
    OPENAI_API_KEY        – your key with Prompt‑Management scope
    HOSTED_PROMPT_ID      – e.g. pmpt_685c013c93a08190906c7cb7cd2273f70aeb15fcd156bf5f
    HOSTED_PROMPT_VERSION – e.g. 1
"""
from __future__ import annotations

import os
import json
import sys
import argparse
from datetime import date

from openai import OpenAI, Error

# ----- Helpers -------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate buy/sell signal via OpenAI hosted prompt")

    # market variables
    p.add_argument("--price", type=float, required=False)
    p.add_argument("--volume", type=float, required=False)
    p.add_argument("--sma-10", dest="sma_10", type=float, required=False)
    p.add_argument("--sma-50", dest="sma_50", type=float, required=False)
    p.add_argument("--rsi-14", dest="rsi_14", type=float, required=False)
    p.add_argument("--atr-14", dest="atr_14", type=float, required=False)
    p.add_argument("--macd", type=float, required=False)
    p.add_argument("--bollinger-upper", dest="bollinger_upper", type=float, required=False)
    p.add_argument("--bollinger-lower", dest="bollinger_lower", type=float, required=False)

    p.add_argument("--prompt", required=True, help="Natural‑language instruction for the model")
    p.add_argument("--model", default="gpt-4o", help="Override model if dashboard allows")
    p.add_argument("--temperature", type=float, default=0.3)
    return p.parse_args()


def collect_interactively(ns: argparse.Namespace):
    """Ask for missing values via input()"""
    fields = [
        "price", "volume", "sma_10", "sma_50", "rsi_14",
        "atr_14", "macd", "bollinger_upper", "bollinger_lower",
    ]
    for f in fields:
        if getattr(ns, f) is None:
            raw = input(f"{f} ⟹ ")
            try:
                setattr(ns, f, float(raw))
            except ValueError:
                print(f"Invalid float for {f}; exiting.")
                sys.exit(1)


def build_variables(ns: argparse.Namespace) -> dict[str, str | float]:
    return {
        "user_prompt": ns.prompt,
        "template_block": "",                 # optional; kept for compatibility
        "csv_block": "",                      # optional

        "ts": str(date.today()),
        "price": ns.price,
        "volume": ns.volume,
        "sma_10": ns.sma_10,
        "sma_50": ns.sma_50,
        "rsi_14": ns.rsi_14,
        "atr_14": ns.atr_14,
        "macd": ns.macd,
        "bollinger_upper": ns.bollinger_upper,
        "bollinger_lower": ns.bollinger_lower,
    }


def call_openai(variables: dict, model: str, temperature: float) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt_id = os.getenv("HOSTED_PROMPT_ID")
    prompt_version = os.getenv("HOSTED_PROMPT_VERSION", "1")
    if not all([prompt_id, os.getenv("OPENAI_API_KEY")]):
        raise RuntimeError("OPENAI_API_KEY and HOSTED_PROMPT_ID must be set in env vars")

    try:
        resp = client.responses.create(
            prompt={"id": prompt_id, "version": prompt_version},
            variables=variables,
            model=model,
            response_format={"type": "json_object"},
            temperature=temperature,
        )
    except Error as exc:
        raise RuntimeError(f"OpenAI error: {exc}")

    return resp.choices[0].message.content


# ----- main ---------------------------------------------------------------- #

def main():
    ns = parse_args()
    collect_interactively(ns)

    vars_dict = build_variables(ns)
    print("\nSending to OpenAI…\n")
    raw_json = call_openai(vars_dict, ns.model, ns.temperature)

    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        print("Model did not return valid JSON:\n", raw_json)
        sys.exit(1)

    print(json.dumps(parsed, indent=2))


if __name__ == "__main__":
    main()
