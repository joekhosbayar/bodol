"""Environment and filesystem paths. Plumbing only — no policy lives here."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PROVIDER = os.getenv("BODOL_PROVIDER", "gemini:gemini-2.0-flash")

TRACE_DIR = Path(os.getenv("BODOL_TRACE_DIR", ROOT / "traces"))
SCRATCHPAD_DIR = Path(os.getenv("BODOL_SCRATCHPAD_DIR", ROOT / "scratchpad"))
PROMPT_DIR = Path(os.getenv("BODOL_PROMPT_DIR", ROOT / "prompts"))
PRICING_FILE = Path(__file__).parent / "data" / "pricing.yaml"


class MissingCredential(RuntimeError):
    pass


def api_key(provider_family: str) -> str:
    """Look up the API key for a provider family, e.g. "gemini"."""
    env_names = {
        "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "openai": ("OPENAI_API_KEY",),
        "anthropic": ("ANTHROPIC_API_KEY",),
    }.get(provider_family, (f"{provider_family.upper()}_API_KEY",))

    for name in env_names:
        if value := os.getenv(name):
            return value

    raise MissingCredential(
        f"No API key for {provider_family!r}. Set one of: {', '.join(env_names)} "
        f"(see .env.example)."
    )


def ensure_dirs() -> None:
    for d in (TRACE_DIR, SCRATCHPAD_DIR):
        d.mkdir(parents=True, exist_ok=True)
