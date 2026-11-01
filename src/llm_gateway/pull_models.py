"""Pull the models the gateway is configured to serve into Ollama.

    uv run python -m llm_gateway.pull_models

The model ids come from ``Settings`` (``GATEWAY_MODEL_HIGH_QUALITY``,
``GATEWAY_MODEL_BUDGET``, ``GATEWAY_MODEL_EMBEDDINGS``), the same values the
gateway serves, so changing a model is one setting and never a second list in
docker-compose or a setup script that can drift from it. docker-compose's
``ollama-init`` service runs this module from the gateway image.
"""

from __future__ import annotations

import sys

import httpx

from llm_gateway.config import Settings

# A first pull of a 4B model over a slow link takes minutes; a timeout here
# would only turn a slow download into a failed setup.
_PULL_TIMEOUT_S = 1800.0


def configured_models(settings: Settings) -> tuple[str, ...]:
    """Distinct model ids of the three tiers, in tier order."""
    return tuple(
        dict.fromkeys(
            (settings.model_high_quality, settings.model_budget, settings.model_embeddings)
        )
    )


def main(settings: Settings | None = None) -> int:
    settings = settings or Settings()
    with httpx.Client(base_url=settings.ollama_base_url, timeout=_PULL_TIMEOUT_S) as client:
        for model in configured_models(settings):
            print(f"pulling {model} from {settings.ollama_base_url}", flush=True)
            response = client.post("/api/pull", json={"model": model, "stream": False})
            if response.is_error:
                print(f"error: {model}: {response.status_code} {response.text}", file=sys.stderr)
                return 1
    print("models ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
