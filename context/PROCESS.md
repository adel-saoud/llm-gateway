# Process Context

**Workflow**: Feature branches → PR → squash-merge to `main`. All 4 quality gates must pass before a task is done.

## Quality Assurance

- **Testing approach**: Unit tests for every public function; integration tests via `fastapi.testclient.TestClient` for the API surface. No real network calls — every Ollama call is intercepted with `respx`; every Redis call uses `fakeredis[lua]`; SQLite uses pytest's `tmp_path`.
- **Coverage goals**: ≥ 85% on the whole package (enforced by `pytest --cov`). 85% is the hard floor; current coverage hovers ~92%.
- **Linting strategy**: `ruff check --fix` then `ruff format` — both must exit 0. `pre-commit` enforces this on every commit (`uv run pre-commit install` on first setup). The only project-specific noqa is `S106` on Prometheus label values (`token_type="prompt"` and `token_type="completion"` trip the bandit hardcoded-password rule).
- **Type safety**: `pyright` strict, 0 errors, 0 warnings. Third-party typing gaps (`redis.asyncio.evalsha`, `respx` request/response, `pytest.approx`) are addressed with narrow per-line `# pyright: ignore[<specific-rule>]` comments — never broad `Any` casts.

## Development Workflow

- **Commit rules**: Conventional Commits — `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`. User commits only; agent never commits or pushes.
- **Issue tracking**: GitHub Issues / PR comments.
- **Version control**: Feature branches `feat/<slug>`, fix branches `fix/<slug>`. Squash-merge preferred.
- **Code review**: PR description follows [`.github/PULL_REQUEST_TEMPLATE.md`](../.github/PULL_REQUEST_TEMPLATE.md). CI runs on every PR; merging is blocked until all four gates are green.

## Code Standards & Conventions

- **Code formatting**: `ruff format` (enforced in CI via `--check`).
- **Directory structure**: `src/llm_gateway/` for all importable code. No `utils.py` — helpers live in their domain module (e.g. `_extract_bearer_token` in `auth/middleware.py`, `_expect_dict` in `providers/ollama.py`). No global state — every component is constructed in `create_app` and attached to `app.state` so tests can swap.
- **Documentation**: Module docstrings explain the *why* (design choices, gotchas, non-obvious invariants), not the *what* (the type signatures already do that). Inline comments are reserved for non-obvious branches or surprising third-party behaviour. No comments that restate well-named code.
- **Naming conventions**: Snake case throughout. `structlog.get_logger(__name__)` at module top, bound to `_log`. Event names in `structlog` use `<noun>_<verb>` (e.g. `cost_event`, `circuit_state_transition`, `rate_limit_exceeded`).
- **Error handling**: Three-layer exception hierarchy at the provider boundary (`ProviderError` / `ProviderTimeoutError` / `ProviderUnavailableError`). The router and circuit breaker treat the latter two as transient (record + fall back); the first is a request-shape problem and bubbles. No bare `except Exception` — every catch names the specific exception type.
