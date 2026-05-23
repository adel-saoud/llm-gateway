# ── Detect compose command ────────────────────────────────────────────────────
# Prefer the "docker compose" plugin (v2); fall back to standalone docker-compose.
COMPOSE_CMD := $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")

# ── Helpers ───────────────────────────────────────────────────────────────────
OVERRIDE_FILE  := docker-compose.override.yml
OVERRIDE_TMPL  := docker-compose.override.yml.example
TEAMS_FILE     := config/teams.yaml
TEAMS_TMPL     := config/teams.example.yaml

.DEFAULT_GOAL := help

.PHONY: help up down reset logs test check demo load-test demo-resilience demo-full setup

help:           ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS=":.*##"}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# Create local config files from their examples if they don't exist yet.
setup:
	@if [ ! -f $(OVERRIDE_FILE) ]; then \
		cp $(OVERRIDE_TMPL) $(OVERRIDE_FILE); \
		echo "Created $(OVERRIDE_FILE) from example — routes to your host Ollama."; \
	fi
	@if [ ! -f $(TEAMS_FILE) ]; then \
		cp $(TEAMS_TMPL) $(TEAMS_FILE); \
		echo "Created $(TEAMS_FILE) from example — edit to add your teams."; \
	fi

up: setup       ## Build and start all services (host-Ollama override applied)
	$(COMPOSE_CMD) up -d --build

down:           ## Stop and remove containers (volumes are preserved)
	$(COMPOSE_CMD) down

reset:          ## Tear down everything including volumes, then rebuild from scratch
	$(COMPOSE_CMD) down -v --remove-orphans
	$(COMPOSE_CMD) up -d --build

logs:           ## Stream gateway logs (Ctrl-C to exit)
	$(COMPOSE_CMD) logs -f gateway

test:           ## Run the full test suite (hermetic — no Docker needed)
	uv run pytest --tb=short -q

check:          ## Run all quality gates: lint, format, type-check, tests
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright
	uv run pytest --tb=short -q

demo:           ## Run the 5-step RAG demo against the running gateway
	uv run python scripts/demo_rag.py

load-test:      ## Run the load test (1 000 requests) against the running gateway
	uv run python scripts/load_test.py

demo-resilience: ## Trip the circuit breaker and show the OPEN→HALF_OPEN→CLOSED cycle
	uv run python scripts/demo_resilience.py

demo-full:      ## Run all demos in sequence: RAG → load test → resilience
	uv run python scripts/demo_rag.py
	uv run python scripts/load_test.py --requests 200
	uv run python scripts/demo_resilience.py
