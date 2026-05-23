## Summary
<!-- One or two sentences. What changes and why. -->

## Type of change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change (API contract, schema, or response-header change)
- [ ] Storage migration (bumped `CURRENT_SCHEMA_VERSION`)
- [ ] Docs / chore

## Operational impact
<!-- If this touches the router, circuit breaker, budget enforcer, or any
     metric, describe the impact:
       - New / renamed Prometheus metrics?
       - New label dimensions?
       - Change to the X-Gateway-* response headers? -->

## Checklist
- [ ] `uv run ruff check .` passes
- [ ] `uv run ruff format --check .` passes
- [ ] `uv run pyright` passes (0 errors, 0 warnings)
- [ ] `uv run pytest` passes (coverage ≥ 85%)
- [ ] No hardcoded model IDs, API keys, or provider URLs
- [ ] No new dependencies without a justification
- [ ] If schema changed: migration is forward-only and idempotent
