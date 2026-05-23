"""Validates the pre-provisioned Grafana dashboards.

We aren't running Grafana in the test suite, but we can catch the most common
mistakes (malformed JSON, missing UID, panel-less file, wrong datasource UID)
in CI so a broken dashboard never lands on ``main``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

DASHBOARDS_DIR = Path(__file__).resolve().parent.parent.parent / "grafana" / "dashboards"

REQUIRED_TOP_LEVEL_KEYS = {"uid", "title", "panels"}
EXPECTED_DATASOURCE_UID = "prometheus"


def _dashboard_files() -> list[Path]:
    return sorted(p for p in DASHBOARDS_DIR.glob("*.json"))


def test_exactly_three_dashboards() -> None:
    files = _dashboard_files()
    assert len(files) == 3, f"expected 3 dashboards, found {[f.name for f in files]}"


@pytest.mark.parametrize("path", _dashboard_files(), ids=lambda p: p.name)
def test_dashboard_parses_as_json_with_required_keys(path: Path) -> None:
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    obj = cast(dict[str, Any], raw)
    missing = REQUIRED_TOP_LEVEL_KEYS - obj.keys()
    assert not missing, f"{path.name} missing keys: {missing}"
    assert isinstance(obj["uid"], str)
    assert obj["uid"]
    assert isinstance(obj["title"], str)
    assert obj["title"]
    panels = obj["panels"]
    assert isinstance(panels, list)
    assert panels, f"{path.name} has no panels"


@pytest.mark.parametrize("path", _dashboard_files(), ids=lambda p: p.name)
def test_every_panel_targets_the_provisioned_datasource(path: Path) -> None:
    obj = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    panels = cast(list[Any], obj["panels"])
    for panel in panels:
        assert isinstance(panel, dict)
        panel_obj = cast(dict[str, Any], panel)
        ds = panel_obj.get("datasource")
        assert isinstance(ds, dict), f"{path.name} panel {panel_obj.get('id')} has no datasource"
        ds_obj = cast(dict[str, Any], ds)
        assert ds_obj.get("uid") == EXPECTED_DATASOURCE_UID, (
            f"{path.name} panel {panel_obj.get('id')} datasource uid is "
            f"{ds_obj.get('uid')!r}, expected {EXPECTED_DATASOURCE_UID!r}"
        )


@pytest.mark.parametrize("path", _dashboard_files(), ids=lambda p: p.name)
def test_panel_queries_reference_gateway_metrics(path: Path) -> None:
    """Every panel target should query a ``llm_gateway_*`` metric."""
    obj = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    panels = cast(list[Any], obj["panels"])
    for panel in panels:
        panel_obj = cast(dict[str, Any], panel)
        targets = panel_obj.get("targets")
        assert isinstance(targets, list), f"{path.name} panel {panel_obj.get('id')} no targets"
        for target in cast(list[Any], targets):
            target_obj = cast(dict[str, Any], target)
            expr = target_obj.get("expr")
            assert isinstance(expr, str), f"target on panel {panel_obj.get('id')} missing expr"
            assert "llm_gateway_" in expr, (
                f"{path.name} panel {panel_obj.get('id')} expr does not reference "
                f"a gateway metric: {expr}"
            )
