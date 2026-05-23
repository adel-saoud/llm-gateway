"""Test-only sys.path bridge for src-layout imports.

Python 3.11.15+ silently skips ``.pth`` files whose basename starts with an
underscore (a recent security backport), which is exactly the file name
hatchling's editable install produces (``_editable_impl_llm_gateway.pth``).
Until that interaction is resolved upstream, this conftest manually puts
``src/`` on ``sys.path`` so the package imports cleanly under ``pytest``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
