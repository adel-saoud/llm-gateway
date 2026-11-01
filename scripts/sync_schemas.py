"""Keep the pinned copies of cross-repo contracts identical to their producer's.

The gateway consumes one contract: llm-cost-autopilot's route API schema. The
canonical file lives in the producer repo; ``tests/contracts/`` holds a copy
pinned to a producer tag, and the contract tests read that copy.

    uv run python scripts/sync_schemas.py            # rewrite the copies from the producer
    uv run python scripts/sync_schemas.py --check    # exit 1 if a copy differs

Where the canonical file is read from (``--source``):

* ``url``: the raw GitHub file at the pinned tag. CI uses this, so a merge
  always matches a published producer version.
* ``sibling``: ``git show <tag>:<path>`` in the producer checked out next to
  this repo. Exact and offline.
* ``auto`` (default, used by pre-commit): ``sibling`` when that checkout has the
  tag; otherwise the sibling's working-tree file, with a warning, because the
  producer may not have tagged the version being developed against yet; and
  ``url`` when there is no sibling checkout at all.

Bumping a contract: change ``tag`` below, run without ``--check``, and add a
CHANGELOG entry (the producer adds one too).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

REPO_ROOT: Final = Path(__file__).resolve().parent.parent

Source = Literal["auto", "sibling", "url"]


@dataclass(frozen=True, slots=True)
class Contract:
    """One consumed schema: where its canonical file lives and where the copy goes."""

    producer: str
    """Repository name under github.com/adel-saoud, also the sibling folder name."""
    tag: str
    """Producer release the copy is pinned to."""
    producer_path: str
    """Path of the canonical file inside the producer repo."""
    local_copy: Path
    """Pinned copy in this repo, read by the contract tests."""

    @property
    def raw_url(self) -> str:
        return (
            f"https://raw.githubusercontent.com/adel-saoud/{self.producer}/"
            f"{self.tag}/{self.producer_path}"
        )

    @property
    def sibling(self) -> Path:
        return REPO_ROOT.parent / self.producer


CONTRACTS: Final = (
    Contract(
        producer="llm-cost-autopilot",
        tag="v0.2.0",
        producer_path="docs/route-api.schema.json",
        local_copy=REPO_ROOT / "tests" / "contracts" / "autopilot-route-api.schema.json",
    ),
)


class CanonicalUnavailableError(RuntimeError):
    """The canonical file could not be read from the requested source."""


def _from_url(contract: Contract) -> bytes:
    try:
        with urllib.request.urlopen(contract.raw_url, timeout=15) as response:  # noqa: S310 (fixed https URL)
            return response.read()
    except OSError as exc:
        raise CanonicalUnavailableError(f"{contract.raw_url}: {exc}") from exc


def _git(contract: Contract, *args: str) -> subprocess.CompletedProcess[bytes]:
    """Run git in the producer's sibling checkout; fixed arguments, no shell."""
    command = ["git", "-C", str(contract.sibling), *args]
    return subprocess.run(command, capture_output=True, check=False)  # noqa: S603


def _sibling_has_tag(contract: Contract) -> bool:
    return (
        _git(contract, "rev-parse", "--verify", "--quiet", f"refs/tags/{contract.tag}").returncode
        == 0
    )


def _from_sibling_tag(contract: Contract) -> bytes:
    result = _git(contract, "show", f"{contract.tag}:{contract.producer_path}")
    if result.returncode != 0:
        raise CanonicalUnavailableError(
            f"{contract.sibling} at {contract.tag}: {result.stderr.decode().strip()}"
        )
    return result.stdout


def canonical(contract: Contract, source: Source) -> bytes:
    """Bytes of the producer's file for ``contract``, read from ``source``."""
    if source == "url":
        return _from_url(contract)
    if source == "sibling":
        return _from_sibling_tag(contract)
    if not contract.sibling.is_dir():
        return _from_url(contract)
    if _sibling_has_tag(contract):
        return _from_sibling_tag(contract)
    working_copy = contract.sibling / contract.producer_path
    print(
        f"warning: {contract.producer} has no tag {contract.tag}; "
        f"comparing with its working tree ({working_copy})",
        file=sys.stderr,
    )
    try:
        return working_copy.read_bytes()
    except OSError as exc:
        raise CanonicalUnavailableError(f"{working_copy}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check or sync pinned contract copies.")
    parser.add_argument("--check", action="store_true", help="fail on a difference, write nothing")
    parser.add_argument("--source", choices=["auto", "sibling", "url"], default="auto")
    args = parser.parse_args(argv)

    failed = False
    for contract in CONTRACTS:
        name = contract.local_copy.relative_to(REPO_ROOT)
        try:
            expected = canonical(contract, args.source)
        except CanonicalUnavailableError as exc:
            print(f"error: {name}: canonical file unavailable: {exc}", file=sys.stderr)
            failed = True
            continue
        current = contract.local_copy.read_bytes() if contract.local_copy.exists() else b""
        if current == expected:
            print(f"ok: {name} matches {contract.producer} {contract.tag}")
        elif args.check:
            print(
                f"error: {name} differs from {contract.producer} {contract.tag}; "
                "run: uv run python scripts/sync_schemas.py",
                file=sys.stderr,
            )
            failed = True
        else:
            contract.local_copy.write_bytes(expected)
            print(f"updated: {name} from {contract.producer} {contract.tag}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
