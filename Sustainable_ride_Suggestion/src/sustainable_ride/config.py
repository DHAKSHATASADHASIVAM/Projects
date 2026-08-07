"""Configuration loading.

All tunable parameters live in ``config/*.yaml`` rather than in code, so that
the modelling assumptions -- emission factors especially -- can be reviewed and
challenged without reading Python.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Repository root: src/sustainable_ride/config.py -> up three levels.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


class Config(dict):
    """A dict that also supports dotted lookup: ``cfg.get_path("a.b.c")``."""

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _load_yaml(name: str) -> Config:
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing config file {path}. Expected it alongside the repo's "
            f"config/ directory."
        )
    with path.open("r", encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh) or {})


@lru_cache(maxsize=1)
def load_config() -> Config:
    """Main project configuration (``config/config.yaml``)."""
    return _load_yaml("config.yaml")


@lru_cache(maxsize=1)
def load_emissions_config() -> Config:
    """Emission factors and their citations (``config/emissions.yaml``)."""
    return _load_yaml("emissions.yaml")


@lru_cache(maxsize=1)
def load_pricing_config() -> Config:
    """Fare structures (``config/pricing.yaml``)."""
    return _load_yaml("pricing.yaml")


def resolve_path(key: str, *parts: str) -> Path:
    """Resolve a configured directory (``paths.<key>``) to an absolute path.

    Creates the directory if it does not exist, so callers never have to.
    """
    cfg = load_config()
    rel = cfg.get_path(f"paths.{key}")
    if rel is None:
        raise KeyError(f"No such configured path: paths.{key}")
    base = PROJECT_ROOT / rel
    base.mkdir(parents=True, exist_ok=True)
    return base.joinpath(*parts) if parts else base


def get_ors_api_key() -> str | None:
    """OpenRouteService key, if the user has supplied one.

    Read from the environment (``ORS_API_KEY``), optionally populated from a
    ``.env`` file. Absent is a completely normal state -- the routing layer
    falls back to its analytic router.
    """
    key = os.environ.get("ORS_API_KEY", "").strip()
    if key:
        return key

    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip() == "ORS_API_KEY":
                value = value.strip().strip("'\"")
                if value and not value.startswith("your_"):
                    return value
    return None


def random_seed() -> int:
    return int(load_config().get_path("project.random_seed", 42))
