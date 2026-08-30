"""Minimal config loading for Phase 1 (Step 1)."""

from pathlib import Path

import yaml

# Default path to the project config file.
DEFAULT_CONFIG_PATH = Path("phase1_anpr/config/config.yaml")

# Sections we expect config.yaml to define.
REQUIRED_SECTIONS = (
    "video",
    "detection",
    "tracking",
    "quality",
    "preprocessing",
    "ocr",
    "confidence",
    "output",
    "models",
)


def load_config(config_path=DEFAULT_CONFIG_PATH):
    """Load config.yaml and return it as a dict.

    Raises FileNotFoundError if the file is missing and
    ValueError if any required section is absent.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    missing = [s for s in REQUIRED_SECTIONS if s not in config]
    if missing:
        raise ValueError(f"Config is missing sections: {', '.join(missing)}")

    return config
