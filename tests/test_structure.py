"""Step 1 sanity checks: structure and config load correctly."""

from pathlib import Path

from phase1_anpr.utils.config import REQUIRED_SECTIONS, load_config

CONFIG_PATH = Path("phase1_anpr/config/config.yaml")

EXPECTED_DIRS = [
    "phase1_anpr/config",
    "phase1_anpr/video",
    "phase1_anpr/detection",
    "phase1_anpr/tracking",
    "phase1_anpr/quality",
    "phase1_anpr/preprocessing",
    "phase1_anpr/ocr",
    "phase1_anpr/normalization",
    "phase1_anpr/models",
    "phase1_anpr/pipeline",
    "phase1_anpr/utils",
    "inputs/videos",
    "outputs/plates",
    "outputs/annotated",
    "outputs/observations",
    "weights",
    "tests",
]


def test_expected_directories_exist():
    for d in EXPECTED_DIRS:
        assert Path(d).is_dir(), f"missing directory: {d}"


def test_config_loads_with_all_sections():
    config = load_config(CONFIG_PATH)
    for section in REQUIRED_SECTIONS:
        assert section in config, f"missing config section: {section}"


def test_output_paths_configured():
    config = load_config(CONFIG_PATH)
    assert config["output"]["observations_file"].endswith(".jsonl")
