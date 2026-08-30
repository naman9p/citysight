"""CitySight Phase 1 — ANPR Engine entry point.

Step 1 only: load and validate config.yaml, then print a summary.
Later steps will wire in the detection → tracking → OCR → fusion pipeline.
"""

import argparse

from phase1_anpr.utils.config import DEFAULT_CONFIG_PATH, load_config


def main():
    parser = argparse.ArgumentParser(description="CitySight Phase 1 ANPR Engine")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    print("CitySight Phase 1 ANPR — config loaded OK")
    print(f"  config file : {args.config}")
    print(f"  sections    : {', '.join(config.keys())}")
    print(f"  model_version: {config['models'].get('model_version')}")
    print("Pipeline not implemented yet (Step 1 only).")


if __name__ == "__main__":
    main()
