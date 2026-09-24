"""
Command-line interface for the automated scientific audit pipeline:
    python -m audit
"""

import sys
import argparse
from audit.pipeline import run_full_audit


def main():
    parser = argparse.ArgumentParser(
        description="CO2 EOR Optimizer - Automated Scientific & Software Audit Pipeline"
    )
    parser.add_argument(
        "--all", action="store_true", default=True, help="Run complete audit suite (default: True)"
    )
    args = parser.parse_args()

    try:
        run_full_audit()
        sys.exit(0)
    except Exception as e:
        print(f"Audit failed with exception: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
