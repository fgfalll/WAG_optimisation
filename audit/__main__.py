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
    parser.add_argument(
        "--checklist-only", action="store_true", help="Generate and refresh Agent Showcase Checklist only"
    )
    parser.add_argument(
        "--continuity",
        action="store_true",
        help="Verify every 'RESOLVED' claim by measurement and check wiki/code drift",
    )
    parser.add_argument(
        "--continuity-only",
        nargs="?",
        const="",
        metavar="FINDING_ID",
        help="Run only the continuity gate (optionally for a single finding, e.g. CRIT-14)",
    )
    parser.add_argument(
        "--issue-gate",
        action="store_true",
        help="Validate the staged commit against the 1-commit-1-issue policy",
    )
    args = parser.parse_args()

    try:
        if args.issue_gate:
            from audit.continuity import gate_staged_commit
            ok, problems = gate_staged_commit()
            if ok:
                print("issue-discipline gate: PASS (1 commit = 1 issue)")
            else:
                print("issue-discipline gate: FAIL")
                for prob in problems:
                    print(f"  - {prob}")
                sys.exit(2)

        if args.continuity_only is not None:
            from audit import continuity
            only = (args.continuity_only or "").upper() or None
            results = continuity.run_checks(only)
            wiki = continuity.check_wiki()
            print(continuity.render_report(results, wiki))
            continuity.write_json_report(
                results, wiki, continuity.REPO_ROOT / "audit" / "continuity_report.json"
            )
            sys.exit(1 if any(r.should_reopen for r in results) else 0)

        if args.continuity:
            from audit import continuity
            results = continuity.run_checks()
            wiki = continuity.check_wiki()
            print(continuity.render_report(results, wiki))
            continuity.write_json_report(
                results, wiki, continuity.REPO_ROOT / "audit" / "continuity_report.json"
            )
            sys.exit(1 if any(r.should_reopen for r in results) else 0)

        if args.checklist_only:
            from audit.generate_agent_checklist import run_checklist_pipeline
            run_checklist_pipeline()
        else:
            run_full_audit()
        sys.exit(0)
    except Exception as e:
        print(f"Audit failed with exception: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
