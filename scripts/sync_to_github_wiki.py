"""
Script to synchronize Agent Wiki documentation with GitHub Wiki (.wiki.git).

Usage:
    python scripts/sync_to_github_wiki.py [--remote <wiki_git_url>] [--dry-run]
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_WIKI_URL = "https://github.com/fgfalll/WAG_optimisation.wiki.git"

def sync_wiki(remote_url: str, dry_run: bool = False):
    root_dir = Path(__file__).resolve().parent.parent
    agent_wiki_dir = root_dir / "agent_wiki"
    temp_wiki_dir = root_dir / ".temp_wiki_sync"

    if not agent_wiki_dir.exists():
        print(f"[ERROR] Source wiki directory not found at: {agent_wiki_dir}")
        sys.exit(1)

    print(f"[INFO] Source directory: {agent_wiki_dir}")
    print(f"[INFO] Target GitHub Wiki remote: {remote_url}")

    if temp_wiki_dir.exists():
        shutil.rmtree(temp_wiki_dir, ignore_errors=True)

    print(f"[INFO] Cloning GitHub Wiki repository...")
    clone_res = subprocess.run(["git", "clone", remote_url, str(temp_wiki_dir)], capture_output=True, text=True)

    if clone_res.returncode != 0:
        print("[WARNING] Could not clone GitHub Wiki repository directly.")
        print("          If GitHub Wiki is uninitialized, create at least one page via the GitHub Web UI first:")
        print("          https://github.com/fgfalll/WAG_optimisation/wiki")
        print(f"          Details: {clone_res.stderr.strip()}")
        return False

    # Copy files
    print("[INFO] Syncing markdown files...")
    copied_count = 0
    for root, _, files in os.walk(agent_wiki_dir):
        for f in files:
            if f.endswith(".md"):
                rel_path = Path(root).relative_to(agent_wiki_dir)
                dest_dir = temp_wiki_dir / rel_path
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(Path(root) / f, dest_dir / f)
                copied_count += 1

    # Ensure Home.md exists for GitHub Wiki landing page
    readme_path = temp_wiki_dir / "README.md"
    home_path = temp_wiki_dir / "Home.md"
    if readme_path.exists():
        shutil.copy2(readme_path, home_path)

    print(f"[INFO] Copied {copied_count} wiki pages into target.")

    if dry_run:
        print("[INFO] Dry run enabled. Skipping git commit and push.")
        return True

    # Git commit and push
    subprocess.run(["git", "-C", str(temp_wiki_dir), "add", "."], check=True)
    status_res = subprocess.run(["git", "-C", str(temp_wiki_dir), "status", "--porcelain"], capture_output=True, text=True)
    
    if not status_res.stdout.strip():
        print("[INFO] No documentation changes to commit. GitHub Wiki is already up-to-date.")
    else:
        subprocess.run(["git", "-C", str(temp_wiki_dir), "commit", "-m", "docs: update GitHub Wiki from agent_wiki"], check=True)
        push_res = subprocess.run(["git", "-C", str(temp_wiki_dir), "push", "origin", "master"], capture_output=True, text=True)
        if push_res.returncode != 0:
            push_res = subprocess.run(["git", "-C", str(temp_wiki_dir), "push", "origin", "main"], capture_output=True, text=True)
        print("[SUCCESS] GitHub Wiki successfully synchronized!")

    shutil.rmtree(temp_wiki_dir, ignore_errors=True)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync Agent Wiki to GitHub Wiki")
    parser.add_argument("--remote", default=DEFAULT_WIKI_URL, help="GitHub Wiki git remote URL")
    parser.add_argument("--dry-run", action="store_true", help="Simulate sync without committing or pushing")
    args = parser.parse_args()

    sync_wiki(remote_url=args.remote, dry_run=args.dry_run)
