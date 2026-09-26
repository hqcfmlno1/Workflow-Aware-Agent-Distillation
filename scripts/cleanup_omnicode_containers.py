"""Remove finished OmniCode Docker containers without touching images."""

from __future__ import annotations

import argparse

from workflow_distillation.omnicode_collector import cleanup_omnicode_containers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", help="Only clean containers whose name contains this task ID.")
    parser.add_argument(
        "--include-running",
        action="store_true",
        help="Force-remove matching running containers; use only after their task has stopped.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    selected = cleanup_omnicode_containers(
        args.task_id,
        include_running=args.include_running,
        dry_run=args.dry_run,
    )
    action = "would remove" if args.dry_run else "removed"
    print(f"{action}_containers={len(selected)}")
    for container_id in selected:
        print(container_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
