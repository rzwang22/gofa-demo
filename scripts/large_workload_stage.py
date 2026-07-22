#!/usr/bin/env python3
import argparse
import subprocess

from large_workload_common import add_common_arguments, stage_commands


def main():
    parser = add_common_arguments(argparse.ArgumentParser(description="Run or emit a GOFA large-workload stage."))
    parser.add_argument("--stage", required=True, choices=("workload", "full-cache", "formal-trace", "gpu"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifest_path, plan_path, commands = stage_commands(args, args.stage)
    print(f"suite_manifest={manifest_path}")
    print(f"command_plan={plan_path}")
    if not args.execute:
        print("dry_run=True; pass --execute to run the generated command plan")
        return
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
