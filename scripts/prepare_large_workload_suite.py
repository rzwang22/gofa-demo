#!/usr/bin/env python3
import argparse
import subprocess

from large_workload_common import add_common_arguments, prepare_suite, stage_commands


def main():
    parser = add_common_arguments(argparse.ArgumentParser(description="Prepare GOFA large-workload suite configs."))
    parser.add_argument("--execute", action="store_true", help="Generate the frozen val/test workload now.")
    args = parser.parse_args()
    manifest_path, manifest = prepare_suite(args)
    print(f"suite_manifest={manifest_path}")
    print(f"tasks={','.join(manifest['tasks'])}")
    print(f"profile={manifest['profile']}")
    _, plan_path, commands = stage_commands(args, "workload")
    print(f"workload_command_plan={plan_path}")
    if args.execute:
        for command in commands:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
