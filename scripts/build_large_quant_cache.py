#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess

from large_workload_common import REPOSITORY_ROOT, add_common_arguments, prepare_suite


def main():
    parser = add_common_arguments(argparse.ArgumentParser(description="Build task-specific M4K2V2 caches."))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifest_path, suite = prepare_suite(args)
    paths = suite["paths"]
    commands = []
    for task in suite["tasks"]:
        commands.append([
            "python3",
            str(REPOSITORY_ROOT / "scripts" / "quantize_scheme_b_cache.py"),
            "--input-cache-dir", paths["full_cache"],
            "--output-cache-dir", f"{paths['quant_cache']}/{task}",
            "--manifest", f"{paths['manifests']}/{task}.json",
            "--base-bits", "4",
            "--delta-bits", "4",
            "--memory-base-bits", "4",
            "--key-base-bits", "2",
            "--value-base-bits", "2",
            "--memory-delta-bits", "4",
            "--key-delta-bits", "2",
            "--value-delta-bits", "2",
        ])
    plan_path = suite["paths"]["plans"] + "/quant-cache.sh"
    with open(plan_path, "w") as handle:
        handle.write("#!/usr/bin/env bash\nset -euo pipefail\n")
        for command in commands:
            handle.write(" ".join(shlex.quote(part) for part in command) + "\n")
    os.chmod(plan_path, 0o755)
    print(f"suite_manifest={manifest_path}")
    print(f"command_plan={plan_path}")
    if not args.execute:
        print("dry_run=True; pass --execute to run cache conversion")
        return
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
