#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path

from large_workload_common import add_common_arguments, stage_commands, write_json_yaml
from large_workload_trace_state import prepare_formal_trace_outputs


def main():
    parser = add_common_arguments(argparse.ArgumentParser(description="Run or emit a GOFA large-workload stage."))
    parser.add_argument("--stage", required=True, choices=("workload", "full-cache", "formal-trace", "gpu"))
    parser.add_argument("--execute", action="store_true")
    trace_mode = parser.add_mutually_exclusive_group()
    trace_mode.add_argument("--fresh", action="store_true")
    trace_mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage != "formal-trace" and (args.fresh or args.resume):
        parser.error("--fresh/--resume are only valid with --stage formal-trace")
    manifest_path, plan_path, commands = stage_commands(args, args.stage)
    if args.stage == "formal-trace":
        with Path(manifest_path).open() as handle:
            manifest = json.load(handle)
        states = prepare_formal_trace_outputs(
            manifest,
            fresh=args.fresh,
            resume=args.resume,
            execute=args.execute,
        )
        for task in manifest["tasks"]:
            state = states[task]
            config_path = Path(manifest["configs"][task]["formal_trace"])
            with config_path.open() as handle:
                config = json.load(handle)
            config["gofa_query_trace"]["resume"] = state["action"] == "resume"
            write_json_yaml(config_path, config)
            print(
                f"formal_trace task={task} action={state['action']} "
                f"existing_entries={state['existing_entries']} output_dir={state['trace_dir']}"
            )
    if args.stage == "gpu" and args.execute:
        raise RuntimeError(
            "GPU execution must use scripts/run_large_gpu_suite.sh so repetitions are monitored and transactional."
        )
    print(f"suite_manifest={manifest_path}")
    print(f"command_plan={plan_path}")
    if not args.execute:
        print("dry_run=True; pass --execute to run the generated command plan")
        return
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
