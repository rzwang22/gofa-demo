#!/usr/bin/env python3
import argparse
import csv
import json
import os
import signal
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    from .large_workload_common import PROFILE_MODES, REPOSITORY_ROOT, add_common_arguments, stage_commands
    from .validate_h100_per_query_latency import load_expected_traces
except ImportError:
    from large_workload_common import PROFILE_MODES, REPOSITORY_ROOT, add_common_arguments, stage_commands
    from validate_h100_per_query_latency import load_expected_traces


QUANT_MODE = "cache_w8a8_m4k2v2"
CACHE_MODES = {"cache_bf16", QUANT_MODE}


def compute_process_pids():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Unable to query GPU compute processes with nvidia-smi: {exc}") from exc
    return sorted({int(line.strip()) for line in output.splitlines() if line.strip().isdigit()})


def _read_rows(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return [], []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def inspect_completed_repetitions(csv_path, trace_index_path, task, mode, samples_per_split):
    expected = load_expected_traces(str(trace_index_path), task)
    expected_total = 2 * int(samples_per_split)
    if len(expected) != expected_total:
        raise RuntimeError(
            f"{task}: formal trace has {len(expected)} entries, expected {expected_total}: {trace_index_path}"
        )
    _fieldnames, rows = _read_rows(csv_path)
    if not rows:
        return set(), set()
    expected_by_order = {int(entry["trace_order"]): entry for entry in expected}
    grouped = defaultdict(list)
    for row in rows:
        if row.get("task") != task or row.get("profile_mode") != mode:
            raise RuntimeError(
                f"Latency CSV identity mismatch in {csv_path}: "
                f"task={row.get('task')!r}, mode={row.get('profile_mode')!r}"
            )
        try:
            grouped[int(row["rep"])].append(row)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Latency CSV has an invalid rep in {csv_path}: {row}") from exc

    complete = set()
    incomplete = set()
    for rep, rep_rows in grouped.items():
        valid = len(rep_rows) == expected_total
        orders = []
        split_indices = defaultdict(list)
        if valid:
            for row in rep_rows:
                try:
                    order = int(row["trace_order"])
                    query_index = int(row["query_index"])
                except (KeyError, TypeError, ValueError):
                    valid = False
                    break
                orders.append(order)
                split_indices[row.get("split")].append(query_index)
                expected_entry = expected_by_order.get(order)
                if expected_entry is None:
                    valid = False
                    break
                for field in ("task", "split", "query_index", "query_uid", "graph_signature", "workload_profile"):
                    actual = query_index if field == "query_index" else row.get(field)
                    if actual != expected_entry[field]:
                        valid = False
                        break
                if not valid:
                    break
                if mode in CACHE_MODES and int(row.get("cache_misses", -1)) != 0:
                    valid = False
                    break
                if mode == QUANT_MODE and int(row.get("fallback_count", -1)) != 0:
                    valid = False
                    break
        valid = valid and sorted(orders) == list(range(expected_total))
        valid = valid and sorted(split_indices["val"]) == list(range(int(samples_per_split)))
        valid = valid and sorted(split_indices["test"]) == list(range(int(samples_per_split)))
        (complete if valid else incomplete).add(rep)
    return complete, incomplete


def remove_repetitions(csv_path, repetitions):
    repetitions = set(int(rep) for rep in repetitions)
    if not repetitions:
        return
    path = Path(csv_path)
    fieldnames, rows = _read_rows(path)
    retained = [row for row in rows if int(row.get("rep", -1)) not in repetitions]
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(retained)
    temporary.replace(path)


def _rollback_csv(path, original_size):
    path = Path(path)
    if original_size == 0:
        if path.exists():
            path.unlink()
        return
    if path.exists():
        with path.open("r+b") as handle:
            handle.truncate(original_size)


def _terminate_process_group(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, AttributeError):
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, AttributeError):
            process.kill()
        process.wait()


def archive_contaminated_log(log_path, timestamp=None):
    log_path = Path(log_path)
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = log_path.with_name(f"{log_path.stem}.contaminated_{timestamp}{log_path.suffix}")
    collision = 1
    while candidate.exists():
        candidate = log_path.with_name(
            f"{log_path.stem}.contaminated_{timestamp}_{collision}{log_path.suffix}"
        )
        collision += 1
    log_path.replace(candidate)
    return candidate


def archive_existing_contaminated_log(log_path):
    log_path = Path(log_path)
    if not log_path.is_file():
        return None
    with log_path.open("rb") as handle:
        handle.seek(max(log_path.stat().st_size - 65536, 0))
        tail = handle.read()
    if b"CONTAMINATED" not in tail:
        return None
    return archive_contaminated_log(log_path)


def run_rep_transaction(
    command,
    csv_path,
    log_path,
    verify_complete,
    query_pids=compute_process_pids,
    popen_factory=subprocess.Popen,
    sleep=time.sleep,
    poll_interval=1.0,
):
    initial_pids = query_pids()
    if initial_pids:
        raise RuntimeError(f"GPU is not idle before repetition; compute_pids={initial_pids}")
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    original_size = csv_path.stat().st_size if csv_path.exists() else 0
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    archive_existing_contaminated_log(log_path)
    contaminated = False
    with log_path.open("w") as log_handle:
        log_handle.write(f"command={command!r}\n")
        log_handle.flush()
        process = popen_factory(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                pids = query_pids()
                log_handle.write(f"gpu_compute_pids={pids}\n")
                log_handle.flush()
                if len(pids) > 1:
                    contaminated = True
                    log_handle.write(f"CONTAMINATED multiple_compute_processes={pids}\n")
                    log_handle.flush()
                    _terminate_process_group(process)
                    break
                sleep(poll_interval)
            return_code = process.wait()
            if contaminated:
                raise RuntimeError(f"GPU repetition contaminated by multiple compute processes; log={log_path}")
            if return_code != 0:
                raise RuntimeError(f"GPU repetition exited with code {return_code}; log={log_path}")
            verify_complete()
        except Exception as exc:
            _terminate_process_group(process)
            _rollback_csv(csv_path, original_size)
            if contaminated:
                log_handle.flush()
                archived_log = archive_contaminated_log(log_path)
                raise RuntimeError(
                    "GPU repetition contaminated by multiple compute processes; "
                    f"audit_log={archived_log}"
                ) from exc
            raise
    return log_path


def _rep_command(config_path, append):
    command = ["python3", str(REPOSITORY_ROOT / "run_gofa.py"), "--override", str(config_path)]
    if append:
        command.extend(["gofa_per_query_latency_append", "True"])
    return command


def run_suite(args):
    manifest_path, plan_path, _commands = stage_commands(args, "gpu")
    with Path(manifest_path).open() as handle:
        suite = json.load(handle)
    print(f"suite_manifest={manifest_path}")
    print(f"command_plan={plan_path}")
    if not args.execute:
        print("dry_run=True; no GPU process checks or experiments were run")
        return

    samples = int(suite["profile"]["samples_per_split"])
    requested_reps = int(suite["reps"])
    for mode in PROFILE_MODES:
        for task in suite["tasks"]:
            csv_path = Path(suite["paths"]["latency"]) / mode / f"{task}.csv"
            trace_index = Path(suite["paths"]["traces"]) / f"{task}_formal_v1" / "trace_index.jsonl"
            complete, incomplete = inspect_completed_repetitions(
                csv_path, trace_index, task, mode, samples
            )
            unexpected = (complete | incomplete) - set(range(requested_reps))
            if unexpected:
                raise RuntimeError(f"{task}/{mode}: CSV contains unexpected repetitions {sorted(unexpected)}")
            if incomplete:
                print(f"GOFA GPU resume removing incomplete reps task={task} mode={mode} reps={sorted(incomplete)}")
                remove_repetitions(csv_path, incomplete)
                complete, _ = inspect_completed_repetitions(csv_path, trace_index, task, mode, samples)
            if complete and complete != set(range(max(complete) + 1)):
                raise RuntimeError(f"{task}/{mode}: completed reps are not contiguous: {sorted(complete)}")

            for rep in range(requested_reps):
                if rep in complete:
                    print(f"GOFA GPU resume skip completed task={task} mode={mode} rep={rep}")
                    continue
                if complete != set(range(rep)):
                    raise RuntimeError(
                        f"{task}/{mode}: cannot start rep={rep}; completed={sorted(complete)}"
                    )
                config_path = suite["configs"][task][f"gpu_{mode}"]
                command = _rep_command(config_path, append=csv_path.exists() and csv_path.stat().st_size > 0)
                log_path = Path(suite["paths"]["logs"]) / "gpu" / mode / task / f"rep_{rep:03d}.log"

                def verify(rep=rep):
                    verified, partial = inspect_completed_repetitions(
                        csv_path, trace_index, task, mode, samples
                    )
                    if partial or rep not in verified:
                        raise RuntimeError(
                            f"{task}/{mode}/rep={rep} did not produce a complete validated repetition; "
                            f"complete={sorted(verified)}, incomplete={sorted(partial)}"
                        )

                print(f"GOFA GPU run task={task} mode={mode} rep={rep} log={log_path}")
                run_rep_transaction(command, csv_path, log_path, verify)
                complete.add(rep)


def main():
    parser = add_common_arguments(argparse.ArgumentParser(description="Run the resumable GOFA large GPU suite."))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    run_suite(args)


if __name__ == "__main__":
    main()
