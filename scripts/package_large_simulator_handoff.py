#!/usr/bin/env python3
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from large_workload_common import add_common_arguments, suite_root


INCLUDE_PATHS = ("suite_manifest.json", "configs", "plans", "manifests", "traces", "latency", "summary")


def main():
    parser = add_common_arguments(argparse.ArgumentParser(description="Package GOFA simulator handoff artifacts."))
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    root = suite_root(args)
    output = Path(args.output).expanduser().resolve() if args.output else root / f"{args.profile_name}_simulator_handoff.tar.gz"
    checksums = {}
    with tarfile.open(output, "w:gz") as archive:
        for relative in INCLUDE_PATHS:
            path = root / relative
            if not path.exists():
                continue
            archive.add(path, arcname=f"{args.profile_name}/{relative}")
            if path.is_file():
                checksums[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            else:
                for file_path in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
                    key = str(file_path.relative_to(root))
                    checksums[key] = hashlib.sha256(file_path.read_bytes()).hexdigest()
    checksum_path = output.with_suffix(output.suffix + ".sha256.json")
    with checksum_path.open("w") as handle:
        json.dump({"archive": str(output), "files": checksums}, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"handoff={output}")
    print(f"checksums={checksum_path}")


if __name__ == "__main__":
    main()
