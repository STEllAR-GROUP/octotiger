#!/usr/bin/env python3
"""Run the reviewed PowerTiger update without depending on a moving branch."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=pathlib.Path)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--jobs", required=True, type=int)
    parser.add_argument("--dirty", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    stack = pathlib.Path(__file__).resolve().parent
    lock = json.loads((stack / "powertiger-lock.json").read_text())
    patch = stack / lock["patch"]["file"]
    if hashlib.sha256(patch.read_bytes()).hexdigest() != lock["patch"]["sha256"]:
        raise RuntimeError("PowerTiger patch checksum differs from its lock")
    source = args.source.resolve()
    checkout = source / ".jenkins-powertiger"
    # Each Jenkins row starts in a fresh source copy. Refuse to reuse a checkout
    # whose provenance or previous patch application could be ambiguous.
    checkout.mkdir()
    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(checkout), *arguments], text=True)
    git("init")
    git("remote", "add", "origin", lock["url"])
    git("fetch", "--depth=1", "origin", lock["commit"])
    git("checkout", "--detach", "FETCH_HEAD")
    if git("rev-parse", "HEAD").strip() != lock["commit"]:
        raise RuntimeError("PowerTiger fetched revision differs from its lock")
    git("apply", "--check", str(patch))
    git("apply", str(patch))
    (source / ".jenkins-powertiger-provenance.json").write_text(json.dumps(lock, indent=2) + "\n")
    command = [sys.executable, str(checkout / "build-stack.py"),
               "--stack-dir", str(stack), "--source", str(source),
               "--spec", "octotiger@develop " + args.spec, "--jobs", str(args.jobs)]
    if args.dirty:
        command.append("--dirty")
    return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())
