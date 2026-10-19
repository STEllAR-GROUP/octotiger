#!/usr/bin/env python3
"""Supervise intentional crashes without core files or lingering HPX processes."""

import argparse
import os
from pathlib import Path
import re
import resource
import signal
import socket
import subprocess
import sys
import tempfile
import time


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_trace(output, locality, source, frame):
    require(re.search(
        rf"Octo-Tiger stacktrace: host=\S+ pid=\d+ locality={locality}\b", output),
        "Missing process/locality identity in stacktrace")
    require("SIGABRT" in output, "Missing SIGABRT label")
    require(frame in output, f"Missing crashing frame {frame}")
    require(re.search(rf"{re.escape(source)}:\d+\b", output),
            f"Missing source line for {source}")


def thread_crash(executable):
    result = subprocess.run([executable, "crash"], capture_output=True, text=True,
                            timeout=20, check=False)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    require(result.returncode == -signal.SIGABRT,
            f"Expected SIGABRT termination, got {result.returncode}")
    check_trace(result.stderr, 7, "fixture.cpp", "crash_marker_stacktrace_thread")


def local_ports():
    # Hold both reservations simultaneously so that their port numbers differ.
    with socket.socket() as first, socket.socket() as second:
        first.bind(("127.0.0.1", 0))
        second.bind(("127.0.0.1", 0))
        return first.getsockname()[1], second.getsockname()[1]


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


def distributed(executable, crash):
    console_port, worker_port = local_ports()
    common = [executable, "--hpx:localities=2", "--hpx:threads=1", "--hpx:bind=none",
              f"--hpx:agas=127.0.0.1:{console_port}",
              "--hpx:ini=hpx.parcel.bootstrap=tcp",
              "--hpx:ini=hpx.parcel.tcp.enable=1",
              "--hpx:ini=hpx.parcel.mpi.enable!=0",
              "--hpx:ini=hpx.parcel.lci.enable!=0",
              "--hpx:ini=hpx.parcel.lcw.enable!=0"]
    processes = []
    with tempfile.TemporaryDirectory(prefix="octotiger-stacktrace-") as temporary:
        paths = [Path(temporary) / "console.log", Path(temporary) / "worker.log"]
        try:
            with paths[0].open("w") as console_log, paths[1].open("w") as worker_log:
                console_arguments = common + ["--hpx:console",
                    f"--hpx:hpx=127.0.0.1:{console_port}"]
                if crash:
                    console_arguments.append("--crash-worker")
                processes.append(subprocess.Popen(console_arguments, stdout=console_log,
                                                  stderr=subprocess.STDOUT))
                processes.append(subprocess.Popen(common + ["--hpx:worker",
                    f"--hpx:hpx=127.0.0.1:{worker_port}"], stdout=worker_log,
                    stderr=subprocess.STDOUT))
                deadline = time.monotonic() + 40
                while time.monotonic() < deadline:
                    if crash and processes[1].poll() is not None:
                        break
                    if not crash and all(process.poll() is not None for process in processes):
                        break
                    time.sleep(0.05)
                else:
                    raise RuntimeError("Two-locality fixture timed out")
                if crash:
                    require(processes[1].returncode == -signal.SIGABRT,
                            f"Worker did not die from SIGABRT: {processes[1].returncode}")
                else:
                    require(all(process.returncode == 0 for process in processes),
                            "Two-locality fixture failed: " +
                            repr([process.returncode for process in processes]))
        finally:
            for process in reversed(processes):
                stop(process)
            outputs = [path.read_text(errors="replace") if path.exists() else ""
                       for path in paths]
            for label, output in zip(("console", "worker"), outputs):
                sys.stdout.write(f"--- {label} ---\n{output}")
        require("CONSOLE_MAIN locality=0" in outputs[0], "Console hpx_main did not run")
        require("CONSOLE_MAIN" not in outputs[1], "Worker unexpectedly ran hpx_main")
        require("WORKER_READY locality=1 hpx_main_calls=0 installed=1" in outputs[1],
                "Stacktrace startup did not cover the worker locality")
        if crash:
            check_trace(outputs[1], 1, "hpx_fixture.cpp", "crash_marker_stacktrace_worker")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("thread-crash", "distributed", "distributed-crash"))
    parser.add_argument("executable")
    arguments = parser.parse_args()
    # Children inherit this limit.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    executable = os.path.abspath(arguments.executable)
    if arguments.mode == "thread-crash":
        thread_crash(executable)
    else:
        distributed(executable, arguments.mode == "distributed-crash")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"Stacktrace test failed: {error}", file=sys.stderr)
        sys.exit(1)
