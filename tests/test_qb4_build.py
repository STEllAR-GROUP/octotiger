#!/usr/bin/env python3
"""Offline build-wrapper checks; every CMake/build command is a local mock."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
REAL_CMAKE = shutil.which("cmake")


class Qb4Build(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="octotiger-qb-build-")
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.bin = self.path / "bin"
        self.bin.mkdir()
        self.log = self.path / "commands.jsonl"
        self.build = self.path / "build with spaces"
        self.environment = os.environ.copy()
        for key in list(self.environment):
            if key.startswith("SLURM") or key in ("QB_ENV_FILE", "JOBS", "MAKEFLAGS", "MFLAGS"):
                del self.environment[key]
        self.environment.update({
            "PATH": f"{self.bin}:{os.environ['PATH']}",
            "SLURM_JOB_ID": "12345", "SLURM_CPUS_PER_TASK": "4",
            "SLURMD_NODENAME": "qbd-test", "MOCK_LOG": str(self.log),
            "MOCK_AFFINITY": "6",
        })
        self.program("hostname", "#!/bin/sh\nprintf 'qbd-test\\n'\n")
        self.program("nproc", "#!/bin/sh\nprintf '%s\\n' \"$MOCK_AFFINITY\"\n")
        self.program("cmake", r'''#!/usr/bin/env python3
import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ["MOCK_LOG"], "a") as stream:
    stream.write(json.dumps({"args": args, "parallel": os.getenv("CMAKE_BUILD_PARALLEL_LEVEL"),
        "makeflags": os.getenv("MAKEFLAGS"), "omp": os.getenv("OMP_NUM_THREADS")}) + "\n")
if "--build" in args:
    target = pathlib.Path(args[args.index("--build") + 1])
    names = ("radiation-results", "gen_radiation_reference") if target.name == "radiation-results-tools" else ("octotiger",)
    for name in names:
        exe = target / name
        exe.write_text("#!/bin/sh\nexit 0\n")
        exe.chmod(0o755)
else:
    target = pathlib.Path(args[args.index("-B") + 1])
    target.mkdir(parents=True, exist_ok=True)
    if target.name == "hpx-check":
        if os.getenv("MOCK_PROBE_FAIL"):
            sys.exit(1)
        (target / "hpx-mpi-selection.txt").write_text("HPX_VERSION=2.0.0\nHPX_WITH_PARCELPORT_MPI=ON\n")
''')
        for variable, name, filename in (
            ("HPX_DIR", "hpx", "HPXConfig.cmake"),
            ("FETCHCONTENT_SOURCE_DIR_KOKKOSSIMD", "simd math", "simd.hpp"),
            ("FETCHCONTENT_SOURCE_DIR_SVESIMD", "sve", "CMakeLists.txt"),
        ):
            directory = self.path / name
            directory.mkdir()
            (directory / filename).touch()
            self.environment[variable] = str(directory)

    def program(self, name, text):
        target = self.bin / name
        target.write_text(text)
        target.chmod(0o755)

    def run_build(self, *args, succeeds=True):
        result = subprocess.run(
            ["bash", str(ROOT / "build.sh"), "--build-dir", str(self.build), *args],
            env=self.environment, cwd=self.path, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode == 0, succeeds, result.stdout + result.stderr)
        return result

    def commands(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_allocation_required_before_any_build(self):
        del self.environment["SLURM_JOB_ID"]
        self.assertIn("allocated Slurm", self.run_build(succeeds=False).stderr)
        self.assertEqual(self.commands(), [])

    def test_login_shell_with_job_id_is_rejected(self):
        self.environment["SLURMD_NODENAME"] = "another-node"
        self.assertIn("not the allocated compute node", self.run_build(succeeds=False).stderr)
        self.assertEqual(self.commands(), [])

    def test_job_count_cannot_exceed_task_or_affinity(self):
        self.run_build("-j", "5", succeeds=False)
        self.assertEqual(self.commands(), [])
        self.environment["MOCK_AFFINITY"] = "2"
        self.run_build("-j", "3", succeeds=False)
        self.assertEqual(self.commands(), [])

    def test_missing_staged_dependency_never_invokes_cmake(self):
        del self.environment["FETCHCONTENT_SOURCE_DIR_SVESIMD"]
        self.assertIn("downloads are disabled", self.run_build(succeeds=False).stderr)
        self.assertEqual(self.commands(), [])

    def test_kokkossimd_requires_the_header_not_cmake(self):
        directory = Path(self.environment["FETCHCONTENT_SOURCE_DIR_KOKKOSSIMD"])
        self.assertFalse((directory / "CMakeLists.txt").exists())
        (directory / "simd.hpp").unlink()
        (directory / "CMakeLists.txt").touch()
        self.assertIn("containing simd.hpp", self.run_build(succeeds=False).stderr)
        self.assertEqual(self.commands(), [])

    def test_probe_failure_stops_before_application_configure(self):
        self.environment["MOCK_PROBE_FAIL"] = "1"
        self.run_build(succeeds=False)
        self.assertEqual(len(self.commands()), 1)

    @unittest.skipUnless(REAL_CMAKE, "CMake is needed to interpret the probe guards")
    def test_cmake_probe_rejects_wrong_version_and_transport(self):
        # Interpret the real guard code while mocking package discovery and
        # project/compiler setup. This performs no compile, MPI or Slurm call.
        probe = self.path / "probe.cmake"
        for major, networking, mpi, succeeds in (
            (2, "ON", "ON", True), (1, "ON", "ON", False),
            (2, "OFF", "ON", False), (2, "ON", "OFF", False),
        ):
            with self.subTest(major=major, networking=networking, mpi=mpi):
                probe.write_text(
                    "macro(project)\nendmacro()\nmacro(find_package)\nendmacro()\n"
                    f"set(HPX_VERSION_MAJOR {major})\nset(HPX_VERSION_STRING {major}.0.0)\n"
                    f"set(HPX_WITH_NETWORKING {networking})\nset(HPX_WITH_PARCELPORT_MPI {mpi})\n"
                    f"set(CMAKE_CURRENT_BINARY_DIR [=[{self.path}]=])\n"
                    f"include([=[{ROOT / 'tools/qb4/hpx-check/CMakeLists.txt'}]=])\n"
                )
                result = subprocess.run([REAL_CMAKE, "-P", str(probe)],
                                        text=True, capture_output=True, cwd=self.path)
                self.assertEqual(result.returncode == 0, succeeds, result.stdout + result.stderr)

    def test_environment_paths_outputs_and_bounded_builds(self):
        environment_file = self.path / "site environment.sh"
        environment_file.write_text("export CMAKE_PREFIX_PATH='/opt/tool one:/opt/tool-two'\nexport Silo_DIR='/opt/silo'\n")
        self.environment["QB_ENV_FILE"] = str(environment_file)
        self.environment["MAKEFLAGS"] = "-j999"
        self.environment["OMP_NUM_THREADS"] = "999"
        self.run_build()
        commands = self.commands()
        self.assertEqual(len(commands), 5)
        self.assertEqual([c["args"][0] for c in commands], ["-S", "-S", "--build", "-S", "--build"])
        for command in commands:
            self.assertEqual(command["parallel"], "4")
            self.assertEqual(command["makeflags"], "-j4")
            self.assertEqual(command["omp"], "1")
            if command["args"][0] == "--build":
                self.assertEqual(command["args"][-2:], ["--parallel", "4"])
            else:
                self.assertIn("-DCMAKE_PREFIX_PATH=/opt/tool one;/opt/tool-two", command["args"])
        app_args = commands[1]["args"]
        for option in ("-DOCTOTIGER_WITH_RADIATION=ON", "-DOCTOTIGER_WITH_TESTS=ON",
                       "-DOCTOTIGER_WITH_GRIDDIM=8", "-DOCTOTIGER_WITH_FAST_FP_CONTRACT=OFF",
                       "-DOCTOTIGER_WITH_CUDA=OFF", "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"):
            self.assertIn(option, app_args)
        self.assertTrue((self.build / "octotiger").is_file())
        self.assertTrue((self.build / "radiation-results-tools/radiation-results").is_file())
        self.assertIn("jobs=4", (self.build / "qb4-build-manifest.txt").read_text())


if __name__ == "__main__":
    unittest.main()
