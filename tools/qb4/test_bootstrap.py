#!/usr/bin/env python3
"""Allocation/command checks with fake CMake only; no package is compiled."""
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('bootstrap_cpu_stack.sh')


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='qb-bootstrap-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'sources'
        for name in ('hpx/CMakeLists.txt', 'cppuddle/CMakeLists.txt',
                     'stdexec/include/stdexec/execution.hpp', 'asio/asio/include/asio.hpp'):
            path = self.source / name
            path.parent.mkdir(parents=True)
            path.write_text('staged source fixture\n')
        fake = self.root / 'bin'
        fake.mkdir()
        (fake / 'cmake').write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
            'with open(os.environ["MOCK_CALLS"],"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n')
        (fake / 'nproc').write_text('#!/usr/bin/env bash\nprintf "4\\n"\n')
        for path in fake.iterdir():
            path.chmod(0o755)
        self.environment = self.root / 'environment.sh'
        self.environment.write_text('export CC=/bin/true CXX=/bin/true MPI_C_COMPILER=/bin/true MPI_CXX_COMPILER=/bin/true\n')
        self.env = dict(os.environ, PATH=str(fake)+':'+os.environ['PATH'],
                        SLURM_JOB_ID='42', SLURM_CPUS_PER_TASK='4',
                        SLURMD_NODENAME=socket.gethostname().split('.')[0],
                        MOCK_CALLS=str(self.root/'calls'))
        self.env.pop('CMAKE_PREFIX_PATH', None)
        self.cmd = ['bash', str(SCRIPT), '--env', str(self.environment),
                    '--sources', str(self.source), '--build-dir', str(self.root/'build'),
                    '--prefix', str(self.root/'install'), '-j', '2']

    def test_configures_cpu_mpi_and_records_sources(self):
        subprocess.run(self.cmd, env=self.env, check=True, capture_output=True)
        calls = [json.loads(line) for line in (self.root/'calls').read_text().splitlines()]
        self.assertEqual(len(calls), 6)
        for expected in ('-DHPX_WITH_CUDA=OFF', '-DHPX_WITH_PARCELPORT_MPI=ON',
                         '-DHPX_WITH_THREAD_STACK_MMAP=OFF',
                         '-DHPX_WITH_PARCELPORT_TCP=OFF', '-DFETCHCONTENT_FULLY_DISCONNECTED=ON'):
            self.assertIn(expected, calls[0])
        self.assertIn('-DCPPUDDLE_WITH_KOKKOS=OFF', calls[3])
        manifest = json.loads((self.root/'build/source-manifest.json').read_text())
        self.assertEqual(set(manifest), {'hpx', 'cppuddle', 'asio', 'stdexec'})
        self.assertTrue(all(len(value['tree_sha256']) == 64 for value in manifest.values()))

    def test_login_shell_rejected(self):
        self.env['SLURMD_NODENAME'] = 'not-this-host'
        result = subprocess.run(self.cmd, env=self.env, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertFalse((self.root/'calls').exists())

    def test_excess_jobs_rejected(self):
        result = subprocess.run(self.cmd[:-1]+['5'], env=self.env, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertFalse((self.root/'calls').exists())


if __name__ == '__main__':
    unittest.main()
