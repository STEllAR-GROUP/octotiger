#!/usr/bin/env python3
"""Mocked timing/guard tests; no MPI, scheduler, or solver execution."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import benchmark as b

SOURCE = Path(__file__).resolve().parents[2]


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='qb4-benchmark-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.build = self.root / 'build'
        self.build.mkdir()
        (self.build / 'CMakeCache.txt').write_text('OCTOTIGER_WITH_GRIDDIM:STRING=8\n')
        (self.build / 'octotiger').write_text('#!/bin/sh\nexit 88\n')
        (self.build / 'octotiger').chmod(0o755)
        self.output = self.root / 'timing'
        self.env = dict(SLURM_JOB_ID='42', SLURMD_NODENAME='compute',
                        SLURM_NNODES='2', SLURM_NTASKS='2', SLURM_CPUS_PER_TASK='64',
                        QB_MPI_TYPE='pmix')
        self.context = contextlib.ExitStack()
        self.addCleanup(self.context.close)
        self.context.enter_context(patch.dict(os.environ, self.env))
        self.context.enter_context(patch.object(b.socket, 'gethostname', return_value='compute'))
        self.context.enter_context(patch.object(b.os, 'sched_getaffinity', return_value=set(range(64))))
        self.commands = []

    def solver(self, command, **kwargs):
        self.commands.append(command)
        folder = kwargs['cwd']
        slices = folder / 'radiation-hydro-slices'
        self.assertTrue(slices.is_dir(), 'Hydro CSV export must be enabled before solver startup')
        cfg = b.config(folder / 'run.ini')
        self.assertEqual(cfg['stop_step'], '19')
        self.assertEqual(cfg['disable_diagnostics'], 'on')
        self.assertEqual(cfg['disable_output'], 'on')
        self.assertEqual(cfg['disable_analytic'], 'on')
        self.assertNotIn('hard_dt', cfg)
        self.assertIn('--hpx:threads=64', command)
        self.assertEqual(kwargs['env']['OMP_THREAD_LIMIT'], '1')
        log = kwargs['stdout']
        for step in range(1, 21):
            log.write('{} {:.8e} 1e-1 2e-1 {:.8e}\n'.format(step, step * .1, step * .1))
        log.write('RADIATION_HYDRO_FINISHED {} t=2.00000000000000000e+00 requested={}\n'.format(
            cfg['problem'], cfg['stop_time']))
        (slices / 'slice-mock.csv').write_text('mock final output\n')
        (folder / 'radiation-conservation.csv').write_text('mock conservation\n')
        if cfg['unigrid'] == 'off':
            level = int(cfg['max_level'])
            blocks = 8**(level-1)
            (folder/'gradient-amr.csv').write_text('t,step,level,leaf_blocks,leaf_cells\n'
                '0,0,{},{},{}\n1,10,{},{},{}\n'.format(level-1,blocks,blocks*512,level-1,blocks,blocks*512))
        return subprocess.CompletedProcess(command, 0)

    def test_all_three_samples_and_fresh_output(self):
        with patch.object(b.subprocess, 'run', side_effect=self.solver), contextlib.redirect_stdout(io.StringIO()):
            report = b.benchmark(SOURCE, self.build, self.output, 4, b.FACTORS)
            self.assertEqual(report['status'], 'timing-samples-complete')
            self.assertEqual(len(self.commands), 3)
            for record in report['cases']:
                self.assertEqual(record['executed_steps'], 20)
                self.assertEqual(record['actual_physical_time_s'], 2)
                self.assertEqual(record['radiation_reduced_c_factor'], b.FACTORS[record['case']])
                self.assertGreater(record['estimated_full_duration_wall_seconds'], record['wall_seconds_including_startup'])
                self.assertIn('does not validate RSLA accuracy', record['scope'])
            with self.assertRaises(FileExistsError):
                b.benchmark(SOURCE, self.build, self.output, 4, b.FACTORS)
            self.assertEqual(len(self.commands), 3)

    def test_login_host_rejected_before_output(self):
        with patch.object(b.socket, 'gethostname', return_value='login'), patch.object(b.subprocess, 'run') as run:
            with self.assertRaises(ValueError):
                b.benchmark(SOURCE, self.build, self.output, 4, b.FACTORS)
            run.assert_not_called()
            self.assertFalse(self.output.exists())

    def test_amr_timing_uses_tuned_plan_and_records_ratio_misses(self):
        plan = json.loads((SOURCE/'radiation_results/amr/local-thresholds.json').read_text())
        with patch.object(b.subprocess, 'run', side_effect=self.solver), contextlib.redirect_stdout(io.StringIO()):
            report = b.benchmark(SOURCE, self.build, self.output, 4, b.FACTORS, plan)
        self.assertTrue(report['adaptive'])
        self.assertEqual(report['amr_plan'], plan)
        for case in report['cases']:
            cfg = case['configuration']
            row = next(row for row in plan['rows'] if row['case']==case['case'] and row['finest_level']==4)
            self.assertEqual(cfg['min_level'], '3')
            self.assertEqual(cfg['unigrid'], 'off')
            self.assertEqual(float(cfg['gradient_refine_threshold']), row['selected']['threshold'])
            self.assertEqual(case['amr_history']['samples_outside_target_ratio'], 2)

    def test_solver_failure_retains_metadata_and_stops(self):
        with patch.object(b.subprocess, 'run', side_effect=subprocess.CalledProcessError(17, ['mock'])) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                b.benchmark(SOURCE, self.build, self.output, 4, b.FACTORS)
            self.assertEqual(run.call_count, 1)
        report = json.loads((self.output / 'benchmark.json').read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['cases'][0]['status'], 'failed')
        self.assertIn('wall_seconds_including_startup', report['cases'][0])

    def test_incomplete_step_count_rejected(self):
        log = '1 0.1 0.1 0.2 0.1\nRADIATION_HYDRO_FINISHED RADIATION_BLAST_WAVE t=0.1 requested=22000\n'
        with self.assertRaises(ValueError):
            b.timing(log, 'RADIATION_BLAST_WAVE', 22000.)


if __name__ == '__main__':
    unittest.main()
