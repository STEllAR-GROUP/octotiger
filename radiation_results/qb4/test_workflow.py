#!/usr/bin/env python3
"""Offline contract tests: every scheduler invocation is mocked."""
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
from contextlib import redirect_stdout
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('qb_workflow', HERE/'workflow.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)

class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='qb offline test ')
        self.root = Path(self.temp.name)
        self.config = json.loads((HERE/'site.example.json').read_text())
        self.config.update(source_root=str(self.root/'source checkout'),
            build_dir=str(self.root/'build'), output_root=str(self.root/'results'),
            environment_file=str(self.root/'site environment.sh'), account='test_account',
            mpi_type='pmix', build_walltime='01:00:00', run_walltime='24:00:00',
            plot_walltime='01:00:00')
        Path(self.config['source_root']).mkdir()
        Path(self.config['environment_file']).write_text('# test environment\n')
        self.config_file = self.root/'site.json'
        self.config_file.write_text(json.dumps(self.config))
        with patch.object(w.subprocess, 'run', side_effect=AssertionError('No processes in prepare')), redirect_stdout(io.StringIO()):
            self.plan_path = w.prepare(self.config_file, self.root/'plan')
        self.plan = w.read(self.plan_path)

    def tearDown(self):
        self.temp.cleanup()

    def test_matrix_and_prepare(self):
        runs = [t for t in self.plan['tasks'] if t['kind'] == 'run']
        self.assertEqual(len(runs), 28)
        self.assertEqual({(t['case'], t['level']) for t in runs}, {(c,l) for c in w.CASES for l in w.LEVELS})
        wave = [t for t in runs if t['case'] == 'streaming_wave']
        self.assertEqual([(t['nodes'],t['cells']) for t in wave], [(1,32),(1,64),(2,128),(4,256)])
        self.assertTrue(all(t['time'] == 4 for t in runs if t['case'] not in w.TIMES))
        self.assertEqual([t['level'] for t in runs], sorted(t['level'] for t in runs))
        self.assertEqual([t['minimum_transport_steps'] for t in runs if t['case'] in w.TIMES and t['level']==2],
                         [12664,1233432,6298726])
        self.assertEqual(self.plan['tasks'][1]['kind'], 'tests')
        self.assertEqual(self.plan['tasks'][1]['cpus'], 4)
        for script in self.plan_path.parent.glob('*.sbatch'):
            subprocess.run(['bash','-n',str(script)],check=True)
        self.assertFalse(Path(self.config['output_root']).exists())

    def test_serial_dependencies_and_no_duplicate_submit(self):
        calls = []
        def sbatch(cmd, **kwargs):
            calls.append(cmd)
            job = str(1000+len(calls))
            return SimpleNamespace(stdout='sbatch: 55085.78 SUs available in test_account\n'
                'sbatch: 128.00 SUs estimated for this job.\n'
                'sbatch: lua: Submitted job '+job+'\n'+job+';mockcluster\n',
                stderr='mock informational stderr\n', returncode=0)
        with patch.object(w.subprocess,'run',side_effect=sbatch), redirect_stdout(io.StringIO()):
            w.submit(self.plan_path)
            with self.assertRaises(ValueError):
                w.submit(self.plan_path)
        self.assertEqual(len(calls),31)
        self.assertFalse(any(a.startswith('--dependency=') for a in calls[0]))
        for i, cmd in enumerate(calls[1:],1):
            self.assertIn('--dependency=afterok:'+str(1000+i),cmd)
        for task,cmd in zip(self.plan['tasks'],calls):
            self.assertIn('--nodes='+str(task['nodes']),cmd)
            self.assertIn('--ntasks='+str(task['nodes']),cmd)
            self.assertIn('--ntasks-per-node=1',cmd)
            self.assertIn('--cpus-per-task='+str(task['cpus']),cmd)
        self.assertIn('--partition=single',calls[0])
        self.assertIn('--partition=single',calls[-1])
        self.assertIn('--partition=single',calls[1])
        self.assertIn('--partition=workq',calls[2])
        saved = w.read(self.plan_path.parent/'submission.json')
        self.assertTrue(all(entry['stdout'].startswith('sbatch: 55085.78') and
                            entry['stderr'] == 'mock informational stderr\n' and
                            entry['returncode'] == 0 for entry in saved))

    def test_sbatch_parser_accepts_only_known_unambiguous_announcements(self):
        observed = ('sbatch: 55085.78 SUs available in loni_graphshpx\n'
                    'sbatch: 128.00 SUs estimated for this job.\n'
                    'sbatch: lua: Submitted job 1082916\n1082916')
        self.assertEqual(w.parse_sbatch_output(observed, 'loni_graphshpx'), '1082916')
        self.assertEqual(w.parse_sbatch_output('1082916\n'), '1082916')
        self.assertEqual(w.parse_sbatch_output('1082916;qb4-cluster\n'), '1082916')
        self.assertEqual(w.parse_sbatch_output(observed+';qb4\n'), '1082916')
        invalid = (
            '', '0', '1082916;', '1082916;bad cluster', '1082916;one;two',
            '1082916\n1082917\n', '1082916\n1082916\n',
            'sbatch: lua: Submitted job 1082915\n1082916\n',
            'sbatch: lua: Submitted job 1082916\n',
            'sbatch: warning: unknown plugin output\n1082916\n',
            'unrecognized output\n1082916\n', '1082916\ntrailing text\n',
            'sbatch: lua: Submitted job 1082916\nsbatch: lua: Submitted job 1082916\n1082916\n')
        for stdout in invalid:
            with self.subTest(stdout=stdout), self.assertRaises(RuntimeError):
                w.parse_sbatch_output(stdout)
        with self.assertRaises(RuntimeError):
            w.parse_sbatch_output(observed, 'different_account')

    def test_raw_output_saved_before_parsing_and_ambiguous_submission_stops(self):
        stdout, stderr = 'unexpected accounting text\n1082916\n', 'informational stderr\n'
        original_parser = w.parse_sbatch_output
        def inspect_evidence(value, account):
            pending = w.read(self.plan_path.parent/'submission.json')
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0]['status'], 'submitting')
            self.assertEqual(pending[0]['stdout'], stdout)
            self.assertEqual(pending[0]['stderr'], stderr)
            return original_parser(value, account)
        with patch.object(w.subprocess, 'run', return_value=SimpleNamespace(
                stdout=stdout, stderr=stderr, returncode=0)) as run, \
                patch.object(w, 'parse_sbatch_output', side_effect=inspect_evidence):
            with self.assertRaises(RuntimeError):
                w.submit(self.plan_path)
            self.assertEqual(run.call_count, 1)
            with self.assertRaises(ValueError):
                w.submit(self.plan_path)
            self.assertEqual(run.call_count, 1)
        saved = w.read(self.plan_path.parent/'submission.json')
        self.assertEqual(saved[0]['status'], 'submission-ambiguous')
        self.assertEqual(saved[0]['stdout'], stdout)
        self.assertEqual(saved[0]['stderr'], stderr)

    def test_failed_submission_stops_and_retains_record(self):
        calls = []
        def fail(cmd, **kwargs):
            calls.append(cmd)
            raise subprocess.CalledProcessError(1,cmd,output='',stderr='mock refusal')
        with patch.object(w.subprocess,'run',side_effect=fail):
            with self.assertRaises(subprocess.CalledProcessError):
                w.submit(self.plan_path)
        self.assertEqual(len(calls),1)
        self.assertEqual(w.read(self.plan_path.parent/'submission.json')[0]['status'],'submission-failed')

    def test_worker_launch_and_login_host_rejection(self):
        task = next(t for t in self.plan['tasks'] if t['key']=='radiating_blast-l5')
        env = {'SLURM_JOB_ID':'mock','SLURM_NNODES':'4','SLURM_NTASKS':'4',
               'SLURM_CPUS_PER_TASK':'64','SLURMD_NODENAME':'mocknode'}
        with patch.dict(os.environ,env), patch.object(w.socket,'gethostname',return_value='mocknode'), patch.object(w.subprocess,'run') as run:
            w.worker(self.plan_path,task['key'])
            cmd = run.call_args.args[0]
            self.assertEqual(cmd[1:4],['run','radiating_blast','5'])
            self.assertIn('--launcher',cmd)
            self.assertNotIn('--time',cmd)
            self.assertEqual(os.environ['OMP_NUM_THREADS'],'1')
        with patch.dict(os.environ,env), patch.object(w.socket,'gethostname',return_value='login'), patch.object(w.subprocess,'run') as run:
            with self.assertRaises(RuntimeError):
                w.worker(self.plan_path,task['key'])
            run.assert_not_called()

    def test_placeholder_cannot_submit(self):
        self.plan['config']['mpi_type']='EDIT_ME'
        w.write(self.plan_path,self.plan)
        with patch.object(w.subprocess,'run') as run:
            with self.assertRaises(ValueError):
                w.submit(self.plan_path)
            run.assert_not_called()

    def test_changed_plan_and_invalid_boolean_are_rejected(self):
        self.plan['tasks'][1]['nodes'] = 99
        w.write(self.plan_path, self.plan)
        with patch.object(w.subprocess, 'run') as run:
            with self.assertRaises(ValueError):
                w.submit(self.plan_path)
            run.assert_not_called()
        self.config['no_silo'] = 'false'
        with self.assertRaises(ValueError):
            w.validate(self.config)

    def test_per_case_reduced_speed_and_analytic_full_speed(self):
        self.config['reduced_c_factors'] = {'radiating_blast': .02, 'subcritical_shock': .001}
        w.validate(self.config)
        selected = {task['key']: task for task in w.tasks(self.config) if task['kind'] == 'run'}
        self.assertEqual(selected['radiating_blast-l2']['radiation_reduced_c_factor'], .02)
        self.assertEqual(selected['subcritical_shock-l2']['minimum_transport_steps'], 6299)
        self.assertEqual(selected['supercritical_shock-l2']['radiation_reduced_c_factor'], 1)
        self.assertTrue(all(selected[case+'-l2']['radiation_reduced_c_factor'] == 1 for case in w.CASES[:4]))
        self.config['reduced_c_factors']['streaming_wave'] = .1
        with self.assertRaises(ValueError):
            w.validate(self.config)

    def test_mpi_launcher_uses_mock_only(self):
        binary = self.root/'mock-bin'
        binary.mkdir()
        fake = binary/'srun'
        fake.write_text('#!/usr/bin/env python3\nimport json, pathlib,sys\n'
            'pathlib.Path("arguments.json").write_text(json.dumps(sys.argv[1:]))\n'
            'pathlib.Path("mpi-logs/task-0.out").write_text("normalized constants\\n1 1 1 1\\n| c = 2.99792458e10 |\\n")\n'
            'pathlib.Path("mpi-logs/task-0.err").write_text("root stderr\\n")\n')
        fake.chmod(0o755)
        env = dict(os.environ, PATH=str(binary)+':'+os.environ['PATH'], SLURM_JOB_ID='mock',
                   SLURM_NNODES='4',SLURM_NTASKS='4',SLURM_CPUS_PER_TASK='64',QB_MPI_TYPE='pmix',
                   SLURMD_NODENAME=socket.gethostname().split('.')[0])
        result = subprocess.run(['bash',str(HERE/'mpi-launch.sh'),'/bin/true','--config_file=mock.ini','--hpx:threads=64'],
                                cwd=self.root,env=env,text=True,capture_output=True,check=True)
        argv=json.loads((self.root/'arguments.json').read_text())
        self.assertEqual(argv.count('--hpx:threads=64'),1)
        self.assertIn('--hpx:localities=4',argv)
        self.assertIn('--mpi=pmix',argv)
        self.assertIn('--hpx:ini=hpx.parcel.bootstrap=mpi',argv)
        self.assertIn('--hpx:ini=hpx.parcel.mpi.enable=1',argv)
        self.assertIn('--hpx:ini=hpx.parcel.tcp.enable!=0',argv)
        self.assertNotIn('--hpx:ini=hpx.parcel.tcp.enable=0',argv)
        self.assertNotIn('--label',argv)
        self.assertIn('\nnormalized constants\n1 1 1 1\n',result.stdout)
        self.assertEqual(result.stderr,'root stderr\n')

    def adaptive_plan(self):
        sample = w.read(HERE/'site.amr.example.json')
        for field in ('amr_plan', 'run_selection', 'timing_samples', 'timing_walltime', 'test_walltime',
                      'checkpoint_odt', 'reduced_c_factors', 'snapshots'):
            self.config[field] = sample[field]
        self.config['amr_plan'] = str(HERE.parent/'amr/local-thresholds.json')
        self.config_file.write_text(json.dumps(self.config))
        with patch.object(w.subprocess, 'run', side_effect=AssertionError('Offline only')), redirect_stdout(io.StringIO()):
            path = w.prepare(self.config_file, self.root/'adaptive-plan')
        return path, w.load_plan(path)

    def test_phased_adaptive_plan_conversion_factors_and_freeze(self):
        path, plan = self.adaptive_plan()
        self.assertEqual(len(plan['tasks']), 26)
        self.assertEqual(plan['matrix_scope'], 'selected-subset')
        self.assertEqual(plan['selected_runs'], 20)
        self.assertEqual([t['kind'] for t in plan['tasks'][:6]], ['build', 'tests']+['timing']*4)
        self.assertFalse(any(t['kind'] == 'collect' for t in plan['tasks']))
        runs = {t['key']: t for t in plan['tasks'] if t['kind']=='run'}
        sphere = runs['equilibrium_sphere-l4']['amr']
        self.assertEqual(sphere['min_level'], 3)
        self.assertEqual(sphere['gradient_refine_threshold'], sphere['local_threshold']/3e10)
        self.assertEqual(sphere['gradient_refine_floor'], 1e-30)
        self.assertEqual(runs['subcritical_shock-l2']['radiation_reduced_c_factor'], .03)
        self.assertEqual(runs['supercritical_shock-l2']['radiation_reduced_c_factor'], .1)
        blast = runs['radiating_blast-l3']['amr']
        self.assertEqual(blast['gradient_refine_threshold'], blast['local_threshold'])
        plan['amr_plan']['rows'][0]['selected']['threshold'] *= 2
        w.write(path, plan)
        with self.assertRaises(ValueError):
            w.load_plan(path)

    def test_adaptive_worker_passes_exact_plan_settings(self):
        path, plan = self.adaptive_plan()
        env = dict(SLURM_JOB_ID='mock',SLURM_NNODES='2',SLURM_NTASKS='2',
                   SLURM_CPUS_PER_TASK='64',SLURMD_NODENAME='mocknode')
        with patch.dict(os.environ,env), patch.object(w.socket,'gethostname',return_value='mocknode'), \
                patch.object(w.subprocess,'run') as run:
            w.worker(path, 'equilibrium_sphere-l4')
        cmd = run.call_args.args[0]
        expected = next(t for t in plan['tasks'] if t['key']=='equilibrium_sphere-l4')['amr']
        self.assertEqual(cmd[cmd.index('--gradient-refine-threshold')+1], str(expected['gradient_refine_threshold']))
        self.assertEqual(cmd[cmd.index('--min-level')+1], '3')
        state = w.read(Path(self.config['output_root'])/'workflow-status/equilibrium_sphere-l4.json')
        self.assertEqual(state['status'], 'complete')
        self.assertEqual(state['scope'], 'selected-subset')

    def test_test_gate_and_failure_status(self):
        build = Path(self.config['build_dir'])
        build.mkdir()
        (build/'CMakeCache.txt').write_text('_Python3_EXECUTABLE:INTERNAL=/mock/python3.11\n')
        env = dict(SLURM_JOB_ID='mock',SLURM_NNODES='1',SLURM_NTASKS='1',
                   SLURM_CPUS_PER_TASK='4',SLURMD_NODENAME='mocknode')
        with patch.dict(os.environ,env), patch.object(w.socket,'gethostname',return_value='mocknode'), \
                patch.object(w.subprocess,'run') as run:
            w.worker(self.plan_path, 'tests')
            self.assertEqual(run.call_count, 3)
            self.assertEqual(run.call_args_list[0].args[0][0], '/mock/python3.11')
            self.assertTrue(all(call.args[0][-2:]==['-j','1'] for call in run.call_args_list[1:]))
            solver_command = run.call_args_list[1].args[0]
            self.assertEqual(solver_command[solver_command.index('-L')+1], 'radiation|gradient')
            reader_command = run.call_args_list[2].args[0]
            self.assertEqual(reader_command[reader_command.index('-R')+1], '^radiation-results-')
        with patch.dict(os.environ,env), patch.object(w.socket,'gethostname',return_value='mocknode'), \
                patch.object(w.subprocess,'run',side_effect=subprocess.CalledProcessError(1,['mock'])):
            with self.assertRaises(subprocess.CalledProcessError):
                w.worker(self.plan_path, 'tests')
        state = w.read(Path(self.config['output_root'])/'workflow-status/tests.json')
        self.assertEqual(state['status'], 'failed')

    def test_selection_and_cadence_validation(self):
        for selection in ([], [{'case':'missing','level':2}], [{'case':'streaming_wave','level':6}],
                          [{'case':'streaming_wave','level':2}]*2):
            self.config['run_selection']=selection
            with self.assertRaises(ValueError):
                w.validate(self.config)
        self.config.pop('run_selection')
        self.config['checkpoint_odt']={'radiating_blast-l2':17.5}
        w.validate(self.config)
        task = next(t for t in w.tasks(self.config) if t['key']=='radiating_blast-l2')
        self.assertEqual(task['checkpoint_odt'],17.5)
        self.config['checkpoint_odt']['missing']=1
        with self.assertRaises(ValueError):
            w.validate(self.config)

if __name__=='__main__':
    unittest.main()
