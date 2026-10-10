#!/usr/bin/env python3
"""Local collector tests: all solver/results commands are mocked."""
import copy
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import collect as c
import workflow as w


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='qb4-collect-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / 'results'
        self.output.mkdir()
        self.build = self.root / 'build'
        self.runner = self.build / 'radiation-results-tools/radiation-results'
        self.runner.parent.mkdir(parents=True)
        self.runner.write_text('mock results tool\n')
        self.runner.chmod(0o755)
        (self.build / 'octotiger').write_text('mock solver, never executed\n')
        self.executable_hash = c.sha256(self.build / 'octotiger')
        config = w.read(Path(__file__).parent / 'site.example.json')
        config.update(source_root=str(self.root), build_dir=str(self.build),
                      output_root=str(self.output), environment_file=str(self.root / 'environment.sh'))
        self.plan = {'config': config, 'tasks': w.tasks(config), 'directory': str(self.root)}
        self.plan_path = self.root / 'plan.json'
        c.write_json(self.plan_path, self.plan)
        self.runs = []
        for task in self.plan['tasks']:
            if task['kind'] != 'run':
                continue
            case, level = task['case'], task['level']
            batch = self.output / 'batches' / task['key']
            folder = batch / case / f'l{level}'
            folder.mkdir(parents=True)
            self.runs.append(folder)
            length = 2e13 if case == 'radiating_blast' else 7e10 if case in w.TIMES else 6e10
            template = Path(__file__).parent.parent / 'configs' / f'{case}.ini'
            problem = c.read_ini(template)['problem']
            cfg = {'problem': problem, 'min_level': str(level),
                   'max_level': str(level), 'stop_time': str(task['time']),
                   'code_to_g': '1', 'code_to_cm': '1', 'code_to_s': '1', 'xscale': str(length / 2),
                   'radiation_reduced_c_factor': '1'}
            meta = {'status': 'complete', 'case': case, 'level': level, 'cells': task['cells'],
                    'inx': 8, 'time': task['time'], 'c': c.C, 'units': c.UNITS,
                    'radiation_reduced_c_factor': 1, 'physical_c': c.C, 'transport_c': c.C,
                    'origin': c.ORIGIN, 'executable_sha256': self.executable_hash,
                    'length': length, 'dx': length / task['cells'], 'config': cfg}
            if case in w.TIMES:
                meta.update(reference_kind='numerical-only', benchmark_time=task['time'])
            if case == 'gaussian_pulse':
                (folder / 'reference.bin').write_bytes(b'mock reference')
                meta['reference_sha256'] = c.sha256(folder / 'reference.bin')
            c.write_json(folder / 'run.json', meta)
            (folder / 'run.ini').write_text(''.join(f'{key}={value}\n' for key, value in cfg.items()))
            (folder / 'run.log').write_text('mock application log\n')
            for norm in ('L1', 'L2', 'Linf'):
                (folder / (norm + '.dat')).write_text('mock norms\n')
            header = 't,volume,er,fx,fy,fz,er_boundary,fx_boundary,fy_boundary,fz_boundary,er_source,fx_source,fy_source,fz_source\n'
            (folder / 'radiation-conservation.csv').write_text(
                header + '0,1,1,0,0,0,0,0,0,0,0,0,0,0\n' +
                f'{task["time"]},1,1,0,0,0,0,0,0,0,0,0,0,0\n')
            slices = folder / ('radiation-hydro-slices' if case in w.TIMES else 'radiation-slices')
            slices.mkdir()
            (slices / 'slice-0.csv').write_text('mock numerical data\n')
            (folder / 'final.silo').write_bytes(b'large binary is never collected')
            c.write_json(batch / 'batch.json', {
                'cases': [case], 'levels': [level], 'units': c.UNITS, 'origin': c.ORIGIN,
                'executable_sha256': self.executable_hash,
                'source_sha256': {'octotiger/radiation/m1.hpp': 'a' * 64}})
        self.commands = []

    def configure_factors(self, factors):
        self.plan['config']['reduced_c_factors'] = factors
        self.plan['tasks'] = w.tasks(self.plan['config'])
        c.write_json(self.plan_path, self.plan)
        for folder in self.runs:
            meta = c.read_json(folder / 'run.json')
            factor = factors.get(meta['case'], 1)
            meta.update(radiation_reduced_c_factor=factor, transport_c=factor * c.C)
            meta['config']['radiation_reduced_c_factor'] = str(factor)
            c.write_json(folder / 'run.json', meta)
            (folder / 'run.ini').write_text(''.join(
                f'{key}={value}\n' for key, value in meta['config'].items()))

    def results(self, command, **kwargs):
        self.commands.append(command)
        self.assertEqual(kwargs['env']['OMP_THREAD_LIMIT'], '1')
        if command[1] == 'plot':
            stage = Path(command[2])
            (stage / 'index.html').write_text(''.join(f'<a href="{case}.html">{case}</a>' for case in w.CASES))
            for case in w.CASES:
                (stage / f'{case}.html').write_text(
                    '<a href="index.html">Home</a><a href="https://doi.org/10.1086/173889">Source</a>'
                    f'<a href="{case}/l2/run.json">Metadata</a>')
            (stage / 'unportable.gnuplot').write_text('references the temporary staging directory')
        return subprocess.CompletedProcess(command, 0)

    def test_complete_matrix_portable_files_and_exact_metadata(self):
        original = (self.runs[0] / 'run.json').read_bytes()
        with patch.object(c.subprocess, 'run', side_effect=self.results):
            site = c.collect(self.plan_path)
        self.assertEqual([cmd[1] for cmd in self.commands], ['check', 'plot'])
        self.assertEqual((site / 'streaming_wave/l2/run.json').read_bytes(), original)
        self.assertEqual(len(list((site / 'provenance').glob('*-batch.json'))), 28)
        self.assertFalse(list(site.rglob('*.silo')))
        self.assertFalse(list(site.rglob('*.bin')))
        self.assertFalse(list(site.rglob('*.gnuplot')))
        self.assertFalse(any(path.is_symlink() for path in site.rglob('*')))
        manifest = c.read_json(site / 'provenance/collection.json')
        self.assertEqual(manifest['run_count'], 28)
        self.assertEqual(manifest['status'], 'complete')
        self.assertEqual(c.read_json(site / 'batch.json')['source_sha256'],
                         {'octotiger/radiation/m1.hpp': 'a' * 64})
        # The static site is independent of subsequent changes to raw files.
        (self.runs[0] / 'run.json').write_text('changed raw metadata')
        self.assertEqual((site / 'streaming_wave/l2/run.json').read_bytes(), original)

    def test_incomplete_or_wrong_target_never_calls_results(self):
        for field, value in (('status', 'failed'), ('time', 0.01), ('cells', 16), ('origin', 'fixture')):
            with self.subTest(field=field):
                path = self.runs[0] / 'run.json'
                original = c.read_json(path)
                changed = copy.deepcopy(original)
                changed[field] = value
                c.write_json(path, changed)
                with patch.object(c.subprocess, 'run') as run:
                    with self.assertRaises(ValueError):
                        c.collect(self.plan_path)
                    run.assert_not_called()
                self.assertFalse((self.output / 'site').exists())
                c.write_json(path, original)

    def test_distinct_coupled_factors_preserve_exact_metadata(self):
        self.configure_factors({'subcritical_shock': 0.03, 'supercritical_shock': 0.1,
                                'radiating_blast': 1})
        with patch.object(c.subprocess, 'run', side_effect=self.results):
            site = c.collect(self.plan_path)
        for folder in self.runs:
            target = site / folder.parent.name / folder.name / 'run.json'
            self.assertEqual(target.read_bytes(), (folder / 'run.json').read_bytes())

    def configure_adaptive(self):
        document = w.read(Path(__file__).parent.parent/'amr/local-thresholds.json')
        self.plan['config']['amr_plan'] = 'radiation_results/amr/local-thresholds.json'
        self.plan['amr_plan'] = document
        self.plan['amr_plan_sha256'] = w.fingerprint(document)
        self.plan['tasks'] = w.tasks(self.plan['config'], document)
        c.write_json(self.plan_path, self.plan)
        selected = {t['key']:t for t in self.plan['tasks'] if t['kind']=='run'}
        for folder in self.runs:
            meta = c.read_json(folder/'run.json')
            level = meta['level']
            amr = selected[meta['case']+'-l'+str(level)]['amr']
            meta.update(adaptive=True, min_level=level-1, max_level=level)
            meta['config'].update({key:str(value) for key,value in amr.items()
                if key in ('min_level','max_level') or key.startswith('gradient_refine_')})
            meta['config']['unigrid']='off'
            # One refined parent: at L2 this intentionally misses R<=1.
            coarse, fine = 8**(level-1)-1, 8
            histogram = folder/'gradient-amr.csv'
            histogram.write_text('t,step,level,leaf_blocks,leaf_cells\n' + ''.join(
                f'{time},{step},{lev},{count},{count*512}\n'
                for time,step in ((0,0),(meta['time']/2,5)) for lev,count in ((level-1,coarse),(level,fine))))
            meta['mesh']={'full_mesh':c.amr_history(histogram,level,meta['time'])['full_mesh']}
            c.write_json(folder/'run.json',meta)
            (folder/'run.ini').write_text(''.join(f'{key}={value}\n' for key,value in meta['config'].items()))

    def test_adaptive_collection_preserves_native_history_and_ratio_misses(self):
        self.configure_adaptive()
        with patch.object(c.subprocess,'run',side_effect=self.results):
            site = c.collect(self.plan_path)
        records = c.read_json(site/'provenance/collection.json')['runs']
        self.assertEqual(records[0]['amr_history']['samples_outside_target_ratio'],2)
        self.assertEqual(records[0]['amr_history']['last_fine_to_coarse_cell_ratio'],8/7)
        self.assertEqual((site/'streaming_wave/l2/gradient-amr.csv').read_bytes(),
                         (self.runs[0]/'gradient-amr.csv').read_bytes())
        self.assertEqual(c.read_json(site/'provenance/plan.json')['amr_plan'],self.plan['amr_plan'])

    def test_adaptive_wrong_threshold_counts_or_metadata_reject_before_tools(self):
        self.configure_adaptive()
        folder = self.runs[0]
        histogram = folder/'gradient-amr.csv'
        original = histogram.read_text()
        bad = original.replace(',7,3584',',6,3072')
        histogram.write_text(bad)
        with patch.object(c.subprocess,'run') as run:
            with self.assertRaises(ValueError):
                c.collect(self.plan_path)
            run.assert_not_called()
        histogram.write_text(original)
        meta = c.read_json(folder/'run.json')
        meta['config']['gradient_refine_threshold']='999'
        c.write_json(folder/'run.json',meta)
        (folder/'run.ini').write_text(''.join(f'{key}={value}\n' for key,value in meta['config'].items()))
        with patch.object(c.subprocess,'run') as run:
            with self.assertRaises(ValueError):
                c.collect(self.plan_path)
            run.assert_not_called()

    def test_partial_selection_cannot_publish_complete_matrix(self):
        self.plan['config']['run_selection']=[dict(case='streaming_wave',level=2)]
        self.plan['tasks']=w.tasks(self.plan['config'])
        c.write_json(self.plan_path,self.plan)
        with patch.object(c.subprocess,'run') as run:
            with self.assertRaises(ValueError):
                c.collect(self.plan_path)
            run.assert_not_called()

    def test_speed_or_factor_mismatch_never_calls_results(self):
        self.configure_factors({'subcritical_shock': 0.03})
        folder = next(folder for folder in self.runs if folder.parent.name == 'subcritical_shock')
        path = folder / 'run.json'
        original = c.read_json(path)
        for field, value in (('radiation_reduced_c_factor', 1), ('physical_c', c.C * 0.03),
                             ('transport_c', c.C), ('transport_c', None)):
            with self.subTest(field=field, value=value):
                changed = copy.deepcopy(original)
                if value is None:
                    del changed[field]
                else:
                    changed[field] = value
                c.write_json(path, changed)
                with patch.object(c.subprocess, 'run') as run:
                    with self.assertRaises(ValueError):
                        c.collect(self.plan_path)
                    run.assert_not_called()
                self.assertFalse((self.output / 'site').exists())
        c.write_json(path, original)
        original['config']['radiation_reduced_c_factor'] = '1'
        c.write_json(path, original)
        (folder / 'run.ini').write_text(''.join(
            f'{key}={value}\n' for key, value in original['config'].items()))
        with patch.object(c.subprocess, 'run') as run:
            with self.assertRaises(ValueError):
                c.collect(self.plan_path)
            run.assert_not_called()

    def test_analytic_case_cannot_use_reduced_light_speed(self):
        self.configure_factors({'streaming_wave': 0.1})
        with patch.object(c.subprocess, 'run') as run:
            with self.assertRaises(ValueError):
                c.collect(self.plan_path)
            run.assert_not_called()

    def test_missing_case_or_early_conservation_is_rejected(self):
        path = self.runs[0] / 'radiation-conservation.csv'
        text = path.read_text()
        path.write_text(text.replace('4.0,1,1', '0.1,1,1'))
        with patch.object(c.subprocess, 'run') as run:
            with self.assertRaises(ValueError):
                c.collect(self.plan_path)
            run.assert_not_called()
        path.write_text(text)
        self.plan['tasks'].pop(1)
        c.write_json(self.plan_path, self.plan)
        with self.assertRaises(ValueError):
            c.collect(self.plan_path)

    def test_source_changes_and_bad_reference_are_rejected(self):
        batch = self.runs[0].parent.parent / 'batch.json'
        original = c.read_json(batch)
        changed = copy.deepcopy(original)
        changed['source_sha256']['octotiger/radiation/m1.hpp'] = 'b' * 64
        c.write_json(batch, changed)
        with self.assertRaises(ValueError):
            c.collect(self.plan_path)
        c.write_json(batch, original)
        reference = next(self.output.glob('batches/gaussian_pulse-l2/**/reference.bin'))
        reference.write_bytes(b'changed reference')
        with self.assertRaises(ValueError):
            c.collect(self.plan_path)

    def test_renderer_failure_or_broken_link_never_publishes(self):
        with patch.object(c.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, ['mock'])):
            with self.assertRaises(subprocess.CalledProcessError):
                c.collect(self.plan_path)
        self.assertFalse((self.output / 'site').exists())
        def bad_link(command, **kwargs):
            result = self.results(command, **kwargs)
            if command[1] == 'plot':
                (Path(command[2]) / 'index.html').write_text('<a href="missing.csv">Missing</a>')
            return result
        with patch.object(c.subprocess, 'run', side_effect=bad_link):
            with self.assertRaises(ValueError):
                c.collect(self.plan_path)
        self.assertFalse((self.output / 'site').exists())
        self.assertFalse(list(self.output.glob('.site-stage-*')))

    def test_existing_site_is_never_overwritten(self):
        site = self.output / 'site'
        site.mkdir()
        (site / 'index.html').write_text('previous complete publication')
        with patch.object(c.subprocess, 'run') as run:
            with self.assertRaises(ValueError):
                c.collect(self.plan_path)
            run.assert_not_called()
        self.assertEqual((site / 'index.html').read_text(), 'previous complete publication')

    def test_external_assets_and_escaping_links_are_rejected(self):
        for url in ('../outside.dat', '/absolute.dat', 'file:///tmp/file',
                    'https://example.com/asset.png', '#missing-anchor'):
            with self.subTest(url=url):
                def bad_asset(command, **kwargs):
                    result = self.results(command, **kwargs)
                    if command[1] == 'plot':
                        (Path(command[2]) / 'index.html').write_text(f'<img src="{url}">')
                    return result
                with patch.object(c.subprocess, 'run', side_effect=bad_asset):
                    with self.assertRaises(ValueError):
                        c.collect(self.plan_path)
                self.assertFalse((self.output / 'site').exists())


if __name__ == '__main__':
    unittest.main()
