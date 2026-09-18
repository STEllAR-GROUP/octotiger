"""Synthetic fixtures validate budget accounting; they are not solver results."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from results import CONSERVATION_COLUMNS, CONSERVATION_FILE, FIELDS, NORMS, read_conservation
import plot


class ConservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.batch = Path(self.temp.name)
        self.folder = self.batch/'streaming_wave'/'l0'
        self.folder.mkdir(parents=True)
        self.meta = dict(status='complete', case='streaming_wave', level=0, cells=4,
            dx=.5, length=2., time=1., c=2., background=0., origin='synthetic test fixture',
            units={'system': 'CGS'})
        (self.folder/'run.json').write_text(json.dumps(self.meta))
        self.path = self.folder/CONSERVATION_FILE

    def row(self, time, energy=10., **values):
        row = dict.fromkeys(CONSERVATION_COLUMNS, 0.)
        row.update(t=time, volume=8., er=energy, **values)
        return row

    def write(self, rows, columns=CONSERVATION_COLUMNS):
        with self.path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader(); writer.writerows(rows)

    def read(self):
        return read_conservation(self.folder, self.meta)

    def test_balanced_sources_and_boundary_are_not_drift(self):
        self.write([self.row(0), self.row(1, 13., er_boundary=2., er_source=5.,
                                       fx=-3., fx_boundary=1., fx_source=-2.)])
        result = self.read()
        for field in FIELDS:
            np.testing.assert_array_equal(result['residuals'][field], 0)
            np.testing.assert_array_equal(result['normalized_residuals'][field], 0)
        energy = result['summary'][0]
        self.assertEqual(energy['raw_change'], 3.)
        self.assertEqual(energy['boundary'], 2.)
        self.assertEqual(energy['source'], 5.)
        self.assertEqual(energy['integral_units'], 'erg')
        self.assertEqual(result['summary'][1]['integral_units'], 'erg cm/s')
        json.dumps(result['summary'], allow_nan=False)

    def test_flux_normalization_does_not_divide_by_zero_net_flux(self):
        self.write([self.row(0), self.row(.5, fx=2.), self.row(1, fx=1.)])
        result = self.read()
        self.assertEqual(result['summary'][1]['initial'], 0)
        self.assertEqual(result['summary'][1]['normalized_error'], .05)
        self.assertEqual(result['summary'][1]['max_normalized_error'], .1)
        self.assertEqual(result['summary'][1]['normalization_scale'], 20.)

    def test_small_physical_units_have_no_absolute_floor(self):
        self.write([self.row(0, 1.e-30), self.row(1, 1.e-30, fx=2.e-31)])
        self.assertAlmostEqual(self.read()['summary'][1]['normalized_error'], .1)

    def test_all_zero_state(self):
        self.write([self.row(0, 0.), self.row(1, 0.)])
        result = self.read()
        for record in result['summary']:
            self.assertEqual(record['normalization_scale'], 0.)
            self.assertEqual(record['normalized_error'], 0.)
            self.assertEqual(record['max_normalized_error'], 0.)

    def test_energy_and_flux_drift_are_reported(self):
        self.write([self.row(0), self.row(1, 11., fy=-2.)])
        result = self.read()
        self.assertEqual(result['summary'][0]['residual'], 1.)
        self.assertAlmostEqual(result['summary'][0]['normalized_error'], 1/11)
        self.assertEqual(result['summary'][2]['residual'], -2.)
        self.assertAlmostEqual(result['summary'][2]['normalized_error'], 2/22)

    def test_missing_is_optional_but_present_invalid_is_rejected(self):
        self.assertIsNone(self.read())
        bad_rows = (
            [],
            [self.row(0)],  # final time does not match
            [self.row(.1), self.row(1)],
            [self.row(0), self.row(.5), self.row(.5), self.row(1)],
            [self.row(0), self.row(1, float('nan'))],
            [self.row(0), self.row(1, fx=float('inf'))],
            [self.row(0, er_source=1.), self.row(1)],
            [self.row(0, fx_boundary=1.), self.row(1)],
            [self.row(0), dict(self.row(1), volume=7.)],
        )
        for rows in bad_rows:
            with self.subTest(rows=rows):
                self.write(rows)
                with self.assertRaisesRegex(ValueError, CONSERVATION_FILE): self.read()
        self.path.write_text('t,er\n0,10\n1,10\n')
        with self.assertRaises(ValueError): self.read()
        self.path.write_text(','.join(CONSERVATION_COLUMNS)+'\n0,8\n')
        with self.assertRaises(ValueError): self.read()

    def stub_plot_run(self, folder, meta, target):
        target.mkdir(parents=True, exist_ok=True)
        for stem in ('profiles', *(f'slice_{field}' for field in FIELDS)):
            for suffix in ('.png', '.pdf'):
                (target/(stem+suffix)).write_text('synthetic fixture')
        (target/'slice.csv').write_text('synthetic fixture')
        (target/'slice.units.json').write_text('{}')

    def stub_save(self, fig, path):
        for suffix in ('.png', '.pdf'):
            path.with_suffix(suffix).write_text('synthetic fixture')
        plot.plt.close(fig)

    def stub_convergence(self, runs, target):
        target.mkdir(parents=True, exist_ok=True)
        for suffix in ('.png', '.pdf'):
            (target/'convergence').with_suffix(suffix).write_text('synthetic fixture')

    def renderer(self):
        norms = {field: dict.fromkeys(NORMS, .01) for field in FIELDS}
        patches = (mock.patch.object(plot, 'read_norms', return_value=norms),
                   mock.patch.object(plot, 'convergence', side_effect=self.stub_convergence),
                   mock.patch.object(plot, 'plot_run', side_effect=self.stub_plot_run),
                   mock.patch.object(plot, 'save', side_effect=self.stub_save))
        for patch in patches:
            patch.start(); self.addCleanup(patch.stop)
        return patches

    def test_convergence_requires_two_complete_resolutions(self):
        self.renderer()
        self.write([self.row(0), self.row(1)])
        page_path = plot.render(self.batch)
        convergence = page_path.parent/'streaming_wave'/'convergence.png'
        self.assertFalse(convergence.exists())
        self.assertEqual(plot.convergence.call_count, 0)
        self.assertIn('Convergence becomes available after two resolutions are complete.',
                      page_path.read_text())

        fine = self.batch/'streaming_wave'/'l1'
        fine.mkdir()
        fine_meta = dict(self.meta, level=1, cells=8, dx=.25)
        (fine/'run.json').write_text(json.dumps(fine_meta))
        plot.render(self.batch)
        self.assertTrue(convergence.is_file())
        self.assertEqual(plot.convergence.call_count, 1)
        self.assertIn('streaming_wave/convergence.png', page_path.read_text())

        fine_meta['status'] = 'running'
        (fine/'run.json').write_text(json.dumps(fine_meta))
        plot.render(self.batch)
        self.assertFalse(convergence.exists())
        self.assertNotIn('streaming_wave/convergence.png', page_path.read_text())

    def test_webpage_exports_cache_changes_and_missing_input(self):
        self.renderer()
        self.write([self.row(0), self.row(1)])
        page_path = plot.render(self.batch)
        page = page_path.read_text()
        target = page_path.parent/'streaming_wave'/'l0'
        self.assertIn('Radiation conservation', page)
        self.assertIn('Radiation momentum', page)
        self.assertIn('streaming_wave/l0/conservation.csv', page)
        self.assertIn('Max |R|/scale', page)
        self.assertEqual((target/'conservation.csv').read_text(), self.path.read_text())
        self.assertEqual(len(json.loads((page_path.parent/'conservation.json').read_text())), 4)
        self.assertEqual(len(json.loads((target/'conservation-summary.json').read_text())), 4)
        cached = json.loads((target/'plot-inputs.json').read_text())
        self.assertIsNotNone(cached['conservation'])
        self.assertIn('reader', cached)
        # An unchanged input reuses its figures.
        before = plot.plot_run.call_count
        plot.render(self.batch)
        self.assertEqual(plot.plot_run.call_count, before)
        # Input changes invalidate the cache and update measured summaries.
        self.write([self.row(0), self.row(1, 12.)])
        plot.render(self.batch)
        self.assertEqual(plot.plot_run.call_count, before+1)
        summary = json.loads((target/'conservation-summary.json').read_text())
        self.assertEqual(summary[0]['residual'], 2.)
        # Removed input produces a rerun notice and removes all old diagnostics.
        self.path.unlink()
        plot.render(self.batch)
        self.assertIn('Conservation diagnostics unavailable; rerun this test.', page_path.read_text())
        self.assertNotIn('streaming_wave/l0/conservation.png', page_path.read_text())
        self.assertEqual(json.loads((page_path.parent/'conservation.json').read_text()), [])
        for name in plot.CONSERVATION_PRODUCTS:
            self.assertFalse((target/name).exists(), name)
        # An old run gains diagnostics as soon as a new input appears.
        self.write([self.row(0), self.row(1)])
        plot.render(self.batch)
        self.assertTrue((target/'conservation.png').is_file())
        self.assertNotIn('Conservation diagnostics unavailable', page_path.read_text())

    def test_cached_run_still_validates_present_diagnostics(self):
        self.renderer()
        self.write([self.row(0), self.row(1)])
        plot.render(self.batch)
        self.write([self.row(0), self.row(.5)])
        with self.assertRaisesRegex(ValueError, 'final sample time'):
            plot.render(self.batch)


if __name__ == '__main__':
    unittest.main()
