#!/usr/bin/env python3
"""Independent Fourier checks of the serial analytic reference generator.

This runs only gen_radiation_reference, never the OctoTiger solver. All transforms
below use direct sums, independent of the generator's FFTW implementation.
"""
import argparse
import cmath
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

GENERATOR = os.environ.get('RADIATION_REFERENCE_GENERATOR')


class DiffusionReference(unittest.TestCase):
    def setUp(self):
        if not GENERATOR:
            self.skipTest('pass --generator or set RADIATION_REFERENCE_GENERATOR')
        self.temporary = tempfile.TemporaryDirectory(prefix='fld-analytic-reference-')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.n = 8
        self.length, self.c, self.chi, self.time = 2., 1., 3., .4

    def generate(self, model, name=None, time=None, chi=None):
        path = self.directory / (name or (model or 'default'))
        arguments = [str(GENERATOR), '--output', str(path), '--cells', str(self.n),
            '--length', str(self.length), '--c', str(self.c), '--chi', str(self.chi if chi is None else chi),
            '--width', '.3', '--background', '1', '--amplitude', '.02',
            '--time', str(self.time if time is None else time)]
        if model is not None:
            arguments.extend(('--model', model))
        result = subprocess.run(arguments, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        content = path.read_bytes()
        self.assertEqual(content[:8], b'OTRAD001')
        self.assertEqual(struct.unpack_from('<I', content, 8)[0], self.n)
        self.assertEqual(len(content), 76 + 64 * self.n**3)
        parameters = struct.unpack_from('<8d', content, 12)
        self.assertEqual(parameters[:3], (self.length, self.c, self.chi if chi is None else chi))
        values = struct.unpack_from('<' + str(8 * self.n**3) + 'd', content, 76)
        self.assertTrue(all(math.isfinite(value) for value in values))
        return content, values

    def mode(self, values, snapshot, field, wave):
        n = self.n
        offset = (4 * snapshot + field) * n**3
        terms = []
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    phase = -2j * math.pi * (wave[0] * i + wave[1] * j + wave[2] * k) / n
                    terms.append(values[offset + k + n * (j + n * i)] * cmath.exp(phase))
        return complex(math.fsum(value.real for value in terms),
                       math.fsum(value.imag for value in terms)) / n**3

    def test_fld_solves_heat_equation_and_derived_flux_for_signed_modes(self):
        _, values = self.generate('fld')
        diffusion = self.c / (3 * self.chi)
        for wave in ((1, 0, 0), (0, -1, 1), (1, -2, 0), (-1, 1, -1)):
            with self.subTest(wave=wave):
                wavevector = tuple(2 * math.pi * component / self.length for component in wave)
                k2 = sum(component**2 for component in wavevector)
                initial = self.mode(values, 0, 0, wave)
                self.assertGreater(abs(initial), 1e-8)
                final = self.mode(values, 1, 0, wave)
                self.assertLess(abs(final - initial * math.exp(-diffusion * k2 * self.time)), 3e-14)
                for snapshot in (0, 1):
                    energy = self.mode(values, snapshot, 0, wave)
                    for dimension in range(3):
                        flux = self.mode(values, snapshot, dimension + 1, wave)
                        self.assertLess(abs(flux + 1j * diffusion * wavevector[dimension] * energy), 3e-14)

    def test_mean_energy_and_zero_net_flux_are_preserved(self):
        _, values = self.generate('fld')
        initial = self.mode(values, 0, 0, (0, 0, 0))
        self.assertGreater(initial.real, 1.)
        self.assertLess(abs(initial - self.mode(values, 1, 0, (0, 0, 0))), 4e-15)
        for snapshot in (0, 1):
            for field in (1, 2, 3):
                self.assertLess(abs(self.mode(values, snapshot, field, (0, 0, 0))), 1e-16)
            self.assertLess(abs(self.mode(values, snapshot, 0, (self.n // 2, 0, 0))), 3e-15)

    def test_models_share_initial_energy_with_model_specific_initial_flux(self):
        default, m1 = self.generate(None)
        explicit, _ = self.generate('m1')
        self.assertEqual(default, explicit, 'default M1 reference must stay byte-for-byte identical')
        _, fld = self.generate('fld')
        cells = self.n**3
        self.assertEqual(m1[:cells], fld[:cells])
        self.assertEqual(max(abs(value) for value in m1[cells:4*cells]), 0.)
        self.assertGreater(max(abs(value) for value in fld[cells:4*cells]), 1e-5)
        self.assertGreater(max(abs(left - right) for left, right in zip(m1[4*cells:], fld[4*cells:])), 1e-5)

    def test_tiny_time_retains_stable_low_mode_attenuation(self):
        interval = 1e-9
        _, values = self.generate('fld', time=interval)
        wave = (1, 0, 0)
        initial = self.mode(values, 0, 0, wave)
        final = self.mode(values, 1, 0, wave)
        expected = initial * math.exp(-self.c / (3*self.chi) * (2*math.pi/self.length)**2 * interval)
        self.assertLess(abs(final - expected), 3e-14)
        self.assertLess(abs(final), abs(initial))

    def test_zero_opacity_and_unknown_models_are_rejected_without_output(self):
        for arguments in (['--model', 'fld', '--chi', '0'], ['--model', 'unknown']):
            path = self.directory / 'invalid.bin'
            result = subprocess.run([str(GENERATOR), '--output', str(path), *arguments],
                capture_output=True, text=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(path.exists())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator', required=True)
    options, rest = parser.parse_known_args()
    GENERATOR = str(Path(options.generator).resolve())
    unittest.main(argv=[__file__, *rest])
