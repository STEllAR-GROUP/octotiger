#!/usr/bin/env python3
"""Exercise malformed/stale output and explicit error-limit rejection.

All limits below are synthetic comparator-unit-test inputs, NOT calibrated
Gaussian or other physics acceptance thresholds. The missing historical helper
never existed in the available history, so no such thresholds are assumed.
"""

import copy
import importlib.util
import math
import pathlib
import unittest


path = pathlib.Path(__file__).with_name("check.py")
spec = importlib.util.spec_from_file_location("radiation_check", path)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)

# Distinct powers of two keep inclusive threshold tests exact and catch swapped
# field/norm mappings. These limits exercise comparator mechanics only.
FIXTURE_LIMITS = {
    field: {norm: 2.0 ** (-6 + 3 * f + n) for n, norm in enumerate(check.NORMS)}
    for f, field in enumerate(check.FIELDS)
}
VALID = "RADIATION_TEST_FINISHED RADIATION_GAUSSIAN_PULSE t=0.2\n" + "".join(
    f"{field} 0 0 0\n" for field in check.FIELDS
)


def with_norm(field, index, value):
    values = ["0", "0", "0"]
    values[index] = str(value)
    return VALID.replace(f"{field} 0 0 0", f"{field} " + " ".join(values))


class ComparatorTests(unittest.TestCase):
    def compare(self, text, case="gaussian_pulse", time=0.2, limits=FIXTURE_LIMITS):
        return check.check(text, case, time, limits=limits)

    def test_valid_zero_and_scientific_output(self):
        rows = self.compare(VALID)
        self.assertEqual(set(rows), set(check.FIELDS))
        self.assertTrue(all(value == 0 for row in rows.values() for value in row.values()))
        self.compare("solver startup\n" + VALID.replace("t=0.2", "t=2.00000000000000011e-01\nL1, L2")
                     .replace("er 0 0 0", "rho 0 0 0\n  er +0.0 0e+00 .0") + "solver exit\n")

    def test_all_cases_and_missing_marker(self):
        for case in check.CASES:
            with self.subTest(case=case):
                self.compare(VALID.replace("GAUSSIAN_PULSE", case.upper()), case=case)
        for text in ("", "\n", "er 0 0 0\n", VALID.split("\n", 1)[1]):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.compare(text)

    def test_original_rejection_cases(self):
        for text in (
            VALID.replace("t=0.2", "t=0.1"),
            VALID.replace("er 0 0 0", "er nan 0 0"),
            VALID.replace("fx 0 0 0", "fx 0 inf 0"),
            VALID.replace("fy 0 0 0\n", ""),
            VALID.replace("fz 0 0 0", "fz -1 0 0"),
            VALID.replace("er 0 0 0", "er 1 1 1"),
            VALID + VALID,
            VALID.replace("GAUSSIAN_PULSE", "STREAMING_FRONT"),
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.compare(text)

    def test_each_norm_rejects_invalid_numbers(self):
        for field in check.FIELDS:
            for index in range(3):
                for value in ("nan", "NaN", "inf", "-inf", "1e999", "-1e-20", "1_0", "0x1p0", "0junk"):
                    with self.subTest(field=field, index=index, value=value), self.assertRaises(ValueError):
                        self.compare(with_norm(field, index, value))

    def test_each_norm_limit_is_inclusive_and_enforced(self):
        for field in check.FIELDS:
            for index, norm in enumerate(check.NORMS):
                with self.subTest(field=field, norm=norm):
                    limit = FIXTURE_LIMITS[field][norm]
                    rows = self.compare(with_norm(field, index, limit))
                    self.assertEqual(rows[field][norm], limit)
                    with self.assertRaisesRegex(ValueError, f"Inaccurate {field} {norm}"):
                        self.compare(with_norm(field, index, math.nextafter(limit, math.inf)))
        zeros = {field: dict.fromkeys(check.NORMS, 0.0) for field in check.FIELDS}
        self.compare(VALID, limits=zeros)
        with self.assertRaises(ValueError):
            self.compare(with_norm("er", 0, 1e-300), limits=zeros)

    def test_missing_duplicate_malformed_and_pre_marker_rows(self):
        for field in check.FIELDS:
            row = f"{field} 0 0 0"
            for text in (
                VALID.replace(row + "\n", ""), VALID + row + "\n",
                VALID.replace(row, f"{field} 0 0"), VALID.replace(row, row + " 0"),
                VALID.replace(row, row + " # invalid"), row + "\n" + VALID,
                VALID + f"{field} invalid\n",
            ):
                with self.subTest(field=field, text=text), self.assertRaises(ValueError):
                    self.compare(text)

    def test_malformed_duplicate_and_stale_markers(self):
        for marker in (
            "RADIATION_TEST_FINISHED", "RADIATION_TEST_FINISHED RADIATION_GAUSSIAN_PULSE",
            "RADIATION_TEST_FINISHED RADIATION_GAUSSIAN_PULSE t=0.2 junk",
        ):
            for text in (VALID.replace(VALID.splitlines()[0], marker), VALID + marker + "\n"):
                with self.subTest(text=text), self.assertRaises(ValueError):
                    self.compare(text)
        for time in ("0.1", "0.3", "nan", "inf", "-0.2", "1e999", "0.2junk"):
            with self.subTest(time=time), self.assertRaises(ValueError):
                self.compare(VALID.replace("t=0.2", "t=" + time))
        with self.assertRaises(ValueError):
            self.compare(VALID + VALID.splitlines()[0] + "\n")

    def test_relative_time_tolerance_and_zero(self):
        self.compare(VALID.replace("t=0.2", "t=0.200000000001"))
        with self.assertRaises(ValueError):
            self.compare(VALID.replace("t=0.2", "t=0.200000000003"))
        self.compare(VALID.replace("t=0.2", "t=0"), time=0)
        with self.assertRaises(ValueError):
            self.compare(VALID.replace("t=0.2", "t=1e-20"), time=0)
        self.compare(VALID.replace("t=0.2", "t=1e-20"), time=1e-20)

    def test_expected_metadata_must_be_valid(self):
        for case in ("", "unknown", "RADIATION_GAUSSIAN_PULSE", None):
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.compare(VALID, case=case)
        for time in (-1, math.nan, math.inf, None, "invalid"):
            with self.subTest(time=time), self.assertRaises(ValueError):
                self.compare(VALID, time=time)

    def test_explicit_complete_finite_policy_is_required(self):
        with self.assertRaises(TypeError):
            check.check(VALID, "gaussian_pulse", 0.2)
        for limits in (None, {}, [], {**FIXTURE_LIMITS, "other": {}}):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                self.compare(VALID, limits=limits)
        for field in check.FIELDS:
            limits = copy.deepcopy(FIXTURE_LIMITS)
            del limits[field]
            with self.assertRaises(ValueError):
                self.compare(VALID, limits=limits)
            for norm in check.NORMS:
                limits = copy.deepcopy(FIXTURE_LIMITS)
                del limits[field][norm]
                with self.assertRaises(ValueError):
                    self.compare(VALID, limits=limits)
                for value in (-1, math.nan, math.inf, None, "invalid"):
                    limits = copy.deepcopy(FIXTURE_LIMITS)
                    limits[field][norm] = value
                    with self.subTest(field=field, norm=norm, value=value), self.assertRaises(ValueError):
                        self.compare(VALID, limits=limits)
            limits = copy.deepcopy(FIXTURE_LIMITS)
            limits[field]["other"] = 0
            with self.assertRaises(ValueError):
                self.compare(VALID, limits=limits)


if __name__ == "__main__":
    unittest.main(verbosity=2)
