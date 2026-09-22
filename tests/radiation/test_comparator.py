#!/usr/bin/env python3
"""Exercise structured radiation result validation."""

import copy
import importlib.util
import math
import pathlib
import unittest

path = pathlib.Path(__file__).with_name("check.py")
spec = importlib.util.spec_from_file_location("radiationCheck", path)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)

fixtureLimits = {
    field: {norm: 2.0 ** (-6 + 3 * fieldIndex + normIndex)
            for normIndex, norm in enumerate(check.norms)}
    for fieldIndex, field in enumerate(check.fields)
}
valid = {
    "schemaVersion": 1, "status": "completed",
    "problem": "RADIATION_GAUSSIAN_PULSE", "finalTime": 0.2,
    "analytic": {field: {"l1": 0.0, "l2": 0.0, "linf": 0.0}
                 for field in check.fields},
}


class ComparatorTests(unittest.TestCase):
    def compare(self, summary=valid, case="gaussian_pulse", time=0.2, limits=fixtureLimits):
        return check.check(summary, case, time, limits=limits)

    def testValidSummary(self):
        rows = self.compare()
        self.assertEqual(set(rows), set(check.fields))
        self.assertTrue(all(value == 0 for row in rows.values() for value in row.values()))

    def testMetadataIsRequired(self):
        for key, value in (("schemaVersion", 2), ("status", "running"),
                           ("problem", "RADIATION_STREAMING_FRONT"), ("finalTime", 0.1)):
            summary = copy.deepcopy(valid)
            summary[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.compare(summary)

    def testFieldsAndNormsAreExact(self):
        for field in check.fields:
            summary = copy.deepcopy(valid)
            del summary["analytic"][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.compare(summary)
        summary = copy.deepcopy(valid)
        del summary["analytic"]["er"]["linf"]
        with self.assertRaises(ValueError):
            self.compare(summary)

    def testEachLimitIsInclusiveAndEnforced(self):
        for field in check.fields:
            for norm, key in check.normKeys.items():
                limit = fixtureLimits[field][norm]
                summary = copy.deepcopy(valid)
                summary["analytic"][field][key] = limit
                self.assertEqual(self.compare(summary)[field][norm], limit)
                summary["analytic"][field][key] = math.nextafter(limit, math.inf)
                with self.assertRaisesRegex(ValueError, f"Inaccurate {field} {norm}"):
                    self.compare(summary)

    def testInvalidValuesFail(self):
        for value in (math.nan, math.inf, -1.0):
            summary = copy.deepcopy(valid)
            summary["analytic"]["er"]["l1"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.compare(summary)

    def testLimitPolicyMustBeComplete(self):
        with self.assertRaises(TypeError):
            check.check(valid, "gaussian_pulse", 0.2)
        with self.assertRaises(ValueError):
            self.compare(limits={})


if __name__ == "__main__":
    unittest.main()
