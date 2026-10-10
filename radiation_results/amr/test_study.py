#!/usr/bin/env python3
"""Focused tests for the AMR smoke evidence and threshold-feasibility checks."""
import csv
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import study


class GradientStudyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def histogram(self, entries):
        path = self.folder/"gradient-amr.csv"
        with path.open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(("t","step","level","leaf_blocks","leaf_cells"))
            for t,step,level,blocks in entries:
                writer.writerow((t,step,level,blocks,blocks*512))
        return path

    def test_cell_ratio_is_not_volume_fraction(self):
        self.assertAlmostEqual(study.ratio(8,1),8/7)
        self.assertAlmostEqual(study.occupancy(512,8)["fine_volume_fraction"],1/64)
        self.assertAlmostEqual(study.ratio(512,8),8/63)
        self.assertIsNone(study.ratio(8,8))

    def test_near_equal_scores_cannot_break_symmetry(self):
        values = [1.+i*1e-12 for i in range(8)] + [.01]*56
        candidates = study.threshold_candidates(values)
        self.assertEqual(candidates[0]["refined_base_blocks"],8)
        self.assertGreater(candidates[0]["fine_to_coarse_leaf_cell_ratio"],1)
        self.assertEqual(len(candidates),2)

    def test_constant_profile_has_no_initial_tags(self):
        self.assertEqual(study.threshold_candidates([0.]*64),[])

    def test_valid_mixed_histogram_has_true_cell_ratio(self):
        path = self.histogram([(0,0,3,480),(0,0,4,256),(.1,5,3,480),(.1,5,4,256)])
        result = study.read_histogram(path,4)
        self.assertEqual(result[-1]["status"],"ratio_in_target")
        self.assertAlmostEqual(result[-1]["fine_to_coarse_leaf_cell_ratio"],8/15)

    def test_all_fine_coarse_exception_is_not_ratio_success(self):
        path = self.histogram([(0,0,2,64),(.1,5,2,64)])
        result = study.read_histogram(path,2)
        self.assertEqual(result[-1]["status"],"all_fine_coarse_exception")
        self.assertIsNone(result[-1]["fine_to_coarse_leaf_cell_ratio"])

    def test_fine_level_four_all_fine_is_outside_target(self):
        path = self.histogram([(0,0,4,4096),(.1,5,4,4096)])
        self.assertEqual(study.read_histogram(path,4)[-1]["status"],"all_fine_outside_target")

    def test_missing_volume_is_rejected(self):
        path = self.histogram([(0,0,3,479),(0,0,4,256),(.1,5,3,479),(.1,5,4,256)])
        with self.assertRaisesRegex(AssertionError,"cover the domain"):
            study.read_histogram(path,4)

    def test_nonadjacent_active_leaves_are_rejected(self):
        path = self.histogram([(0,0,2,1),(.1,5,2,1)])
        with self.assertRaisesRegex(AssertionError,"adjacent"):
            study.read_histogram(path,4)

    def test_no_evolution_is_not_success(self):
        path = self.histogram([(0,0,2,64)])
        with self.assertRaisesRegex(AssertionError,"evolved"):
            study.read_histogram(path,2)

    def test_duplicate_level_is_rejected(self):
        path = self.histogram([(0,0,2,64),(.1,5,2,64),(.1,5,2,64)])
        with self.assertRaisesRegex(AssertionError,"Duplicate"):
            study.read_histogram(path,2)

    def test_radiation_ledger_detects_regrid_energy_change(self):
        path = self.folder/"radiation-conservation.csv"
        path.write_text("t,er,er_source,er_boundary\n0,10,0,0\n.1,11,2,1\n.2,10,2,2\n")
        self.assertEqual(study.check_ledger(path)["worst_relative_energy_residual"],0)
        path.write_text("t,er,er_source,er_boundary\n0,10,0,0\n.1,11,2,1\n.2,10.01,2,2\n")
        with self.assertRaisesRegex(AssertionError,"does not close"):
            study.check_ledger(path)

    def test_application_timeout_stops_local_child(self):
        with (self.folder/"run.log").open("w") as log:
            with self.assertRaises(subprocess.TimeoutExpired):
                study.run_bounded([sys.executable,"-c","import time; time.sleep(2)"],self.folder,log,.1,1)

    def test_nonzero_application_exit_is_not_success(self):
        with (self.folder/"run.log").open("w") as log:
            with self.assertRaises(subprocess.CalledProcessError):
                study.run_bounded([sys.executable,"-c","raise SystemExit(7)"],self.folder,log,3,1)


if __name__ == "__main__":
    unittest.main()
