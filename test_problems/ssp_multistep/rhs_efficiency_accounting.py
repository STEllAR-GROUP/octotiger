#!/usr/bin/env python3
# Copyright (c) 2026 AUTHORS
# Distributed under the Boost Software License, Version 1.0.
"""Check RHS work accounting and printed timestep consistency with synthetic data.

No application run or Silo library is required. Run this script directly.
"""

import copy
import unittest

from rhs_efficiency import rhs_work, step_statistics


CELLS_PER_LEAF = 8 ** 3
ROOT = ([1], 1)
LEVEL_ONE = (list(range(8, 16)), 9)
# Refine the (0, 0, 0) level-one leaf. Octo-Tiger's IDs interleave
# coordinate bits starting at the least significant refinement level.
REFINED = (sorted(list(range(9, 16)) + list(range(64, 121, 8))), 17)


def make_result(mode, schedule):
    """Each (step, mesh) entry describes the mesh after that step's regrid."""
    final_step = schedule[-1][0]
    snapshots = []
    for step, (leaves, nodes) in schedule:
        snapshots.append({"path": f"/synthetic/step-{step}.silo", "step": step,
                          "time": step / 10, "leaf_ids": list(leaves),
                          "leaf_count": len(leaves), "node_count": nodes})
    errors = [{"path": row["path"], "step": row["step"], "time": row["time"],
               "cells": row["leaf_count"] * CELLS_PER_LEAF} for row in snapshots]
    final = dict(snapshots[-1], cells=errors[-1]["cells"])
    stages = [(3, 0.0)] + [(2, 1.0)] * (final_step - 1) if mode == "transfer" else []
    return {"mode": mode, "stages": stages, "final": final,
            "snapshots": snapshots, "snapshot_errors": errors}


def ts_log(records):
    """TS counts describe the mesh used to complete the preceding batch."""
    return "\n".join(
        f"TS {step}:: t: {step / 10:e}, dt: 1.000000e-01, "
        f"time_elapsed: 1.000000e-02, rotational_time: 0.000000e+00, "
        f"x: 0, y: 0, z: 0, a: 1, ur: 0, ul: 0, vr: 0, vl: 0, "
        f"dim: 0, ngrids: {nodes}, leafs: {leaves}, amr_boundaries: 0"
        for step, leaves, nodes in records) + "\n"


class RhsEfficiencyAccounting(unittest.TestCase):
    def assert_work(self, result, records, expected_leaf_rhs, adaptive=True,
                    expected_intervals=None):
        work = rhs_work(result, ts_log(records), batch_steps=5, adaptive=adaptive)
        expected = expected_leaf_rhs * CELLS_PER_LEAF
        self.assertEqual(work["cell_rhs_evaluations"], expected)
        self.assertAlmostEqual(work["cell_rhs_per_simulated_time"],
                               expected / result["final"]["time"])
        self.assertEqual(work["cells_per_leaf"], CELLS_PER_LEAF)
        if expected_intervals is not None:
            self.assertEqual(len(work["mesh_work_intervals"]), expected_intervals)
        return work

    def test_refinement_boundary_uses_preceding_mesh(self):
        result = make_result("transfer", [(0, ROOT), (5, LEVEL_ONE), (10, REFINED)])
        # First five steps: 3+4*2 RHS on one leaf. Next five: 5*2 on eight.
        # The fifteen-leaf final mesh has not advanced and contributes no RHS.
        self.assert_work(result, [(5, 1, 1), (10, 8, 9)], 11 + 10 * 8,
                         expected_intervals=2)

    def test_coarsening_boundary_uses_preceding_mesh(self):
        result = make_result("transfer", [(0, REFINED), (5, LEVEL_ONE), (10, ROOT)])
        self.assert_work(result, [(5, 15, 17), (10, 8, 9)], 11 * 15 + 10 * 8,
                         expected_intervals=2)

    def test_partial_final_batch_uses_last_boundary_mesh(self):
        result = make_result("transfer", [(0, ROOT), (5, LEVEL_ONE),
                                          (10, REFINED), (12, LEVEL_ONE)])
        # Only steps 11 and 12 use the fifteen-leaf mesh; the final coarsening
        # must not retroactively change their weight.
        self.assert_work(result, [(5, 1, 1), (10, 8, 9), (12, 15, 17)],
                         11 + 10 * 8 + 4 * 15, expected_intervals=3)

    def test_missing_adaptive_boundary_snapshot_rejected(self):
        result = make_result("transfer", [(0, ROOT), (5, LEVEL_ONE), (10, REFINED)])
        result["snapshots"].pop(1)
        result["snapshot_errors"].pop(1)
        with self.assertRaises(AssertionError):
            rhs_work(result, ts_log([(5, 1, 1), (10, 8, 9)]), 5, True)

    def test_ts_leaf_count_disagreement_rejected(self):
        result = make_result("transfer", [(0, ROOT), (5, LEVEL_ONE), (10, REFINED)])
        # This is itself a valid octree size, but not the mesh used at step 10.
        with self.assertRaises(AssertionError):
            rhs_work(result, ts_log([(5, 1, 1), (10, 15, 17)]), 5, True)

    def test_ts_node_count_disagreement_rejected(self):
        result = make_result("transfer", [(0, ROOT), (5, LEVEL_ONE), (10, REFINED)])
        with self.assertRaises(AssertionError):
            rhs_work(result, ts_log([(5, 1, 1), (10, 8, 17)]), 5, True)

    def test_missing_ts_endpoint_rejected(self):
        result = make_result("transfer", [(0, ROOT), (5, LEVEL_ONE), (10, REFINED)])
        with self.assertRaises(AssertionError):
            rhs_work(result, ts_log([(5, 1, 1)]), 5, True)

    def test_duplicate_ts_endpoint_rejected(self):
        result = make_result("transfer", [(0, ROOT), (5, LEVEL_ONE), (10, REFINED)])
        with self.assertRaises(AssertionError):
            rhs_work(result, ts_log([(5, 1, 1), (5, 1, 1), (10, 8, 9)]), 5, True)

    def test_uniform_multistep_needs_only_initial_and_final_snapshots(self):
        result = make_result("transfer", [(0, LEVEL_ONE), (10, LEVEL_ONE)])
        self.assert_work(result, [(5, 8, 9), (10, 8, 9)], 21 * 8, adaptive=False)

    def test_uniform_ordinary_rk3_counts_three_rhs_per_step(self):
        result = make_result("rk3", [(0, LEVEL_ONE), (10, LEVEL_ONE)])
        self.assert_work(result, [(5, 8, 9), (10, 8, 9)], 30 * 8, adaptive=False)

    def test_adaptive_ordinary_rk3_uses_its_own_mesh_history(self):
        result = make_result("rk3", [(0, ROOT), (5, LEVEL_ONE), (10, REFINED)])
        self.assert_work(result, [(5, 1, 1), (10, 8, 9)], 15 + 15 * 8,
                         expected_intervals=2)

    def test_matching_final_snapshot_duplicate_does_not_duplicate_work(self):
        result = make_result("transfer", [(0, ROOT), (5, LEVEL_ONE), (10, REFINED)])
        snapshot = copy.deepcopy(result["snapshots"][-1])
        error = copy.deepcopy(result["snapshot_errors"][-1])
        snapshot["path"] = error["path"] = "/synthetic/final.silo"
        result["snapshots"].append(snapshot)
        result["snapshot_errors"].append(error)
        result["final"] = dict(snapshot, cells=error["cells"])
        self.assert_work(result, [(5, 1, 1), (10, 8, 9)], 11 + 10 * 8,
                         expected_intervals=2)


class PrintedStepStatistics(unittest.TestCase):
    @staticmethod
    def log(values):
        return "\n".join(f"{step} {time:e} {dt:e} 1.000000e-03 0.000000e+00"
                         for step, (time, dt) in enumerate(values, 1)) + "\n"

    def test_long_time_adjacent_small_dt_respects_printed_time_precision(self):
        first, second_dt = 1.00000046, 0.00000110
        final = first + second_dt
        result = step_statistics(self.log([(first, first), (final, second_dt)]), 2, final)
        self.assertEqual(result["accepted_steps"], 2)
        # Printed consecutive times differ by 2e-6 although dt is 1.1e-6.
        self.assertAlmostEqual(result["rows"][1][1] - result["rows"][0][1], 2e-6)

    def test_printed_exponent_change_uses_each_time_own_roundoff(self):
        for scale in (0.1, 1.0, 10.0, 1000.0):
            with self.subTest(scale=scale):
                first, second_dt = 0.99999946 * scale, 0.00000110 * scale
                final = first + second_dt
                result = step_statistics(self.log([(first, first), (final, second_dt)]), 2, final)
                self.assertEqual(result["accepted_steps"], 2)

    def test_positive_step_can_share_rounded_time_with_previous_step(self):
        first, second_dt = 1e6, 1e-4
        final = first + second_dt
        result = step_statistics(self.log([(first, first), (final, second_dt)]), 2, final)
        self.assertEqual(result["rows"][0][1], result["rows"][1][1])
        self.assertGreater(result["rows"][1][2], 0)

    def test_material_successive_time_mismatch_rejected(self):
        with self.assertRaisesRegex(AssertionError, "successive times"):
            step_statistics(self.log([(1.0, 1.0), (1.001, 1.1e-6)]), 2, 1.001)

    def test_accumulated_mismatch_rejected_despite_local_rounding_bounds(self):
        # Every increment differs from dt by at most the adjacent timestamps'
        # combined 1e-6 rounding bound. Their cumulative drift is still invalid.
        values = [(1.0, 1.0)] + [(1.0 + 2e-6 * step, 1e-6) for step in range(1, 5)]
        with self.assertRaisesRegex(AssertionError, "cumulative timesteps"):
            step_statistics(self.log(values), len(values), values[-1][0])

    def test_full_precision_final_time_mismatch_rejected(self):
        with self.assertRaisesRegex(AssertionError, "Final step time mismatch"):
            step_statistics(self.log([(1.0, 1.0)]), 1, 1.000002)

    def test_final_sum_is_tighter_than_printed_time_agreement(self):
        # Each printed interval and its cumulative prefix are compatible with
        # timestamp rounding; their sum cannot reach the exact final time.
        values = [(1.0, 1.0), (1.000001, 4e-7)]
        with self.assertRaisesRegex(AssertionError, "timestep sum"):
            step_statistics(self.log(values), 2, 1.0000014)

    def test_backward_printed_time_rejected(self):
        with self.assertRaisesRegex(AssertionError, "Invalid accepted-step record"):
            step_statistics(self.log([(1.0, 1.0), (0.999999, 1e-7)]), 2, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
