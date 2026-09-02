import unittest
from unittest.mock import patch

import numpy as np

from case3 import run_design_grid
from case3 import run_fault_time_scan


class SensitivityScriptTests(unittest.TestCase):
    def test_design_grid_fault_retries_pass_seed_and_budget(self):
        expected = {"cv_f": 0.0}
        with patch.object(
            run_design_grid.model,
            "solve_fault",
            side_effect=[RuntimeError("first seed failed"), expected],
        ) as solve_fault:
            seed, result = run_design_grid._solve_fault_with_retries(
                {}, None, None, None, 32, 4, 20
            )

        self.assertEqual(seed, run_design_grid.RETRY_SEEDS[1])
        self.assertIs(result, expected)
        self.assertEqual(
            [call.kwargs["seed"] for call in solve_fault.call_args_list],
            list(run_design_grid.RETRY_SEEDS[:2]),
        )
        self.assertEqual(solve_fault.call_args.kwargs["np_max"], 32)
        self.assertEqual(solve_fault.call_args.kwargs["np_min"], 4)
        self.assertEqual(solve_fault.call_args.kwargs["iterations"], 20)

    def test_fault_scan_normal_retries_pass_formal_budget(self):
        expected = {"total_cost": 1.0}
        with patch.object(
            run_fault_time_scan.model,
            "solve_ra_lshade",
            side_effect=[
                RuntimeError("first seed failed"),
                (None, expected, []),
                (None, expected, []),
            ],
        ) as solve_normal:
            plans = run_fault_time_scan.solve_normal_plans(None, None, None)

        no_reserve_calls = solve_normal.call_args_list[:2]
        self.assertEqual(
            [call.args[4] for call in no_reserve_calls],
            list(run_fault_time_scan.RETRY_SEEDS[:2]),
        )
        self.assertEqual(
            plans["no_reserve"]["seed"], run_fault_time_scan.RETRY_SEEDS[1]
        )
        for call in solve_normal.call_args_list:
            self.assertEqual(call.kwargs["np_max"], run_fault_time_scan.model.NP_MAX)
            self.assertEqual(call.kwargs["np_min"], run_fault_time_scan.model.NP_MIN)
            self.assertEqual(
                call.kwargs["iterations"], run_fault_time_scan.model.MAX_ITERATIONS
            )

    def test_fault_scan_fault_retries_before_failure(self):
        expected = {"cv_f": 0.0}
        with patch.object(
            run_fault_time_scan.model,
            "solve_fault",
            side_effect=[RuntimeError("first seed failed"), expected],
        ) as solve_fault:
            seed, result = run_fault_time_scan.solve_fault_with_retries(
                {}, None, None, None, 5
            )

        self.assertEqual(seed, run_fault_time_scan.RETRY_SEEDS[1])
        self.assertIs(result, expected)
        self.assertEqual(
            [call.kwargs["seed"] for call in solve_fault.call_args_list],
            list(run_fault_time_scan.RETRY_SEEDS[:2]),
        )
        self.assertTrue(
            all(call.kwargs["fault_start_hour"] == 5 for call in solve_fault.call_args_list)
        )

    def test_shortfall_diagnostic_reserves_energy_for_future_critical_load(self):
        normal = {
            "energy_ess": np.array([3.0, 3.0, 3.0, 4.0, 4.0, 4.0]),
            "p_g2": np.zeros(6),
            "p_propulsion": np.array([0.0, 0.0, 0.0, 0.0, 1.0, 4.95]),
        }
        p_vital = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        p_nonvital = np.array([0.0, 0.0, 0.0, 0.0, 3.0, 0.0])
        p_pv = np.zeros(6)
        with patch.object(run_fault_time_scan, "FAULT_DURATION", 2), patch.object(
            run_fault_time_scan.model, "G2_MAX", 5.0
        ), patch.object(run_fault_time_scan.model, "FAULT_G1_MAX", 0.0):
            result = run_fault_time_scan.minimum_shortfall_diagnostic(
                normal, p_vital, p_nonvital, p_pv, 4
            )

        self.assertAlmostEqual(result["p_ess"][0], 0.0)
        self.assertGreater(result["p_ess"][1], 0.0)
        self.assertAlmostEqual(result["vital_shortfall_energy"], 0.0)


if __name__ == "__main__":
    unittest.main()
