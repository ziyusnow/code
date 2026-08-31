import unittest
from pathlib import Path

import numpy as np

from case3.solve_case3 import (
    DISTANCE_NM,
    ESS_ENERGY_INITIAL,
    FAULT_G1_MAX,
    HOURS,
    RESERVE_HORIZON,
    configure_ess_capacity,
    configure_fault_g1_max,
    configure_g2_max,
    ess_energy_trajectory,
    evaluate_normal,
    load_input_data,
    normal_validation,
    project_speeds,
    reserve_requirements,
    solve_fault,
    solve_ra_lshade,
)


ROOT = Path(__file__).resolve().parent.parent


class Case3ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hours, cls.p_vital, cls.p_nonvital, cls.p_pv = load_input_data(
            ROOT / "data.md"
        )

    def test_input_data_has_24_ordered_hours(self):
        np.testing.assert_array_equal(self.hours, np.arange(HOURS))
        self.assertEqual(self.p_vital.shape, (HOURS,))
        self.assertEqual(self.p_nonvital.shape, (HOURS,))
        self.assertEqual(self.p_pv.shape, (HOURS,))

    def test_speed_projection_enforces_distance_and_bounds(self):
        projected = project_speeds(np.linspace(-5.0, 20.0, HOURS))
        self.assertAlmostEqual(float(projected.sum()), DISTANCE_NM, places=9)
        self.assertGreaterEqual(float(projected.min()), 0.0)
        self.assertLessEqual(float(projected.max()), 11.0)

    def test_ess_energy_sign_convention(self):
        energy = ess_energy_trajectory(np.array([-1.0, 1.0]))
        self.assertAlmostEqual(energy[0], ESS_ENERGY_INITIAL + 0.95)
        self.assertAlmostEqual(energy[1], ESS_ENERGY_INITIAL + 0.95 - 1.0 / 0.95)

    def test_15_mwh_configuration_preserves_soc_limits(self):
        import case3.solve_case3 as model

        try:
            configure_ess_capacity(15.0)
            self.assertEqual(model.ESS_ENERGY_MAX, 15.0)
            self.assertEqual(model.ESS_ENERGY_INITIAL, 7.5)
            self.assertEqual(model.ESS_ENERGY_MIN, 3.0)
            self.assertEqual(float(ess_energy_trajectory(np.zeros(HOURS))[0]), 7.5)
        finally:
            configure_ess_capacity(75.0)

    def test_fault_g1_max_configuration(self):
        import case3.solve_case3 as model

        try:
            configure_fault_g1_max(2.0)
            self.assertEqual(model.FAULT_G1_MAX, 2.0)
            self.assertEqual(model.FAULT_ALPHA, 0.2)
        finally:
            configure_fault_g1_max(6.0)

    def test_g2_max_configuration(self):
        import case3.solve_case3 as model

        try:
            configure_g2_max(15.0)
            self.assertEqual(model.G2_MAX, 15.0)
        finally:
            configure_g2_max(20.0)

    def test_reserve_uses_g2_ramp_and_two_hour_horizon(self):
        demand = np.full(HOURS, 12.0)
        p_g2 = np.full(HOURS, 2.0)
        energy, power = reserve_requirements(demand, p_g2)
        expected_first = max(0.0, 12.0 - FAULT_G1_MAX - 5.0)
        expected_second = max(0.0, 12.0 - FAULT_G1_MAX - 8.0)
        self.assertAlmostEqual(power[1], max(expected_first, expected_second))
        self.assertAlmostEqual(energy[1], (expected_first + expected_second) / 0.95)
        self.assertTrue(np.isnan(energy[0]))
        self.assertFalse(np.isnan(energy[HOURS - RESERVE_HORIZON]))

    def test_normal_decode_has_exact_power_balance(self):
        speeds = np.full(HOURS, 10.0)
        demand = self.p_vital + self.p_nonvital + 0.0022 * speeds**3 - self.p_pv
        p_g2 = 0.6 * demand
        position = np.r_[p_g2, np.zeros(HOURS), speeds]
        result = evaluate_normal(
            position,
            self.p_vital,
            self.p_nonvital,
            self.p_pv,
            "no_reserve",
        )
        residual = result["p_g1"][0] + result["p_g2"][0] - result["net_demand"][0]
        self.assertLess(float(np.max(np.abs(residual))), 1.0e-12)

    def test_small_ra_lshade_and_fault_stage_are_feasible(self):
        _, result, _ = solve_ra_lshade(
            self.p_vital,
            self.p_nonvital,
            self.p_pv,
            "dynamic_reserve",
            seed=7,
            np_max=32,
            np_min=4,
            iterations=20,
        )
        validation = normal_validation(result, "dynamic_reserve")
        self.assertLess(validation["distance_error"], 1.0e-8)
        self.assertLess(validation["balance_error"], 1.0e-8)
        fault = solve_fault(result, self.p_vital, self.p_nonvital, self.p_pv)
        self.assertLess(float(np.max(np.abs(fault["balance_residual"]))), 1.0e-7)
        self.assertGreaterEqual(fault["minimum_inequality_slack"], -1.0e-7)

    def test_fault_stage_accepts_explicit_start_and_duration(self):
        speeds = np.full(HOURS, 10.0)
        demand = self.p_vital + self.p_nonvital + 0.0022 * speeds**3 - self.p_pv
        position = np.r_[0.6 * demand, np.zeros(HOURS), speeds]
        result = evaluate_normal(
            position,
            self.p_vital,
            self.p_nonvital,
            self.p_pv,
            "no_reserve",
        )
        result = {key: value[0] for key, value in result.items()}
        fault = solve_fault(
            result,
            self.p_vital,
            self.p_nonvital,
            self.p_pv,
            fault_start_hour=20,
            fault_duration=4,
        )
        np.testing.assert_array_equal(fault["hours"], np.arange(20, 24))
        self.assertEqual(fault["fault_start_hour"], 20)
        self.assertEqual(fault["fault_duration"], 4)

    def test_fault_interval_rejects_schedule_overrun(self):
        dummy = {
            "energy_ess": np.full(HOURS, 7.5),
            "p_g2": np.zeros(HOURS),
            "p_g1": np.zeros(HOURS),
            "p_ess": np.zeros(HOURS),
            "p_propulsion": np.zeros(HOURS),
        }
        with self.assertRaises(ValueError):
            solve_fault(
                dummy,
                self.p_vital,
                self.p_nonvital,
                self.p_pv,
                fault_start_hour=21,
                fault_duration=4,
            )


if __name__ == "__main__":
    unittest.main()
