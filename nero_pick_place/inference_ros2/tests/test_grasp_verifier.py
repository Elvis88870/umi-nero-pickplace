import unittest

from env.grasp_verifier import GraspVerifier


class GraspVerifierTest(unittest.TestCase):
    def verifier(self, width=0.065, effort=0.8, calibrated=True, valid=True):
        v = GraspVerifier({
            "effort_calibrated": calibrated, "contact_effort_min": 0.5,
        })
        for i in range(9):
            t = 10.0 + i * 0.05
            v.update(t, t + 1000, width, effort, valid)
        return v

    def test_stable_contact_current_confirms(self):
        self.assertTrue(self.verifier().check(10.4, 1010.4, 10.0, 0.096)[0])

    def test_uncalibrated_never_confirms(self):
        self.assertFalse(self.verifier(calibrated=False).check(
            10.4, 1010.4, 10.0, 0.096)[0])

    def test_empty_closed_jaws_do_not_confirm_even_with_high_current(self):
        self.assertFalse(self.verifier(width=0.001, effort=2.0).check(
            10.4, 1010.4, 10.0, 0.096)[0])

    def test_low_current_at_model_width_does_not_confirm(self):
        self.assertFalse(self.verifier(effort=0.1).check(
            10.4, 1010.4, 10.0, 0.096)[0])

    def test_invalid_or_stale_feedback_does_not_confirm(self):
        self.assertFalse(self.verifier(valid=False).check(
            10.4, 1010.4, 10.0, 0.096)[0])
        self.assertFalse(self.verifier().check(11.0, 1011.0, 10.0, 0.096)[0])

    def test_duplicate_stamps_do_not_prove_contact(self):
        v = GraspVerifier({"effort_calibrated": True, "contact_effort_min": 0.5})
        for i in range(10):
            v.update(10.0 + i * 0.05, 1010.0, 0.065, 0.8, True)
        self.assertFalse(v.check(10.45, 1010.45, 10.0, 0.096)[0])

    def test_moving_jaws_do_not_confirm(self):
        v = GraspVerifier({"effort_calibrated": True, "contact_effort_min": 0.5})
        for i in range(9):
            t = 10.0 + i * 0.05
            v.update(t, t + 1000, 0.080 - i * 0.001, 0.8, True)
        self.assertFalse(v.check(10.4, 1010.4, 10.0, 0.096)[0])

    def test_current_spike_does_not_confirm(self):
        v = self.verifier(effort=0.1)
        v.update(10.45, 1010.45, 0.065, 0.9, True)
        self.assertFalse(v.check(10.45, 1010.45, 10.0, 0.096)[0])

    def test_stable_low_current_model_stop_allows_correction(self):
        v = self.verifier(width=0.063, effort=0.1)
        self.assertTrue(v.correction_ready(
            10.4, 1010.4, 10.0, 0.096, 0.063, 0.001)[0])

    def test_moving_or_stale_feedback_blocks_correction(self):
        moving = GraspVerifier({
            "effort_calibrated": True, "contact_effort_min": 0.5,
        })
        for i in range(9):
            t = 10.0 + i * 0.05
            moving.update(t, t + 1000, 0.070 - i * 0.001, 0.1, True)
        self.assertFalse(moving.correction_ready(
            10.4, 1010.4, 10.0, 0.096, 0.063, 0.001)[0])
        self.assertFalse(self.verifier(width=0.063, effort=0.1).correction_ready(
            11.0, 1011.0, 10.0, 0.096, 0.063, 0.001)[0])

    def test_current_object_measurements_are_separated(self):
        config = {
            "effort_calibrated": True,
            "contact_effort_min": 300.0,
            "effort_required": False,
            "contact_width_error_min_m": 0.0009,
            "empty_closed_width_m": 0.0607,
            "empty_width_margin_m": 0.0002,
        }
        held = GraspVerifier(config)
        empty = GraspVerifier(config)
        for i in range(9):
            t = 10.0 + i * 0.05
            held.update(t, t + 1000, 0.0612, 73.0, True)
            empty.update(t, t + 1000, 0.06065, 214.0, True)
        self.assertTrue(held.check(
            10.4, 1010.4, 10.0, 0.096, command_width=0.060)[0])
        self.assertFalse(empty.check(
            10.4, 1010.4, 10.0, 0.096, command_width=0.060)[0])

    def test_width_obstruction_allows_tightening_when_empty(self):
        config = {
            "effort_required": False,
            "contact_width_error_min_m": 0.0009,
        }
        empty = GraspVerifier(config)
        held = GraspVerifier(config)
        for i in range(9):
            t = 10.0 + i * 0.05
            empty.update(t, t + 1000, 0.06365, 50.0, True)
            held.update(t, t + 1000, 0.06420, 50.0, True)
        self.assertTrue(empty.correction_ready(
            10.4, 1010.4, 10.0, 0.096, 0.063, 0.001,
            command_width=0.063)[0])
        self.assertTrue(held.check(
            10.4, 1010.4, 10.0, 0.096, command_width=0.063)[0])


if __name__ == "__main__":
    unittest.main()
