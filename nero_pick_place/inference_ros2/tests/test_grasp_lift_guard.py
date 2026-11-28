import unittest

import numpy as np

from env.grasp_lift_guard import GraspLiftGuard


class LiftGuardTest(unittest.TestCase):
    def guard_at_lift(self):
        guard = GraspLiftGuard({"enabled": True})
        for z, w, t in ((.18, .079, 0), (.10, .067, .2), (.08, .063, .4),
                        (.09, .063, .6)):
            guard.observe_model(np.asarray([.1, .2, z, 0, 0, 0]), w, .080, t)
        return guard

    def test_grasp_point_comes_from_model_lowest_approach(self):
        guard = self.guard_at_lift()
        self.assertTrue(guard.holding)
        self.assertAlmostEqual(guard.low_pose[2], .08)
        self.assertAlmostEqual(guard.base_width, .063)
        guard.observe_model(np.asarray([0, 0, .2, 0, 0, 0]), .096, .08, 1)
        self.assertTrue(guard.holding)  # Late open/lift forecasts cannot bypass it.
        self.assertAlmostEqual(guard.low_pose[2], .08)

    def test_correction_contact_and_real_tcp_are_all_required(self):
        guard = self.guard_at_lift()
        kwargs = dict(epoch_now=1000, actual_xyz=np.asarray([.1, .2, .08]),
                      feedback_fresh=True, command=.060, correction_done=True,
                      contact_check=lambda since: (True, "contact"), stable_sec=.3)
        guard.verify(now=.7, **kwargs)
        guard.verify(now=1.1, **{**kwargs, "correction_done": False})
        self.assertIsNone(guard.confirmed_epoch)
        guard.verify(now=1.2, **{**kwargs, "feedback_fresh": False})
        self.assertIsNone(guard.confirmed_epoch)
        guard.verify(now=1.3, **{**kwargs, "actual_xyz": np.zeros(3)})
        self.assertIsNone(guard.confirmed_epoch)
        guard.verify(now=1.4, **{**kwargs, "contact_check": lambda since: (False, "empty")})
        self.assertIsNone(guard.confirmed_epoch)
        guard.verify(now=1.5, **kwargs)
        self.assertEqual(guard.confirmed_epoch, 1000)
        self.assertFalse(guard.accept_observation(999.9))
        self.assertTrue(guard.holding)
        self.assertTrue(guard.accept_observation(1000.1))
        self.assertFalse(guard.holding)
        self.assertFalse(guard.accept_observation(1000.2))

    def test_contact_at_looser_command_cannot_release_early(self):
        guard = self.guard_at_lift()
        for now in (.7, 1.1):
            guard.verify(now, 1000, np.asarray([.1, .2, .08]), True, .063,
                         False, lambda since: (True, "contact"), .3)
        self.assertIsNone(guard.confirmed_epoch)
        guard.verify(1.2, 1000, np.asarray([.1, .2, .08]), True, .060,
                     True, lambda since: (True, "contact"), .3)
        self.assertIsNone(guard.confirmed_epoch)  # Final command needs settling.

    def test_failed_grip_never_releases_for_fresh_plan(self):
        guard = self.guard_at_lift()
        guard.verify(6, 1000, np.asarray([.1, .2, .08]), True, .060,
                     True, lambda since: (False, "empty"), .3)
        self.assertTrue(guard.failed)
        self.assertTrue(guard.holding)
        self.assertFalse(guard.accept_observation(2000))

    def test_fresh_descending_forecast_corrects_wrong_grasp_point(self):
        guard = self.guard_at_lift()
        guard.hold_epoch = 1000
        correction = np.asarray([
            [.12, .22, .10, 0, 0, 0, .070],
            [.13, .23, .075, 0, 0, 0, .065],
            [.14, .24, .06, 0, 0, 0, .063],
            [.16, .24, .12, 0, 0, 0, .063]])
        # Immediate opening is not requested in this case.
        correction[0, 6] = .063
        self.assertFalse(guard.update_approach(correction, 999.9, .080, 1))
        self.assertTrue(guard.update_approach(correction, 1000.1, .080, 1))
        np.testing.assert_allclose(guard.low_pose[:3], [.14, .24, .06])
        self.assertTrue(guard.holding)
        self.assertFalse(guard.update_approach(correction, 1000.1, .080, 1.2))

    def test_reopening_is_allowed_and_upward_correction_cannot_ratchet(self):
        guard = self.guard_at_lift()
        ceiling = guard.correction_ceiling
        for stamp in (1000, 1001, 1002):
            correction = np.asarray([[.12, .21, .15, 0, 0, 0, .096]])
            guard.update_approach(correction, stamp, .080, 1)
            self.assertTrue(guard.reopening)
            self.assertAlmostEqual(guard.base_width, .096)
            self.assertLessEqual(guard.low_pose[2], ceiling)
        self.assertTrue(guard.holding)
        # Closing reapproach can resume after reopening, even at a lower jaw
        # width than the open threshold, without reviving the old pose.
        correction = np.asarray([
            [.12, .21, .09, 0, 0, 0, .079],
            [.13, .22, .07, 0, 0, 0, .063]])
        self.assertTrue(guard.update_approach(correction, 1003, .080, 2))
        self.assertFalse(guard.reopening)
        np.testing.assert_allclose(guard.low_pose[:3], [.13, .22, .07])

    def test_lift_only_forecast_cannot_shift_held_grasp_or_skip_confirmation(self):
        guard = self.guard_at_lift()
        pose = guard.low_pose.copy()
        forecast = np.asarray([[.15, .25, .12, 0, 0, 0, .063],
                               [.20, .30, .20, 0, 0, 0, .063]])
        self.assertFalse(guard.update_approach(forecast, 1000, .080, 1))
        np.testing.assert_array_equal(guard.low_pose, pose)
        self.assertTrue(guard.holding)

    def test_confirmation_stays_frozen_and_timeout_is_not_reset_by_corrections(self):
        guard = self.guard_at_lift()
        correction = np.asarray([[.12, .21, .1, 0, 0, 0, .063],
                                [.13, .22, .07, 0, 0, 0, .063]])
        guard.update_approach(correction, 1000, .080, 1)
        self.assertAlmostEqual(guard.started, .6)
        for now in (1.1, 1.5):
            guard.verify(now, 1001, guard.low_pose[:3], True, .060,
                         True, lambda since: (True, "contact"), .3)
        self.assertEqual(guard.confirmed_epoch, 1001)
        self.assertAlmostEqual(guard.frozen_command, .060)
        pose = guard.low_pose.copy()
        correction[:, 0] += .1
        self.assertFalse(guard.update_approach(correction, 1002, .080, 2))
        np.testing.assert_array_equal(guard.low_pose, pose)


if __name__ == "__main__":
    unittest.main()
