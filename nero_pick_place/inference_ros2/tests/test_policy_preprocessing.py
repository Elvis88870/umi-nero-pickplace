import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torchvision import transforms as tv

from runtime.policy_runtime import deterministic_camera_transforms


class InferencePreprocessingTest(unittest.TestCase):
    def test_center_crop_retains_geometry_without_rng_or_color_changes(self):
        original = nn.Sequential(
            tv.RandomCrop(212), tv.Resize(224, antialias=True),
            tv.ColorJitter(brightness=.3, contrast=.4, saturation=.5, hue=.08))
        encoder = SimpleNamespace(rgb_keys=["camera0_rgb"],
            key_transform_map=nn.ModuleDict({"camera0_rgb": original}))
        deterministic_camera_transforms(encoder)
        pipeline = encoder.key_transform_map["camera0_rgb"]
        self.assertIsInstance(original[0], tv.RandomCrop)
        self.assertIsInstance(original[2], tv.ColorJitter)  # Training not mutated.
        image = torch.linspace(0, 1, 2 * 3 * 224 * 224).reshape(2, 3, 224, 224)
        expected = tv.Resize(224, antialias=True)(tv.CenterCrop(212)(image))
        for seed in (1, 42, 123):
            torch.manual_seed(seed)
            rng_before = torch.random.get_rng_state().clone()
            result = pipeline(image)
            self.assertTrue(torch.equal(result, expected))
            self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))

    def test_deterministic_normalization_is_preserved_for_all_cameras(self):
        encoder = SimpleNamespace(rgb_keys=["camera0_rgb", "camera1_rgb"],
            key_transform_map=nn.ModuleDict({k: nn.Sequential(
                tv.Normalize([.5] * 3, [.5] * 3))
                for k in ("camera0_rgb", "camera1_rgb")}))
        deterministic_camera_transforms(encoder)
        for pipeline in encoder.key_transform_map.values():
            self.assertTrue(torch.equal(pipeline(torch.zeros(1, 3, 224, 224)),
                                        -torch.ones(1, 3, 224, 224)))

    def test_unknown_random_transform_fails_instead_of_silently_remaining(self):
        encoder = SimpleNamespace(rgb_keys=["camera0_rgb"],
            key_transform_map=nn.ModuleDict({"camera0_rgb": tv.RandomHorizontalFlip()}))
        with self.assertRaisesRegex(ValueError, "unsupported deployment image transform"):
            deterministic_camera_transforms(encoder)


if __name__ == "__main__":
    unittest.main()
