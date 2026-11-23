"""CPU integration test: interrupted training must match uninterrupted training."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dill
import numpy as np
from omegaconf import OmegaConf
import torch
from accelerate import Accelerator

from diffusion_policy.dataset.base_dataset import BaseDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.workspace.train_diffusion_unet_image_workspace import TrainDiffusionUnetImageWorkspace


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(2, 10)
        self.obs_encoder = torch.nn.Linear(2, 2)

    def set_normalizer(self, normalizer):
        pass

    def forward(self, batch):
        prediction = self.model(self.obs_encoder(batch['obs']['x']))
        return (prediction - batch['action'][:, 0]).square().mean()

    def predict_action(self, obs, prefix):
        prediction = self.model(self.obs_encoder(obs['x']))
        return {'action_pred': prediction[:, None].repeat(1, 16, 1)}


class TinyDataset(BaseDataset):
    def __len__(self):
        return 6

    def __getitem__(self, index):
        return {'obs': {'x': torch.tensor([float(index), float(np.random.normal())])},
                'action': torch.randn(16, 10)}

    def get_validation_dataset(self):
        return self

    def get_normalizer(self):
        return None


class TinyRunner(BaseImageRunner):
    def run(self, policy):
        return {}


def instantiate(config, **kwargs):
    if config._target_ == 'tiny_policy':
        return TinyPolicy()
    if config._target_ == 'tiny_dataset':
        return TinyDataset()
    if config._target_ == 'tiny_runner':
        return TinyRunner(**kwargs)
    if config._target_ == 'ema':
        return EMAModel(**kwargs)
    raise AssertionError(config)


class TrainingResumeTest(unittest.TestCase):
    def config(self):
        return OmegaConf.create({
            'policy': {'_target_': 'tiny_policy', 'obs_encoder': {'pretrained': False}},
            'optimizer': {'_target_': 'torch.optim.AdamW', 'lr': 0.01, 'betas': [0.9, 0.99], 'weight_decay': 0.0},
            'ema': {'_target_': 'ema'},
            'training': {'seed': 42, 'resume': False, 'use_ema': True, 'debug': False,
                         'num_epochs': 3, 'lr_scheduler': 'cosine', 'lr_warmup_steps': 1,
                         'gradient_accumulate_every': 2, 'max_train_steps': None,
                         'max_val_steps': None, 'freeze_encoder': False, 'rollout_every': 1,
                         'sample_every': 1, 'checkpoint_every': 1, 'tqdm_interval_sec': 100},
            'task': {'dataset': {'_target_': 'tiny_dataset', 'dataset_path': '/tmp/tiny'},
                     'env_runner': {'_target_': 'tiny_runner'}},
            'dataloader': {'batch_size': 2, 'shuffle': True, 'num_workers': 0},
            'val_dataloader': {'batch_size': 2, 'shuffle': False, 'num_workers': 0},
            'logging': {'project': 'test', 'mode': 'disabled'},
            'checkpoint': {'save_last_ckpt': True, 'save_last_snapshot': False,
                           'topk': {'monitor_key': 'train_loss', 'mode': 'min', 'k': 2,
                                    'format_str': 'epoch={epoch:04d}.ckpt'}},
        })

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_interrupted_run_matches_continuous(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch('diffusion_policy.workspace.train_diffusion_unet_image_workspace.hydra.utils.instantiate', side_effect=instantiate), \
                patch('diffusion_policy.workspace.train_diffusion_unet_image_workspace.Accelerator',
                      side_effect=lambda **kwargs: Accelerator(cpu=True)):
            cfg = self.config()
            full_dir = Path(directory) / 'full'
            full_dir.mkdir()
            full = TrainDiffusionUnetImageWorkspace(cfg, str(full_dir))
            full.run()
            split_dir = Path(directory) / 'split'
            split_dir.mkdir()
            first = TrainDiffusionUnetImageWorkspace(cfg, str(split_dir))
            save = first._save_training_checkpoint

            def interrupt(path=None):
                save(path)
                if path is None and first.epoch == 1:
                    raise InterruptedError('Simulated stop after first complete epoch')

            first._save_training_checkpoint = interrupt
            with self.assertRaises(InterruptedError):
                first.run()
            latest = split_dir / 'checkpoints/latest.ckpt'
            saved = torch.load(latest, map_location='cpu', pickle_module=dill)
            self.assertEqual(dill.loads(saved['pickles']['epoch']), 1)
            self.assertEqual(dill.loads(saved['pickles']['global_step']), 3)
            self.assertIn('optimizer', saved['state_dicts'])
            resume_cfg = copy.deepcopy(cfg)
            resume_cfg.training.resume = True
            resumed = TrainDiffusionUnetImageWorkspace(resume_cfg, str(split_dir))
            resumed.run()
            for key in ('model', 'ema_model', 'optimizer', 'lr_scheduler', 'ema'):
                self.assert_nested_equal(getattr(full, key).state_dict(), getattr(resumed, key).state_dict())
            self.assertEqual(resumed.epoch, 3)
            self.assertEqual(resumed.global_step, 9)
            self.assertEqual(resumed.ema.optimization_step, 6)
            # A failed replacement must leave latest readable and unchanged.
            previous = latest.read_bytes()
            with patch.object(resumed, 'save_checkpoint', side_effect=OSError('disk failure')):
                with self.assertRaises(OSError):
                    resumed._save_training_checkpoint()
            self.assertEqual(latest.read_bytes(), previous)
            self.assertFalse(list(latest.parent.glob('*.tmp')))

    def test_legacy_weights_only_checkpoint_is_rejected(self):
        workspace = TrainDiffusionUnetImageWorkspace.__new__(TrainDiffusionUnetImageWorkspace)
        workspace.cfg = self.config()
        with self.assertRaisesRegex(ValueError, 'lacks resume state'):
            workspace._restore_training_checkpoint({'state_dicts': {'model': {}}, 'pickles': {}})


if __name__ == '__main__':
    unittest.main()
