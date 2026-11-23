if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import pickle
import tqdm
import numpy as np
import shutil
import math
import tempfile
import dill
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.policy.diffusion_unet_timm_policy import DiffusionUnetTimmPolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from accelerate import Accelerator
from accelerate.utils import gather_object

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionUnetImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch', 'training_state']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        # self.optimizer = hydra.utils.instantiate(
        #     cfg.optimizer, params=self.model.parameters())

        obs_encorder_lr = cfg.optimizer.lr
        if cfg.policy.obs_encoder.pretrained:
            obs_encorder_lr *= 0.1
            print('==> reduce pretrained obs_encorder\'s lr')
        obs_encorder_params = list()
        for param in self.model.obs_encoder.parameters():
            if param.requires_grad:
                obs_encorder_params.append(param)
        print(f'obs_encorder params: {len(obs_encorder_params)}')
        param_groups = [
            {'params': self.model.model.parameters()},
            {'params': obs_encorder_params, 'lr': obs_encorder_lr}
        ]
        # self.optimizer = hydra.utils.instantiate(
        #     cfg.optimizer, params=param_groups)
        optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
        optimizer_cfg.pop('_target_')
        self.optimizer = torch.optim.AdamW(
            params=param_groups,
            **optimizer_cfg
        )

        # configure training state
        self.global_step = 0
        self.epoch = 0

        # Saving optimizer state is independent of choosing to resume a run.
        self.training_state = {'version': 1}

    def _save_training_checkpoint(self, path=None):
        """Publish a complete checkpoint without truncating the previous latest."""
        path = pathlib.Path(path) if path else self.get_checkpoint_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix='.checkpoint-', suffix='.tmp', dir=path.parent)
        os.close(handle)
        try:
            self.save_checkpoint(path=temporary, use_thread=False)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _restore_training_checkpoint(self, payload):
        required = {'model', 'optimizer', 'lr_scheduler'}
        if self.cfg.training.use_ema:
            required.update(('ema_model', 'ema'))
        missing = required - set(payload['state_dicts'])
        if missing or 'training_state' not in payload['pickles']:
            raise ValueError(f'Checkpoint lacks resume state ({sorted(missing)}). '
                             'Use a new checkpoint saved with optimizer/scheduler state.')
        self.load_payload(payload, include_keys=['global_step', 'epoch', 'training_state'])

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        # Inference callers construct only the models and optimizer, not runtime
        # scheduler/EMA helpers. Their state is restored by run() when resuming.
        excluded = set(exclude_keys or ())
        excluded.update(key for key in ('lr_scheduler', 'ema') if not hasattr(self, key))
        return super().load_payload(payload, exclude_keys=excluded,
                                    include_keys=include_keys, **kwargs)

    def _capture_rank_rng(self, accelerator, train_loader, val_loader):
        return {
            'rank': accelerator.process_index,
            'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(accelerator.device) if accelerator.device.type == 'cuda' else None,
            'loaders': [loader.synchronized_generator.get_state()
                        if getattr(loader, 'synchronized_generator', None) is not None else None
                        for loader in (train_loader, val_loader)],
        }

    def _restore_rank_rng(self, accelerator, train_loader, val_loader):
        states = self.training_state['rank_rng']
        if len(states) != accelerator.num_processes:
            raise ValueError('Resume requires the same number of training processes')
        state = states[accelerator.process_index]
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'])
        if state['cuda'] is not None:
            torch.cuda.set_rng_state(state['cuda'], accelerator.device)
        for loader, rng in zip((train_loader, val_loader), state['loaders']):
            if rng is not None:
                loader.synchronized_generator.set_state(rng)

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        print(
            f"[rank {os.environ.get('RANK', '0')}] Initializing Accelerator "
            f"(master={os.environ.get('MASTER_ADDR', 'local')}:"
            f"{os.environ.get('MASTER_PORT', '-')})",
            flush=True,
        )
        accelerator = Accelerator(log_with='wandb')
        print(f"[rank {accelerator.process_index}] Accelerator ready: {accelerator.device}", flush=True)
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop('project')
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
        )

        # Read on CPU now; restore after scheduler and distributed wrappers exist.
        resume_payload = None
        if cfg.training.resume:
            resume_path = self.get_checkpoint_path()
            if not resume_path.is_file():
                raise FileNotFoundError(f'Resume requested but no checkpoint at {resume_path}')
            resume_payload = torch.load(resume_path, map_location='cpu', pickle_module=dill)

        # configure dataset
        print(f"[rank {accelerator.process_index}] Loading dataset", flush=True)
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset) or isinstance(dataset, BaseDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)

        # compute normalizer on the main process and save to disk
        print(f"[rank {accelerator.process_index}] Dataset loaded; preparing shared normalizer", flush=True)
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        if accelerator.is_main_process:
            normalizer = dataset.get_normalizer()
            with open(normalizer_path, 'wb') as stream:
                pickle.dump(normalizer, stream)

        # load normalizer on all processes
        accelerator.wait_for_everyone()
        with open(normalizer_path, 'rb') as stream:
            normalizer = pickle.load(stream)

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        print('train dataset:', len(dataset), 'train dataloader:', len(train_dataloader))
        print('val dataset:', len(val_dataset), 'val dataloader:', len(val_dataloader))

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # Construct after loader sharding so the schedule counts real updates.
        # Step the plain scheduler once per optimizer update on each rank.
        self.ema = None
        if cfg.training.use_ema:
            self.ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # configure env
        env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'), **cfg.checkpoint.topk)

        train_dataloader, val_dataloader, self.model, self.optimizer = accelerator.prepare(
            train_dataloader, val_dataloader, self.model, self.optimizer)
        device = accelerator.device
        if self.ema_model is not None:
            self.ema_model.to(device)
        batches_per_epoch = min(len(train_dataloader), cfg.training.max_train_steps or len(train_dataloader))
        updates_per_epoch = math.ceil(batches_per_epoch / cfg.training.gradient_accumulate_every)
        self.lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=updates_per_epoch * cfg.training.num_epochs,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=-1
        )

        signature = {
            'num_epochs': cfg.training.num_epochs,
            'world_size': accelerator.num_processes,
            'batches_per_epoch': batches_per_epoch,
            'batch_size': cfg.dataloader.batch_size,
            'accumulation': cfg.training.gradient_accumulate_every,
            'lr_scheduler': cfg.training.lr_scheduler,
            'lr_warmup_steps': cfg.training.lr_warmup_steps,
            'dataset_path': str(pathlib.Path(cfg.task.dataset.dataset_path).resolve()),
            'dataset_seed': cfg.task.dataset.get('seed', cfg.training.seed),
            'val_ratio': cfg.task.dataset.get('val_ratio', 0),
            'num_workers': cfg.dataloader.num_workers,
            'training_seed': cfg.training.seed,
        }
        if resume_payload is not None:
            # DDP prefixes are deliberately not written to the portable .ckpt.
            wrapped_model = self.model
            self.model = accelerator.unwrap_model(wrapped_model)
            try:
                self._restore_training_checkpoint(resume_payload)
            finally:
                self.model = wrapped_model
            if self.training_state['signature'] != signature:
                raise ValueError('Resume configuration changed. Keep total epochs, batch size, '
                                 'process count, dataset and scheduler settings unchanged.')
            topk_manager.path_value_map = self.training_state['topk']
            self._restore_rank_rng(accelerator, train_dataloader, val_dataloader)
            accelerator.print(f'Resumed optimizer, scheduler and EMA; next epoch={self.epoch}, '
                              f'global_step={self.global_step}, lr={self.lr_scheduler.get_last_lr()}')
            del resume_payload
        self.training_state['signature'] = signature
        print(f"Training process {accelerator.process_index}/{accelerator.num_processes}: device={device}", flush=True)
        if self.ema_model is not None:
            self.ema_model.to(device)

        # save batch for sampling
        train_sampling_batch = None

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        if not accelerator.is_main_process:
            log_path = os.path.join(self.output_dir, f'logs.rank{accelerator.process_index}.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(self.epoch, cfg.training.num_epochs):
                train_dataloader.set_epoch(self.epoch)
                val_dataloader.set_epoch(self.epoch)
                self.model.train()

                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        
                        # always use the latest batch
                        train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model(batch)
                        if not torch.isfinite(raw_loss):
                            raise ValueError('Nonfinite training loss')
                        group_start = (batch_idx // cfg.training.gradient_accumulate_every) * cfg.training.gradient_accumulate_every
                        group_size = min(cfg.training.gradient_accumulate_every, batches_per_epoch - group_start)
                        loss = raw_loss / group_size
                        accelerator.backward(loss)

                        # step optimizer
                        if ((batch_idx + 1) % cfg.training.gradient_accumulate_every == 0
                                or batch_idx + 1 == batches_per_epoch):
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            self.lr_scheduler.step()
                            if cfg.training.use_ema:
                                self.ema.step(accelerator.unwrap_model(self.model))

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': self.lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx + 1 == batches_per_epoch)
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            accelerator.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                        self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = accelerator.reduce(
                    torch.tensor(np.mean(train_losses), device=device), reduction='mean').item()
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # All ranks validate the same (EMA, when enabled) policy.
                # Gather per-example losses so Accelerate removes duplicated
                # final-shard examples before computing the validation mean.
                validate_now = (self.epoch % cfg.training.get('val_every', 1) == 0
                                or cfg.checkpoint.topk.monitor_key == 'val_loss')
                if validate_now and len(val_dataloader) > 0:
                    val_losses = []
                    with torch.no_grad():
                        for batch_idx, batch in enumerate(val_dataloader):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            if isinstance(policy, DiffusionUnetTimmPolicy):
                                values = policy.compute_loss(batch, reduction='none')
                            else:
                                values = policy(batch).expand(batch['action'].shape[0])
                            if not torch.isfinite(values).all():
                                raise ValueError('Nonfinite validation loss')
                            values = accelerator.gather_for_metrics(values.detach())
                            val_losses.append(values.cpu())
                            if (cfg.training.max_val_steps is not None
                                    and batch_idx + 1 >= cfg.training.max_val_steps):
                                break
                    step_log['val_loss'] = torch.cat(val_losses).mean().item()
                    accelerator.print(f"Epoch {self.epoch}: train_loss={train_loss:.6f} "
                                      f"val_loss={step_log['val_loss']:.6f}")

                def log_action_mse(step_log, category, pred_action, gt_action):
                    B, T, _ = pred_action.shape
                    pred_action = pred_action.view(B, T, -1, 10)
                    gt_action = gt_action.view(B, T, -1, 10)
                    step_log[f'{category}_action_mse_error'] = torch.nn.functional.mse_loss(pred_action, gt_action).item()
                    step_log[f'{category}_action_mse_error_pos'] = torch.nn.functional.mse_loss(pred_action[..., :3], gt_action[..., :3]).item()
                    step_log[f'{category}_action_mse_error_rot'] = torch.nn.functional.mse_loss(pred_action[..., 3:9], gt_action[..., 3:9]).item()
                    step_log[f'{category}_action_mse_error_width'] = torch.nn.functional.mse_loss(pred_action[..., 9], gt_action[..., 9]).item()
                # All ranks enter the prepared validation loader together because
                # Accelerate may synchronize its RNG when iteration starts.
                val_sampling_batch = None
                if (self.epoch % cfg.training.sample_every) == 0 and len(val_dataloader) > 0:
                    val_sampling_batch = next(iter(val_dataloader))

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0 and accelerator.is_main_process:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        gt_action = batch['action']
                        pred_action = policy.predict_action(batch['obs'], None)['action_pred']
                        log_action_mse(step_log, 'train', pred_action, gt_action)

                        if val_sampling_batch is not None:
                            batch = dict_apply(val_sampling_batch, lambda x: x.to(device, non_blocking=True))
                            gt_action = batch['action']
                            pred_action = policy.predict_action(batch['obs'], None)['action_pred']
                            log_action_mse(step_log, 'val', pred_action, gt_action)

                        del batch
                        del gt_action
                        del pred_action
                
                # The saved epoch is the NEXT epoch to run. File names and log
                # records continue to identify the epoch that just completed.
                completed_epoch = self.epoch
                accelerator.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.epoch += 1
                should_rank = (completed_epoch % cfg.training.checkpoint_every == 0
                               or self.epoch == cfg.training.num_epochs)
                if cfg.checkpoint.save_last_ckpt or should_rank:
                    accelerator.wait_for_everyone()
                    rank_rng = gather_object([self._capture_rank_rng(
                        accelerator, train_dataloader, val_dataloader)])
                    if accelerator.is_main_process:
                        topk_path = topk_manager.get_ckpt_path(step_log) if should_rank else None
                        self.training_state.update(
                            rank_rng=rank_rng, topk=copy.deepcopy(topk_manager.path_value_map),
                            completed_epoch=completed_epoch)
                        model_ddp = self.model
                        self.model = accelerator.unwrap_model(model_ddp)
                        try:
                            if topk_path is not None:
                                self._save_training_checkpoint(topk_path)
                            if cfg.checkpoint.save_last_ckpt:
                                self._save_training_checkpoint()
                            if cfg.checkpoint.save_last_snapshot and should_rank:
                                self.save_snapshot()
                        finally:
                            self.model = model_ddp
                    accelerator.wait_for_everyone()

        accelerator.wait_for_everyone()
        accelerator.end_training()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
