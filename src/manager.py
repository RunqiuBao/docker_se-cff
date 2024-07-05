import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch.distributed as dist

from components import models
from components import datasets
from components import methods

from utils.logger import ExpLogger, TimeCheck
from utils.metrics import SummationMeter, Metric


class DLManager:
    def __init__(self, args, cfg=None):
        self.args = args
        self.cfg = cfg
        self.logger = ExpLogger(save_root=args.save_root) if args.is_master else None

        if self.cfg is not None:
            self._init_from_cfg(cfg)

        self.current_epoch = 0

    def _init_from_cfg(self, cfg):
        assert cfg is not None
        self.cfg = cfg

        self.model = _prepare_model(
            self.cfg.MODEL,
            is_distributed=self.args.is_distributed,
            local_rank=self.args.local_rank if self.args.is_distributed else None,
            logger=self.logger,
        )
        self.optimizer = _prepare_optimizer(self.cfg.OPTIMIZER, self.model)
        self.scheduler = _prepare_scheduler(self.cfg.SCHEDULER, self.optimizer)
        if self.args.resume_cpt is not None:
            device = torch.device(f"cuda:{self.args.local_rank}")
            checkpoint = torch.load(self.args.resume_cpt, map_location=device)
            if self.logger:
                self.logger.write(
                    "loading checkpoint {} ...".format(self.args.resume_cpt)
                )
            # FIXME: adding 'module.' to each key in model state dict
            model_statedict = self.model.state_dict()
            for key, value in checkpoint["model"].items():
                if 'keypt1_predictor' in key or 'keypt2_predictor' in key or '_keypt_feature_extraction_net' in key:
                    continue
                if "module." + key in model_statedict and model_statedict["module." + key].size() == value.size():
                    model_statedict["module." + key] = value
                else:
                    print("Skipping parameter {} due to not previously exist or size mismatch.".format(key))
            self.model.load_state_dict(model_statedict)
            if not self.args.only_resume_weight:
                self.optimizer['optimizer'].load_state_dict(checkpoint["optimizer"])
                if 'optimizer_keypt' in checkpoint:
                    self.optimizer['optimizer_keypt'].load_state_dict(checkpoint["optimizer_keypt"])
                    self.scheduler['scheduler_keypt'].load_state_dict(checkpoint["scheduler_keypt"])
                self.scheduler['scheduler'].load_state_dict(checkpoint["scheduler"])
                self.args.start_epoch = checkpoint["epoch"] + 1
                self.current_epoch = self.args.start_epoch
                print("resumed old training states.")

        self.get_train_loader = getattr(
            datasets, self.cfg.DATASET.TRAIN.NAME
        ).get_dataloader
        self.get_valid_loader = getattr(
            datasets, self.cfg.DATASET.VALID.NAME
        ).get_dataloader
        # self.get_test_loader = getattr(datasets, self.cfg.DATASET.TEST.NAME).get_dataloader

        self.method = getattr(methods, self.cfg.METHOD)

    def trainAndValid(self):
        if self.args.is_master:
            self._log_before_train()
        train_loader = self.get_train_loader(
            args=self.args,
            dataset_cfg=self.cfg.DATASET.TRAIN,
            dataloader_cfg=self.cfg.DATALOADER.TRAIN,
            is_distributed=self.args.is_distributed,
        )
        valid_loader = self.get_valid_loader(
            args=self.args,
            dataset_cfg=self.cfg.DATASET.VALID,
            dataloader_cfg=self.cfg.DATALOADER.VALID,
            is_distributed=self.args.is_distributed,
        )

        # # profiling the network
        # netWorkClass = getattr(models, self.cfg.MODEL.NAME)
        # profile_model = netWorkClass(**self.cfg.MODEL.PARAMS)
        # # torch.Size([4, 1, 360, 576, 1, 10])
        # flops, numParams = netWorkClass.ComputeCostProfile(
        #     profile_model, next(iter(train_loader))["event"]["left"].shape
        # )
        # if self.args.is_master:
        #     self.logger.write(
        #         "[Profile] model(%s) computation cost: gFlops %f | numParams %f M"
        #         % (self.cfg.MODEL.NAME, float(flops / 10**9), float(numParams / 10**6))
        #     )
        # del profile_model

        time_checker = TimeCheck(self.cfg.TOTAL_EPOCH)
        time_checker.start()
        smallestValidEPE = sys.float_info.max
        for epoch in range(self.args.start_epoch, self.cfg.TOTAL_EPOCH):
            if self.args.is_distributed:
                dist.barrier()
                train_loader.sampler.set_epoch(epoch)
            train_log_dict = self.method.train(
                model=self.model,
                data_loader=train_loader,
                optimizer=self.optimizer,
                is_distributed=self.args.is_distributed,
                world_size=self.args.world_size,
            )

            for key, scheduler in self.scheduler.items():
                scheduler.step()
                if self.args.is_master:
                    print(key + "'s lr: {}".format(scheduler.get_lr()))
            if self.args.is_distributed:
                train_log_dict = self._gather_log(train_log_dict)
            if self.args.is_master:
                self._log_after_epoch(
                    epoch + 1, time_checker, train_log_dict, "train", isSaveFinal=False
                )

            if self.args.is_distributed:
                dist.barrier()
                valid_loader.sampler.set_epoch(epoch)
            valid_log_dict = self.method.valid(
                model=self.model,
                data_loader=valid_loader,
                is_distributed=self.args.is_distributed,
                world_size=self.args.world_size,
                logger=self.logger,
            )

            if self.args.is_distributed:
                valid_log_dict = self._gather_log(valid_log_dict)
            if self.args.is_master:
                self._log_after_epoch(
                    epoch + 1,
                    time_checker,
                    valid_log_dict,
                    "valid",
                    isSaveBest=valid_log_dict["BestIndex"].avg < smallestValidEPE,
                )

            if valid_log_dict["BestIndex"].avg < smallestValidEPE:
                smallestValidEPE = valid_log_dict["BestIndex"].avg

            self.current_epoch += 1

    def train(self):
        if self.args.is_master:
            self._log_before_train()
        train_loader = self.get_train_loader(
            args=self.args,
            dataset_cfg=self.cfg.DATASET.TRAIN,
            dataloader_cfg=self.cfg.DATALOADER.TRAIN,
            is_distributed=self.args.is_distributed,
        )

        time_checker = TimeCheck(self.cfg.TOTAL_EPOCH)
        time_checker.start()
        for epoch in range(self.current_epoch, self.cfg.TOTAL_EPOCH):
            if self.args.is_distributed:
                dist.barrier()
                train_loader.sampler.set_epoch(epoch)
            train_log_dict = self.method.train(
                model=self.model,
                data_loader=train_loader,
                optimizer=self.optimizer,
                is_distributed=self.args.is_distributed,
                world_size=self.args.world_size,
            )

            self.scheduler.step()
            self.current_epoch += 1
            if self.args.is_distributed:
                train_log_dict = self._gather_log(train_log_dict)
            if self.args.is_master:
                self._log_after_epoch(epoch + 1, time_checker, train_log_dict, "train")

    def test(self):
        if self.args.is_master:
            test_loader = self.get_test_loader(
                args=self.args,
                dataset_cfg=self.cfg.DATASET.TEST,
                dataloader_cfg=self.cfg.DATALOADER.TEST,
            )

            self.logger.test()

            for sequence_dataloader in test_loader:
                sequence_name = sequence_dataloader.dataset.sequence_name
                sequence_pred_list = self.method.test(
                    model=self.model, data_loader=sequence_dataloader
                )

                for cur_pred_dict in sequence_pred_list:
                    file_name = cur_pred_dict.pop("file_name")
                    for key in cur_pred_dict:
                        self.logger.save_visualize(
                            image=cur_pred_dict[key],
                            visual_type=key,
                            sequence_name=os.path.join("test", sequence_name),
                            image_name=file_name,
                        )

    def save(self, name):
        checkpoint = self._make_checkpoint()
        self.logger.save_checkpoint(checkpoint, name)

    def load(self, name):
        checkpoint = self.logger.load_checkpoint(name)
        self._init_from_cfg(checkpoint["cfg"])

        self.model.module.load_state_dict(checkpoint["model"])

    def _make_checkpoint(self):
        checkpoint = {
            "epoch": self.current_epoch,
            "args": self.args,
            "cfg": self.cfg,
            "model": self.model.module.state_dict(),
            "optimizer": self.optimizer['optimizer'].state_dict(),
            "optimizer_keypt": self.optimizer['optimizer_keypt'].state_dict(),
            "scheduler": self.scheduler['scheduler'].state_dict(),
            "scheduler_keypt": self.scheduler['scheduler_keypt'].state_dict()
        }

        return checkpoint

    def _gather_log(self, log_dict):
        if log_dict is None:
            return None

        for key in log_dict.keys():
            if isinstance(log_dict[key], SummationMeter) or isinstance(
                log_dict[key], Metric
            ):
                log_dict[key].all_gather(self.args.world_size)

        return log_dict

    def _log_before_train(self):
        self.logger.train()
        self.logger.save_args(self.args)
        self.logger.save_cfg(self.cfg)
        self.logger.log_model(self.model)
        self.logger.log_optimizer(self.optimizer)
        self.logger.save_src(os.path.dirname(os.path.abspath(__file__)))

    def _log_after_epoch(
        self, epoch, time_checker, log_dict, part, isSaveFinal=True, isSaveBest=False
    ):
        # Calculate Time
        time_checker.update(epoch)

        # Log Time
        self.logger.write(
            "Epoch: %d | time per epoch: %s | eta: %s"
            % (epoch, time_checker.time_per_epoch, time_checker.eta)
        )

        # Log Learning Process
        log = "%5s" % part
        for key in log_dict.keys():
            log += " | %s: %s" % (key, str(log_dict[key]))
            if isinstance(log_dict[key], SummationMeter) or isinstance(
                log_dict[key], Metric
            ):
                self.logger.add_scalar(
                    "%s/%s" % (part, key), log_dict[key].value, epoch
                )
            else:
                self.logger.add_scalar("%s/%s" % (part, key), log_dict[key], epoch)
        self.logger.write(log=log)

        if isSaveFinal:
            # Make Checkpoint
            checkpoint = self._make_checkpoint()

            # Save Checkpoint
            self.logger.save_checkpoint(checkpoint, "final.pth")
            if epoch % self.args.save_term == 0:
                self.logger.save_checkpoint(checkpoint, "%d.pth" % epoch)
            if isSaveBest:
                self.logger.save_checkpoint(checkpoint, "best.pth")
                self.logger.add_scalar(
                    "%s/%s" % (part, "metric"), log_dict["BestIndex"].avg, epoch
                )


def _prepare_model(model_cfg, is_distributed=False, local_rank=None, logger=None):
    name = model_cfg.NAME
    parameters = model_cfg.PARAMS

    model = getattr(models, name)(logger=logger, **parameters, is_distributed=is_distributed)

    if is_distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    else:
        model = nn.DataParallel(model).cuda()

    return model


def _prepare_optimizer(optimizer_cfg, model):
    name = optimizer_cfg.NAME
    parameters = optimizer_cfg.PARAMS
    learning_rate = parameters.lr

    params_group = model.module.get_params_group(learning_rate, keypt_lr=optimizer_cfg.KEYPT_PARAMS.lr)
    params_group_nonkeypt = [param_group for param_group in params_group if param_group['lr'] != optimizer_cfg.KEYPT_PARAMS.lr]
    params_group_keypt = [param_group for param_group in params_group if param_group['lr'] == optimizer_cfg.KEYPT_PARAMS.lr]

    optimizer = getattr(optim, name)(params_group_nonkeypt, **parameters)
    optimizer_keypt = getattr(optim, name)(params_group_keypt, **optimizer_cfg.KEYPT_PARAMS)

    return {
        'optimizer': optimizer,
        'optimizer_keypt': optimizer_keypt
    }


def _prepare_scheduler(scheduler_cfg, optimizer):
    name = scheduler_cfg.NAME
    parameters = scheduler_cfg.PARAMS

    if name == "CosineAnnealingWarmupRestarts":
        from utils.scheduler import CosineAnnealingWarmupRestarts

        scheduler = CosineAnnealingWarmupRestarts(optimizer['optimizer'], **parameters)
    else:
        scheduler = getattr(optim.lr_scheduler, name)(optimizer['optimizer'], **parameters)

    # scheduler_keypt = CustomStepLRScheduler(optimizer['optimizer_keypt'], milestones=[90, 120], factor=0.1)
    scheduler_keypt = CosineAnnealingWarmupRestarts(optimizer['optimizer_keypt'], **parameters)

    return {
        'scheduler': scheduler,
        'scheduler_keypt': scheduler_keypt
    }


class CustomStepLRScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, milestones: list, factor: float = 0.1, last_epoch: int = -1):
        self.milestones = milestones
        self.factor = factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.milestones:
            return [base_lr * self.factor for base_lr in self.base_lrs]
        else:
            return [group['lr'] for group in self.optimizer.param_groups]