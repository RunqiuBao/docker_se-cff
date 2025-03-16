import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch.distributed as dist
import copy
import re
from typing import Optional

from components import models as MODELCLASSES
from components import datasets
from components import methods

from utils.logger import ExpLogger, TimeCheck
from utils.metrics import SummationMeter, Metric

import logging
logger = logging.getLogger(__name__)


class DLManager:
    scaler = None
    ema = None

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

        self.models = _prepare_models(
            self.cfg.MODEL,
            self.cfg.LOSSES,
            is_distributed=self.args.is_distributed,
            local_rank=self.args.local_rank if self.args.is_distributed else None,
            logger=self.logger,
        )

        self.optimizer = _prepare_optimizer(self.cfg.OPTIMIZER, self.models)
        self.scheduler = _prepare_scheduler(self.cfg.SCHEDULER, self.optimizer)
        if "LEARNING_CONFIG" in self.cfg:
            # techniques to boost training performance
            self.scaler = _prepare_scaler(self.cfg.LEARNING_CONFIG)  # for amp training
            self.ema = _prepare_ema(self.cfg.LEARNING_CONFIG, self.models)

        if self.args.resume_cpt is not None:
            device = torch.device(f"cuda:{self.args.local_rank}")
            checkpoint = torch.load(self.args.resume_cpt, map_location=device)
            if self.logger:
                self.logger.write(
                    "loading checkpoint {} ...".format(self.args.resume_cpt)
                )
            # FIXME: adding 'module.' to each key in model state dict
            for keyModel, model in self.models.items():
                model_statedict = model.state_dict()
                for key, value in checkpoint["models"][keyModel].items():
                    if self.args.only_resume_weight_from is not None:
                        if self.args.only_resume_weight_from not in key:
                            continue
                    # if 'keypt1_predictor' in key or 'keypt2_predictor' in key or '_keypt_feature_extraction_net' in key:
                    #     continue
                    if "module." + key in model_statedict and model_statedict["module." + key].size() == value.size():
                        model_statedict["module." + key] = value
                    else:
                        print("Skipping parameter {} due to not previously exist or size mismatch.".format(key))
                self.models[keyModel].load_state_dict(model_statedict)
            if not self.args.only_resume_weight:
                for key, optimizer in self.optimizer.items():
                    self.optimizer[key].load_state_dict(checkpoint["optimizer"][key])
                for key, scheduler in self.scheduler.items():
                    self.scheduler[key].load_state_dict(checkpoint["scheduler"][key])
                self.args.start_epoch = checkpoint["epoch"] + 1
                self.current_epoch = self.args.start_epoch
                print("resumed old training states.")  

        self.get_train_loader = getattr(
            datasets, self.cfg.DATASET.TRAIN.NAME
        ).get_dataloader
        self.get_valid_loader = getattr(
            datasets, self.cfg.DATASET.VALID.NAME
        ).get_dataloader
        self.get_test_loader = getattr(datasets, self.cfg.DATASET.TEST.NAME).get_dataloader

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

        # profiling the network
        for key, model_cfg in self.cfg.MODEL.items():
            netWorkClass = getattr(MODELCLASSES, model_cfg.CLASSNAME)
            parameters = model_cfg.PARAMS
            loss_cfg = self.cfg.LOSSES[key]
            profile_model = netWorkClass(parameters, loss_cfg, model_cfg["is_freeze"], logger=self.logger, is_distributed=self.args.is_distributed)
            flops, numParams = netWorkClass.ComputeCostProfile(profile_model)
            if self.args.is_master:
                self.logger.write(
                    "[Profile] model(%s) computation cost: gFlops %f | numParams %f M"
                    % (model_cfg.CLASSNAME, float(flops / 10**9), float(numParams / 10**6))
                )
            del profile_model

        # freeze model gradients if static:
        self.method.freeze_static_components(self.models)

        time_checker = TimeCheck(self.cfg.TOTAL_EPOCH)
        time_checker.start()
        smallestValidEPE = sys.float_info.max
        for epoch in range(self.args.start_epoch, self.cfg.TOTAL_EPOCH):
            if self.args.is_distributed:
                dist.barrier()
                train_loader.sampler.set_epoch(epoch)
            
            train_log_dict = self.method.train(
                models=self.models,
                data_loader=train_loader,
                optimizer=self.optimizer,
                scaler=self.scaler,
                ema=self.ema,
                clip_max_norm=self.cfg.LEARNING_CONFIG.clip_max_norm if self.scaler is not None else None,
                is_distributed=self.args.is_distributed,
                world_size=self.args.world_size,
                epoch=epoch,
                tensorBoardLogger=self.logger
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
                models={key: one_ema_model.module for key, one_ema_model in self.ema.items()} if self.ema else {key: model.module for key, model in self.models.items()},
                data_loader=valid_loader,
                is_distributed=self.args.is_distributed,
                world_size=self.args.world_size,
                tensorBoardLogger=self.logger,
                epoch=epoch
            )

            if self.args.is_distributed:
                valid_log_dict = self._gather_log(valid_log_dict)
            if self.args.is_master:
                self._log_after_epoch(
                    epoch + 1,
                    time_checker,
                    valid_log_dict,
                    "valid",
                    isSaveBest=valid_log_dict["Loss"].avg < smallestValidEPE,
                )

            if valid_log_dict["Loss"].avg < smallestValidEPE:
                smallestValidEPE = valid_log_dict["Loss"].avg

            self.current_epoch += 1

    def test(self):
        test_loader = self.get_test_loader(
            args=self.args,
            dataset_cfg=self.cfg.DATASET.TEST,
            dataloader_cfg=self.cfg.DATALOADER.TEST,
        )

        self.logger.test()

        for sequence_dataloader in test_loader:            
            sequence_name = sequence_dataloader.dataset.sequence_name
            self.method.test(
                models=self.models,
                data_loader=sequence_dataloader,
                sequence_name=sequence_name,  # Note: for saving debug images
                save_root=self.args.save_root,
                is_save_onnx=self.args.is_save_onnx
            )

    def save(self, name):
        checkpoint = self._make_checkpoint()
        self.logger.save_checkpoint(checkpoint, name)

    def load(self, name):
        checkpoint = self.logger.load_checkpoint(name)
        self._init_from_cfg(checkpoint["cfg"])

        self.model.module.load_state_dict(checkpoint["model"])

    def _make_checkpoint(self):
        models_checkpoint = {key: model.module.state_dict() for key, model in self.models.items()}
        optimizers_checkpoint = {key: optimizer.state_dict() for key, optimizer in self.optimizer.items()}
        schedulers_checkpoint = {key: scheduler.state_dict() for key, scheduler in self.scheduler.items()}
        checkpoint = {
            "epoch": self.current_epoch,
            "args": self.args,
            "cfg": self.cfg,
            "models": models_checkpoint,
            "optimizer": optimizers_checkpoint,
            "scheduler": schedulers_checkpoint,
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
        for key, model in self.models.items():
            logger.debug("# ---------------------------- model {} --------------------------------".format(model.__class__.__name__))
            # self.logger.log_model(model)
        # self.logger.log_optimizer(self.optimizer)
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


def _prepare_models(models_cfg: dict, losses_cfg: dict, is_distributed: bool = False, local_rank: Optional[int] = None, logger=None):
    models = {}
    for key, model_cfg in models_cfg.items():
        classname = model_cfg.CLASSNAME
        parameters = model_cfg.PARAMS
        loss_cfg = losses_cfg[key]

        model = getattr(MODELCLASSES, classname)(parameters, loss_cfg, model_cfg["is_freeze"], logger=logger, is_distributed=is_distributed)

        if is_distributed:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
            model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model = nn.DataParallel(model).cuda()
        
        models[model_cfg.DICTKEY] = model

    return models


def _prepare_optimizer(optimizers_cfg: dict, models):
    for optimizer_cfg in optimizers_cfg.values():
        if not optimizer_cfg.is_enable:
            continue
        name = optimizer_cfg.NAME
        dict_optimizer = {}
        for key, model in models.items():
            if "NETWORK_TYPE" in optimizer_cfg and optimizer_cfg.NETWORK_TYPE == "DETR":
                params = get_optim_params(optimizer_cfg.PARAMS, model)
                module_kwargs = {
                    "params": params,
                    "lr": optimizer_cfg.PARAMS.lr,
                    "betas": optimizer_cfg.PARAMS.betas,
                    "weight_decay": optimizer_cfg.PARAMS.weight_decay
                }
                optimizer = getattr(optim, name)(**module_kwargs)
                dict_optimizer[key] = optimizer
            else:
                # keypt prediction network
                parameters = optimizer_cfg.PARAMS
                learning_rate = parameters.lr

                if hasattr(model.module, "get_params_group"):
                    params_group = model.module.get_params_group(learning_rate)
                else:
                    params_group = get_optim_params(optimizer_cfg.PARAMS, model)
                optimizer = getattr(optim, name)(params_group, **parameters)
                dict_optimizer[key] = optimizer
        break
    return dict_optimizer


def _prepare_scheduler(schedulers_cfg: dict, optimizers):
    for key, scheduler_cfg in schedulers_cfg.items():
        if not scheduler_cfg.is_enable:
            continue
        dict_scheduler = {}
        for key, optimizer in optimizers.items():
            if scheduler_cfg.get("NAME", None) == "DETR_SCHEDULER":
                name = scheduler_cfg.PARAMS.lr_scheduler.type
                parameters = scheduler_cfg.PARAMS.lr_scheduler.params

                scheduler = getattr(optim.lr_scheduler, name)(optimizer, **parameters)

                dict_scheduler[key] = scheduler
            elif scheduler_cfg.get("NAME", None) == "CosineAnnealingWarmupRestarts":
                name = scheduler_cfg.NAME
                parameters = scheduler_cfg.PARAMS

                if name == "CosineAnnealingWarmupRestarts":
                    from utils.scheduler import CosineAnnealingWarmupRestarts

                    scheduler = CosineAnnealingWarmupRestarts(optimizer, **parameters)
                else:
                    scheduler = getattr(optim.lr_scheduler, name)(optimizer, **parameters)

                dict_scheduler[key] = scheduler
            else:
                raise NotImplementedError
        break
    return dict_scheduler


def _prepare_scaler(learning_cfg):
    """
    prepare scaler for automatic mixed precision learning
    """
    if learning_cfg.use_amp:
        scaler = torch.cuda.amp.grad_scaler.GradScaler()
        return scaler
    else:
        return None
    

def _prepare_ema(learning_cfg, models):
    """
    prepare ema for exponential moving average training
    """
    ema = {}
    if learning_cfg.use_ema:
        for key, model in models.items():
            ema[key] = methods.ema.ModelEMA(model, **learning_cfg.ema.params)
        return ema
    else:
        return None


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


def get_optim_params(cfg: dict, model: nn.Module):
    """
    Used in RTDetr to select submodule params and control their learning rate differently. 
    E.g.:
        ^(?=.*a)(?=.*b).*$  means including a and b
        ^(?=.*(?:a|b)).*$   means including a or b
        ^(?=.*a)(?!.*b).*$  means including a, but not b
    """
    cfg = copy.deepcopy(cfg)

    if 'params' not in cfg:
        return model.parameters() 

    assert isinstance(cfg['params'], list), ''

    param_groups = []
    visited = []
    for pg in cfg['params']:
        pattern = pg['params']
        params = {k: v for k, v in model.named_parameters() if v.requires_grad and len(re.findall(pattern, k)) > 0}
        pg['params'] = params.values()
        param_groups.append(pg)
        visited.extend(list(params.keys()))
        # print(params.keys())

    names = [k for k, v in model.named_parameters() if v.requires_grad]

    if len(visited) < len(names):
        unseen = set(names) - set(visited)
        params = {k: v for k, v in model.named_parameters() if v.requires_grad and k in unseen}
        param_groups.append({'params': params.values()})
        visited.extend(list(params.keys()))
        # print(params.keys())

    assert len(visited) == len(names), ''

    return param_groups

