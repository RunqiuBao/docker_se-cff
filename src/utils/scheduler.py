# base on https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    optimizer (Optimizer): Wrapped optimizer.
    first_cycle_steps (int): First cycle step size.
    cycle_mult(float): Cycle steps magnification. Default: -1.
    max_lr(float): First cycle's max learning rate. Default: 0.1.
    min_lr(float): Min learning rate. Default: 0.001.
    warmup_steps(int): Linear warmup step size. Default: 0.
    gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
    last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        lr_ratio: float = 0.01,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.lr_ratio = lr_ratio
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        self.base_max_lrs = []  # first max learning rate
        self.max_lrs = []  # max learning rate in the current cycle
        self.min_lrs = []  # min learning rate

        for param_group in optimizer.param_groups:
            self.base_max_lrs.append(param_group["lr"])
            self.max_lrs.append(param_group["lr"])
            self.min_lrs.append(param_group["lr"] * self.lr_ratio)

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group, min_lr in zip(self.optimizer.param_groups, self.min_lrs):
            param_group["lr"] = min_lr
            self.base_lrs.append(min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [
                (max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                for max_lr, base_lr in zip(self.max_lrs, self.base_lrs)
            ]
        else:
            return [
                base_lr
                + (max_lr - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi
                        * (self.step_in_cycle - self.warmup_steps)
                        / (self.cur_cycle_steps - self.warmup_steps)
                    )
                )
                / 2
                for max_lr, base_lr in zip(self.max_lrs, self.base_lrs)
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = (
                    int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult)
                    + self.warmup_steps
                )
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(
                        math.log(
                            (
                                epoch / self.first_cycle_steps * (self.cycle_mult - 1)
                                + 1
                            ),
                            self.cycle_mult,
                        )
                    )
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps
                        * (self.cycle_mult**n - 1)
                        / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (
                        n
                    )
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lrs = [
            base_max_lr * (self.gamma**self.cycle) for base_max_lr in self.base_max_lrs
        ]
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class RFDetrRestarts(LambdaLR):
    def __init__(
        self, 
        optimizer, 
        dataset_size, 
        batch_size, 
        world_size, 
        grad_accum_steps, 
        epochs,           # Epochs of one cycle
        total_epochs,     # Total number of epochs for the whole run
        warmup_epochs, 
        lr_scheduler_type,  # 'step' or 'cosine' 
        lr_min_factor, 
        lr_drop_epoch,
    ):        
        def lr_lambda(current_epoch: int):
            # Hard stop if we exceed total_epochs
            if current_epoch >= total_epochs:
                return lr_min_factor

            # Reset the 'clock' at the start of every cycle
            epoch_in_cycle = current_epoch % epochs
            
            # --- Original Logic Applied to Cycle ---
            if epoch_in_cycle < warmup_epochs:
                # Linear warmup
                return float(epoch_in_cycle) / float(max(1, warmup_epochs))
            
            else:
                if lr_scheduler_type == 'cosine':
                    # progress calculated based on steps within the current cycle
                    progress = float(epoch_in_cycle - warmup_epochs) / float(
                        max(1, epochs - warmup_epochs)
                    )
                    return lr_min_factor + (1 - lr_min_factor) * 0.5 * (1 + math.cos(math.pi * progress))
                
                elif lr_scheduler_type == 'step':
                    # Exact logic: 1.0 until drop_step within cycle, then 0.1
                    return 1.0 if epoch_in_cycle < lr_drop_epoch else 0.1
            return 1.0

        super().__init__(optimizer, lr_lambda=lr_lambda)
