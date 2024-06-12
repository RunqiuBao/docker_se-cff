import os
import argparse

import torch

from manager import DLManager
from utils.config import get_cfg

print("baodebug: {}".format(os.environ['LOCAL_RANK']))
# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='/root/code/configs/config.yaml')
parser.add_argument('--data_root', type=str, default='/root/data/DSEC')
parser.add_argument('--save_root', type=str, default='/root/code/save')

parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--save_term', type=int, default=25)
parser.add_argument('--resume_cpt', type=str, default=None)
parser.add_argument('--local-rank', type=int, default=0)  # Note: deprecated. But required by torch.distributed.launch

args = parser.parse_args()

assert int(os.environ['WORLD_SIZE']) >= 1

args.local_rank = int(os.environ['LOCAL_RANK'])
args.is_distributed = True
args.is_master = args.local_rank == 0
args.start_epoch = 0
print("is_master: {}".format(args.is_master))
args.device = 'cuda:%d' % args.local_rank
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')
args.world_size = torch.distributed.get_world_size()
args.rank = torch.distributed.get_rank()

assert os.path.isfile(args.config_path)
assert os.path.isdir(args.data_root)

# Set Config
cfg = get_cfg(args.config_path)

exp_manager = DLManager(args, cfg)
exp_manager.trainAndValid()
