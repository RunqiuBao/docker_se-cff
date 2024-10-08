import os
import argparse

from manager import DLManager
from utils.config import get_cfg

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="../configs/config.yaml")
parser.add_argument("--data_root", type=str, default="/root/data/DSEC")
parser.add_argument("--save_root", type=str, default="/root/code/save")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--save_term", type=int, default=25)
parser.add_argument("--resume_cpt", type=str, default=None)

args = parser.parse_args()

args.is_distributed = False
args.is_master = True
args.world_size = 1
args.local_rank = 0
args.start_epoch = 0

assert os.path.isfile(args.config_path)
assert os.path.isdir(args.data_root)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set Config
cfg = get_cfg(args.config_path)

exp_manager = DLManager(args, cfg)
if args.checkpoint is not None:
    exp_manager.load(args.checkpoint)
# exp_manager.train()
# exp_manager.test()
exp_manager.trainAndValid()
