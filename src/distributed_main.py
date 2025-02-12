import os
import argparse

import torch

from manager import DLManager
from utils.config import get_cfg

import baodebug

baodebug.debugutils.ConfigureRootLogger("info")  # config logger format


print("baodebug: {}".format(os.environ["LOCAL_RANK"]))
# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="/root/code/configs/config.yaml")
parser.add_argument("--data_root", type=str, default="/root/data/DSEC")
parser.add_argument("--save_root", type=str, default="/root/code/save")

parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--save_term", type=int, default=25)
parser.add_argument("--resume_cpt", type=str, default=None)
parser.add_argument("--only_resume_weight", action='store_true', help="whether only resume network weights")
parser.add_argument("--only_resume_weight_from", type=str, default=None, help="only resume network weights for this subnet.")
parser.add_argument("--is_save_onnx", action='store_true', help="simply save the model to a onnx model at 'data_root'")
parser.add_argument(
    "--local-rank", type=int, default=0
)  # Note: deprecated. But required by torch.distributed.launch
parser.add_argument("--only_test", action="store_true", help="only run test")
args = parser.parse_args()
assert int(os.environ["WORLD_SIZE"]) >= 1

if args.is_save_onnx:
    args.only_test = True

args.local_rank = int(os.environ["LOCAL_RANK"])
args.is_distributed = True
args.is_master = args.local_rank == 0
args.start_epoch = 0
print("is_master: {}".format(args.is_master))
args.device = "cuda:%d" % args.local_rank
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend="nccl", init_method="env://")
args.world_size = torch.distributed.get_world_size()
args.rank = torch.distributed.get_rank()

assert os.path.isfile(args.config_path)
assert os.path.isdir(args.data_root)

# Set Config
cfg = get_cfg(args.config_path)

exp_manager = DLManager(args, cfg)
if args.only_test:
    exp_manager.test()
else:
    exp_manager.trainAndValid()
