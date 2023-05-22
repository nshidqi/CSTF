import argparse
import math
import random
import shutil
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.zoo import models

from tqdm import tqdm
from pytorch_msssim import ms_ssim

from torchinfo import summary


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="stf",
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    ) 
    parser.add_argument(
		"--cuda", action="store_true", help="Use cuda"
	)
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
	
    return args


def main(argv):
    args = parse_args(argv)
    print(args)


    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    net = models[args.model]()
    net = net.to(device)
    
    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    # Load Model Checkpoint
    print("Loading", args.checkpoint)
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint["state_dict"])

    model_stats = summary(net, (1, 3, 256, 256))
    summary_str = str(model_stats)

if __name__ == "__main__":
    main(sys.argv[1:])