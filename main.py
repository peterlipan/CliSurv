import os
import random
import torch
import wandb
import argparse
import warnings
import pandas as pd
import numpy as np
from utils import yaml_config_hook, Trainer


def main(args, logger):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trainer = Trainer(args)
    trainer.run()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()

    # Optional config file argument
    parser.add_argument(
        '--config',
        type=str,
        default="eyepacs",
        help='Path to YAML config file'
    )
    parser.add_argument('--debug', action="store_true", help='debug mode (disable wandb)')

    # First parse to get config path
    temp_args, _ = parser.parse_known_args()
    temp_args.config = f"./configs/{temp_args.config}.yaml"
    yaml_config = yaml_config_hook(temp_args.config)

    # Add yaml config items as args
    for k, v in yaml_config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    if not args.debug:
        wandb.login(key="cb1e7d54d21d9080b46d2b1ae2a13d895770aa29")
        config = {k: getattr(args, k) for k in yaml_config.keys()}

        wandb_logger = wandb.init(
            project="OdRep",
            config=config
        )
    else:
        wandb_logger = None

    main(args, wandb_logger)

