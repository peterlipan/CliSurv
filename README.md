# CliSurv

This repository contains the official implementation of **CliSurv: A Deep Cumulative-Link Family for Calibrated Semi-Parametric Survival Modeling**.

The code supports all experiments reported in the paper and is organized to facilitate reproducibility and extension.

## Requirements

We recommend using Conda to manage dependencies. To create the environment with all required packages, run:

```
conda env create -f environment.yml
```

Then activate the environment accordingly.

## Running Experiments

Experiments are configured via dataset-specific config files. To run an experiment, specify the corresponding config name and optionally enable debug mode to disable Weights & Biases logging.

Example (RtmGBSG dataset):

```
python3 main.py --debug --config rtmgbsg
```

Supported datasets and their corresponding config names are:

- `rtmgbsg`
- `flchain`
- `gbsg`
- `metabric`
- `nwtco`
- `support`
- `sac` (Data efficiency and robustness to discretization granularity)

**Note:** Config names should be provided in lowercase.

## Code Structure

- `models/CliSurv.py`: Core implementation of the CliSurv model family, including different link functions and likelihood formulations.
- `main.py`: Entry point for training and evaluation.
- `configs/`: Dataset-specific experiment configurations.