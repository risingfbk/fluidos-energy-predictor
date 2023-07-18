# FLUIDOS Energy Demand Predictor

## Overview

This is a project to predict energy demand for a FLUIDOS node. It uses a neural network that takes as input, given a certain machine,

- the past workload of the machine (at the moment, a week's worth of data)
- a power profile of the machine (more information below)

and outputs the predicted energy demand for the following day.

**This project (as of July 19th, 2023) is currently in development and is not ready for production use. It is provided as-is, with no guarantees of any kind. Features may be added or removed at any time.**

## Installation

Due to the machine-specific nature of the projects, no pre-trained models are provided. To use the project, you will need to train your own model.

You may choose between the following installation methods:

### Classic

Create a `python3` virtual environment using the software of your choice (`venv`, `conda`, or anything else). Install `python3==3.11.4` on it. Then, install the dependencies using `pip install -r requirements.txt`.

On macOS platforms running an Apple Silicon processor, you may need to install `tensorflow-macos` instead of `tensorflow`. [This page](https://developer.apple.com/metal/tensorflow-plugin/) provides precise information on how to correctly install the Tensorflow plugin on these platforms.

### Docker

A Dockerfile is provided within the project. To build the image, run `docker build -t fluidos-energy-demand-predictor .`

You may run the image using classic Docker, `docker-compose`, or any other container orchestration software of your choice.

For your convenience, a `docker-compose.yml` file is provided within the project. Before deploying, make sure to edit the following line:

```
  - /path/to/data/folder:/app/data
  - /path/to/models/folder:/app/models
  - /path/to/output/folder:/app/out
```

assigning the path to a folder on your machine to the:

- `/app/data` folder, which will be used for fetching the training data (read below for more information on the folder structure);
- `/app/models` folder, which will be used for storing the trained models;
- `/app/out` folder, which will be used for storing the predictions and test results.

## Data folder structure

The data folder must contain the following files:
```
├── gcd
├── spec2008_agg
└── spec2008
```

Samples of these files are provided in the *releases* section of this repository. The `gcd` file contains the workload data from the Google Cluster Data ([link](https://github.com/google/cluster-data)), while the SPEC2008 contains power profiles from the SPEC2008 benchmark suite ([link](https://www.spec.org/power_ssj2008/results/power_ssj2008.html)).
The `spec2008_agg` file contains the aggregated power profiles from the SPEC2008 benchmark suite.

For your convenience, scripts for fetching `spec2008` data and generating both `gcd` and `spec2008` data are provided in the `src/datasets/` folder.

Scripts for pulling data from the `gcd` dataset are not included in the scripts, as it requires a Google Cloud Platform account, a project with billing enabled, and a lot of patience. Our data was retrieved with both `bq` and manual downloads, both methods of which are described in the aforementioned link.

## Usage

Run `python3 src/main.py --help` for a list of available commands. The program either catches flags passed to it and if not provided, automatically asks for the required information. 

The program requires at least a model name (`--model`) and  a power curve for the machine (`--power`). Depending on the action (`--action`), the program will either search for hyperparameters, train, or test a certain model (eventually providing a number of epochs for training, `--epochs`).

Models are saved in the `models` folder, and logs and predictions are saved in the `out` folder. The program will automatically create the folders if they do not exist.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project was developed as part of the FLUIDOS project, funded by the European Union's Horizon 2020 research and innovation programme under grant agreement `101070473 - HORIZON-CL4-2021-DATA-01`.
It is an integral part of the Work Package 6 of the project, which aims to define an energy- and carbon-aware computation model that can shift loads both in time and geography; devise cost-effective infrastructure optimisations for industrial environments; use Artificial Intelligence and Machine Learning methods for performance prediction and enhancement.