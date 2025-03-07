# Code for the paper "Optimal Multi-Fidelity Best-Arm Identification"

The file `requirements.txt` provides the dependency for the project. 
We relied on python 3.8 for the development of this project.

### Available Algorithms
This repository contains the implementation of the following best-arm identification algorithms for the multi-fidelity setting:
- Iterative and Imprecise Successive Elimination (IISE), "Multi-Fidelity Best-Arm Identification", Poiani et al., (NeurIPS 2022)
- Multi-Fidelity LUCB, "Multi-Fidelity Multi-Armed Bandits Revisited", Wang et al., (NeurIPS 2023). Specifically, we provide an implementation of ExploreA and ExploreB strategies that are provided in the paper. We denote these algorithms with LUCBExploreA and LUCBExploreB respectively.
- Gradient Ascent for BAI (GRAD) "Gradient Ascent for Active Exploration in Bandit Problems", Ménard, 2019
- Multi-Fidelity Gradient Ascent (MF-GRAD).

### Running the code
To run the code, we provide a `run.py` script. A typical usage is
`python run.py --algo IISE --env-cfg configs/cfg.yml --results-dir results/ --delta 0.1 --n-jobs 5 --n-runs 10`.
This command will run the IISE algorithm on the multi-fidelity bandit model described in the
configuration file `configs/cfg.yml`. More specifically, the experiment is launched `10` times (i.e., `--n-runs`), 
and in parallel over `5` (i.e., `--n-jobs`) cores. The parameter `--delta` specifies the risk parameter of the 
underlying MF-BAI problem.

Concerning `LUCBExploreA` and `LUCBExploreB` we need to specify also `--algo-cfg`, which contains additional parameters
that are needed by the algorithms to work.

Concerning `GRAD` and `MF-GRAD` we also need to specify a config file with `--alg-cfg`, which contains the learning rate. 
Specifically, `configs/grad/grad_cfg_1.yml` uses the learning rate prescribed by the theory, while `configs/grad/grad_cfg_2.yml`
a constant one of 0.25. When running, it is possible to store empirical cost proportions and gradients by adding `--save-weights 1 --stop-at-n 100000 --save-every-x 50`.
This will run the algorithm for `100000` iterations storing gradients/empirical cost proportions every `50` iterations.

All the configurations file are provided within the `config` folder.

We briefly list here the env configuration available in the folder:
- `cfg4x5.yml` is the config file for the 4x5 bandit model in the main text
- `cfg5x2.yml` is the config file for the 5x2 bandit model in the main text
- `mu2.yml` and `mu.yml` are config files for additional bandit models that have been tested in the appendix.


### Reproducing the non-terminating behavior of LUCBExploreA and LUCBExploreB
To reproduce the non-stopping of LUCBExploreA and LUCBExploreB, we provide the following configuration `configs/forever.yml`,
`configs/lucb_only_M.yml` and `configs/lucbforever/lucb_cfg.yml`.

Specifically, to run LUCB (Kalyanakrishnan et al., 2012) that performs identification using only samples at fidelity 2, one can run the following
command `python run.py --algo ALGO --env-cfg configs/lucb_only_M.yml --results-dir results/onlyM --delta 0.01 --n-jobs 100 --n-runs 1 --algo-cfg configs/lucbforever/lucb_cfg.yml`,
where `ALGO` is any algorithm within `LUCBExploreA` and `LUCBExploreB` (indeed, the implementation of the two approaches when only fidelity M is available is identical to the LUCB algorithm).

To reproduce the non-stopping behavior, instead, run `python run.py --algo LUCBExploreA --env-cfg configs/forever.yml --results-dir results/lucbA --delta 0.01 --n-jobs 100 --n-runs 1 --algo-cfg configs/lucbforever/lucb_cfg.yml`
and `python run.py --algo LUCBExploreB --env-cfg configs/forever.yml --results-dir results/lucbB --delta 0.01 --n-jobs 100 --n-runs 1 --algo-cfg configs/lucbforever/lucb_cfg.yml --stop-at-n 100000000`.

By changing `--stop-at-n`, one can change the maximum duration of the algorithm. 
