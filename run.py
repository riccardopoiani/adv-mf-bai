import argparse
import os
import pickle
import time

import numpy as np
import yaml
from joblib import Parallel, delayed

from envs.multi_fidelity_env import MultiFidelityBanditModel, MultiFidelityEnvConfig
from utils.math import conf_interval
from utils.results import ResultSummary, ResultSummaryWeights
from utils.run_utils import read_cfg, mkdir_if_not_exist, run

np.set_printoptions(suppress=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",
                        type=str,
                        required=True,
                        choices=["IISE",
                                 "GRAD",
                                 "MF-GRAD",
                                 "LUCBExploreA",
                                 "LUCBExploreB"])
    parser.add_argument("--env-cfg", type=str, required=True)
    parser.add_argument("--algo-cfg", type=str)
    parser.add_argument("--results-dir", type=str, default="results/")
    parser.add_argument("--delta", type=float, default=0.001)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--use-prac-th", type=int, default=0)
    parser.add_argument("--save-weights", type=int, default=0)
    parser.add_argument("--stop-at-n", type=int, default=0)
    parser.add_argument("--save-every-x", type=int, default=0)

    # Reading common arguments and read environment configuration
    args = parser.parse_args()
    env_cfg = read_cfg(args.env_cfg)

    algo_cfg = None
    if args.algo_cfg is not None:
        algo_cfg = read_cfg(args.algo_cfg)

    use_prac_th = False
    if args.use_prac_th > 0:
        use_prac_th = True

    time_start = time.time()

    # Launch pure-exploration
    seeds = [np.random.randint(1000000) for _ in range(args.n_runs)]
    if args.n_jobs == 1:
        results = [run(run_id=id,
                       seed=seed,
                       env_cfg=env_cfg,
                       algo_cfg=algo_cfg,
                       algo_name=args.algo,
                       delta=args.delta,
                       use_prac_th=use_prac_th,
                       save_weights=args.save_weights,
                       stop_at_n=args.stop_at_n,
                       save_every_x=args.save_every_x)
                   for id, seed in zip(range(args.n_runs), seeds)]
    else:
        results = Parallel(n_jobs=args.n_jobs, backend='loky')(
            delayed(run)(run_id=id,
                         seed=seed,
                         env_cfg=env_cfg,
                         algo_cfg=algo_cfg,
                         algo_name=args.algo,
                         delta=args.delta,
                         use_prac_th=use_prac_th,
                         save_weights=args.save_weights,
                         stop_at_n=args.stop_at_n,
                         save_every_x=args.save_every_x
                         )
            for id, seed in zip(range(args.n_runs), seeds))

    # Dump results on file
    mkdir_if_not_exist(args.results_dir)
    with open(os.path.join(args.results_dir, "results.pkl"), "wb") as output:
        pickle.dump(results, output)

    if args.save_weights:
        res_summary = ResultSummaryWeights(results, MultiFidelityBanditModel(MultiFidelityEnvConfig(**env_cfg)))
    else:
        res_summary = ResultSummary(results, MultiFidelityBanditModel(MultiFidelityEnvConfig(**env_cfg)))

    time_end = time.time()

    time_dict = {'total': time_end - time_start,
                 'start': time_start,
                 'end': time_end}

    # Dump results
    summary = {
        'env_cfg': env_cfg,
        'algo': args.algo,
        'algo_cfg': algo_cfg,
        'delta': args.delta,
        'n_runs': args.n_runs,
        'results':
            {
                'correctness': res_summary.best_arm_stats(),
                'cost_complexity':
                    {
                        'mean': res_summary.cost_complexity_stats()[0].item(),
                        'std': res_summary.cost_complexity_stats()[1].item(),
                        'ci': conf_interval(res_summary.cost_complexity_stats()[1], args.n_runs).item()
                    }
            },
        'time': time_dict,
        'stop_at_n': args.stop_at_n,
        'save_every_x': args.save_every_x,
        'use_prac_th': args.use_prac_th
    }
    print(summary['results'])

    with open(os.path.join(args.results_dir, "run_specs.yml"), 'w') as outfile:
        yaml.dump(summary, outfile, default_flow_style=False)
