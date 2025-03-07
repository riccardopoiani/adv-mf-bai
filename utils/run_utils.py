import os
import random
import yaml
import numpy as np

from algorithms.bai import BAIConfig
from algorithms.bai_factory import BAIFactory
from algorithms.learn import learn
from envs.multi_fidelity_env import MultiFidelityBanditModel, MultiFidelityEnvConfig, MultiFidelityEnvironment
from utils.distribution import DistributionFactory
from utils.results import ResultItem, ResultItemWeights


def mkdir_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def fix_seed(seed_val):
    if seed_val is not None:
        os.environ["PYTHONHASHSEED"] = str(seed_val)

        random.seed(seed_val)
        np.random.seed(seed_val)


def read_cfg(env_cfg_path: str):
    with open(env_cfg_path, "r") as f:
        env_cfg = yaml.load(f, Loader=yaml.FullLoader)

    return env_cfg


def build_bai_cfg(bandit_model: MultiFidelityBanditModel, algo_cfg, delta, variance_proxy, run_id, use_prac_th,
                  save_weights, stop_at_n,save_every_x):
    d = {'n_arms': bandit_model.n_arms,
         'm_fidelity': bandit_model.m_fidelity,
         'delta': delta,
         'precisions': bandit_model.fidelity_bounds,
         'costs': bandit_model.costs,
         'variance_proxy': variance_proxy,
         'kl_f': DistributionFactory.get_kl_f(bandit_model.cfg.dist_type, bandit_model.cfg.other_fixed_dist_param),
         'hyper_params': algo_cfg,
         'run_id': run_id,
         'use_prac_th': use_prac_th,
         'save_weights': save_weights,
         'stop_at_n': stop_at_n,
         'save_every_x': save_every_x
         }
    return BAIConfig(**d)


def run(run_id, seed, env_cfg, algo_name, algo_cfg, delta, use_prac_th, save_weights, stop_at_n, save_every_x):
    print(f"Run {run_id} started.")

    # Fix seed
    fix_seed(seed)

    # Instantiate env and agents
    env_cfg = MultiFidelityEnvConfig(**env_cfg)
    env = MultiFidelityEnvironment(MultiFidelityBanditModel(env_cfg))

    bai_cfg = build_bai_cfg(MultiFidelityBanditModel(env_cfg), algo_cfg, delta, env_cfg.get_var_proxy(), run_id,
                            use_prac_th, save_weights, stop_at_n, save_every_x)
    algo = BAIFactory.get_algo(algo_name, bai_cfg)

    # Learn
    best_arm = learn(algo, env)

    print(f"Run {run_id} completed. - Cost {algo.get_cost()} - Best arm {best_arm}")

    # Prepare results
    if save_weights:
        return ResultItemWeights(best_arm,
                                 algo.get_cost(),
                                 algo.get_cost_per_arm_and_fidelity(),
                                 algo._arm_count,
                                 grad_weight_seq=algo.grad_seq,
                                 w_T_seq=algo.w_T_seq
                                 )

    return ResultItem(best_arm,
                      algo.get_cost(),
                      algo.get_cost_per_arm_and_fidelity(),
                      algo._arm_count)
