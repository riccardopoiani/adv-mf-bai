import os
import yaml
from envs.multi_fidelity_env import gen_rnd_instance

n_arms = 4
m_fidelity = 5
delta = 0.1
algo_cfg = {'lr': 0.1,
            'use_theo_th': True,
            'verbosity': None}  # change weighted to any other string to test the other type of projection
var_proxy = 0.25

correctness = 0
bai_list = []
mf_bai_list = []

model = gen_rnd_instance(n_arms=n_arms,
                         m_fid=m_fidelity,
                         opt_fid_mean_rng=(0, 1),
                         cost_vec=[0.05, 0.1, 0.2, 0.4, 5],
                         bias_vec=[0.075, 0.06, 0.04, 0.02, 0],
                         order_vec=[0.05, 0.04, 0.02, 0.01, 0],
                         var=var_proxy,
                         min_delta=0.1,
                         max_delta=0.4)
arm_list = [[] for a in range(n_arms)]
for a in range(model.n_arms):
    print(f"Mean of arm {a}: {[model.get_means(a)[m] for m in range(m_fidelity)]}")
    for m in range(m_fidelity):
        arm_list[a].append(float(model.get_means(a)[m]))
print(f"Cost: {model.costs}")
print(f"xi: {model.fidelity_bounds}")

cfg = {
    'fidelity_bounds': model.fidelity_bounds,
    'costs': model.costs,
    'arms': arm_list,
    'dist_type': 'gaussian',
    'other_fixed_dist_param': 0.1,
    'n_arms': n_arms,
    'm_fidelity': m_fidelity,
}

with open(os.path.join("configs/cfg4x5.yml"), 'w') as outfile:
    yaml.dump(cfg, outfile, default_flow_style=False)
