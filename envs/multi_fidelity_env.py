from typing import List

import numpy as np

from utils.distribution import GaussianDist, DistributionFactory


def gen_rnd_instance(n_arms: int,
                     m_fid: int,
                     opt_fid_mean_rng: tuple,
                     cost_vec,
                     var: float,
                     order_vec,
                     bias_vec,
                     min_delta,
                     max_delta
                     ):
    # Generate costs
    costs = cost_vec

    # Generate fid bounds

    ok = False

    while not ok:
        # Generate arms
        opt_fid_arms = np.random.uniform(low=opt_fid_mean_rng[0], high=opt_fid_mean_rng[1], size=n_arms).tolist()
        best_val = np.max(opt_fid_arms)
        best_idx = np.argmax(opt_fid_arms)
        ok = True
        for a in range(n_arms):
            if a != best_idx:
                if best_val - opt_fid_arms[a] < min_delta or best_val - opt_fid_arms[a] > max_delta:
                    ok = False

    arms = []
    for arm_idx in range(n_arms):
        arm_dist = []
        anchor = opt_fid_arms[arm_idx]
        for m_idx in range(m_fid - 1):
            mean = np.random.uniform(low=anchor - bias_vec[m_idx] - order_vec[m_idx]/2,
                                     high=anchor + bias_vec[m_idx] + order_vec[m_idx]/2,
                                     size=(1,)).item()
            arm_dist.append(mean)

        arm_dist.append(opt_fid_arms[arm_idx])

        arms.append(arm_dist)

    fid_bounds = [bias_vec[m_idx] + order_vec[m_idx]/2 for m_idx in range(m_fid - 1)]
    fid_bounds.append(0)

    cfg = MultiFidelityEnvConfig(n_arms=n_arms,
                                 m_fidelity=m_fid,
                                 other_fixed_dist_param=var,
                                 dist_type="gaussian",
                                 fidelity_bounds=fid_bounds,
                                 costs=costs,
                                 arms=arms)
    return MultiFidelityBanditModel(cfg)


class MultiFidelityEnvConfig:
    __slots__ = ['costs',
                 'fidelity_bounds',
                 'arms',
                 'dist_type',
                 'other_fixed_dist_param',
                 'n_arms',
                 'm_fidelity',
                 'transpose']

    def __init__(self,
                 n_arms: int,
                 m_fidelity: int,
                 costs: List[float],
                 fidelity_bounds: List[float],
                 arms: List[List[float]],
                 dist_type: str,
                 other_fixed_dist_param,
                 transpose=False):
        if not transpose:
            assert len(arms) == n_arms, f"Expected {m_fidelity}, found {len(arms)}"
            for a in range(n_arms):
                assert len(arms[a]) == m_fidelity, f"Arm {a} found {len(arms[a])}"

        self.n_arms = n_arms
        self.m_fidelity = m_fidelity
        self.costs = costs
        self.fidelity_bounds = fidelity_bounds
        self.dist_type = dist_type
        self.other_fixed_dist_param = other_fixed_dist_param
        self.arms = [[] for _ in range(n_arms)]
        self.transpose = transpose

        if self.transpose:
            for a in range(n_arms):
                for m in range(m_fidelity):
                    self.arms[a].append(DistributionFactory.get_dist(dist_type, arms[m][a], other_fixed_dist_param))
        else:
            for a in range(n_arms):
                for m in range(m_fidelity):
                    self.arms[a].append(DistributionFactory.get_dist(dist_type, arms[a][m], other_fixed_dist_param))

    def get_var_proxy(self):
        return self.arms[0][0].get_var_proxy()


class MultiFidelityBanditModel:

    def __init__(self, cfg: MultiFidelityEnvConfig):
        """
        :param costs: (vector of costs; ordered from fidelity 0 to M-1)
        :param fidelity_bounds: (vector of xi's; ordered from fidelity 0 to M-1
        :param arms: (arm distributions. Outer index is for the arm, inner index for the fidelity of that arm)
        """
        self.cfg = cfg

        # General parameters
        self.costs = cfg.costs
        self.arms = cfg.arms
        self.fidelity_bounds = cfg.fidelity_bounds

        # Store numbers
        self.m_fidelity = cfg.m_fidelity
        self.n_arms = cfg.n_arms

        # Check that everything looks fine
        self._verify()
        self.arms = cfg.arms

    def _verify(self):
        # Check that for each arm there are m fidelities
        assert len(self.arms) == self.n_arms, f"Expected {self.n_arms} Found {len(self.arms)}"
        for arm_idx in range(self.n_arms):
            assert len(
                self.arms[arm_idx]) == self.m_fidelity, f"Expected {self.m_fidelity}, found {len(self.arms[arm_idx])}"

        # Check that the number of fidelity bounds is equal to the number of arms
        assert len(self.fidelity_bounds) == self.m_fidelity

        # Verify that the optimal fidelity is as precise as possible
        assert self.fidelity_bounds[self.m_fidelity - 1] == 0.0

        # Check that fidelity arms satisfy the MF constraint
        for arm_idx in range(self.n_arms):
            for m_idx in range(self.m_fidelity):
                opt_mean = self.arms[arm_idx][self.m_fidelity - 1].get_mean()
                curr_mean = self.arms[arm_idx][m_idx].get_mean()

                assert abs(opt_mean - curr_mean) <= 1e-4 + self.fidelity_bounds[
                     m_idx], f"|{opt_mean}  - {curr_mean}| <= {self.fidelity_bounds[m_idx]} is false"

        # Check that costs are increasing
        assert sorted(self.costs) == self.costs

        # Check that fidelity are decreasing
        assert sorted(self.fidelity_bounds)[::-1] == self.fidelity_bounds

    def get_best_arm(self):
        """
        :return: idx of the best arm
        """
        m = 0
        val_m = self.arms[0][self.m_fidelity - 1].get_mean()
        for i in range(1, self.n_arms):
            if self.arms[i][self.m_fidelity - 1].get_mean() > val_m:
                val_m = self.arms[i][self.m_fidelity - 1].get_mean()
                m = i
        return m

    def get_means(self, arm_idx: int):
        return np.array([self.arms[arm_idx][m].get_mean() for m in range(self.m_fidelity)])

    def to_dict(self):
        res = {}
        for arm in range(self.n_arms):
            res[f"mean_{arm}"] = self.get_means(arm)

        res['cost'] = self.costs
        res['xi'] = self.fidelity_bounds

        return res


class MultiFidelityEnvironment:

    def __init__(self, model: MultiFidelityBanditModel):
        self.model = model

    def step(self, arm_idx: int, fid_idx: int) -> (float, float):
        """
        :param arm_idx: index  of the arms to be pulled
        :param fid_idx: fidelity at which the arm will be pulled
        :return: (reward, cost)
        """
        assert self.model.n_arms > arm_idx >= 0, f"Attempting to play arm {arm_idx}"
        assert self.model.m_fidelity > fid_idx >= 0, f"Attempting to play fidelity {fid_idx}"

        return (self.model.arms[arm_idx][fid_idx].sample(),
                self.model.costs[fid_idx])
