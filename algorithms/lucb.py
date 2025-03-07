from abc import abstractmethod
from typing import List
import numpy as np

from algorithms.bai import BaiAlgorithm, BAIConfig
from utils.math import update_mean_online

"""
Algorithms from the paper:
"Multi-Fidelity Multi-Armed Bandits Revisited", Wang et al. (NeurIPS 2023)
"""


class LUCB(BaiAlgorithm):

    def __init__(self, cfg: BAIConfig):
        super(LUCB, self).__init__(cfg)

        self.mu_1_tilde = self.cfg.hyper_params['mu_1_tilde']
        self.mu_2_tilde = self.cfg.hyper_params['mu_2_tilde']

        self.lt = 0
        self.ut = 1
        self.t = 1

    def stopping_condition(self) -> bool:
        if 0 < self.cfg.stop_at_n <= self.t:
            return True

        ucb = self.compute_ucb(self.ut)
        lcb = self.compute_lcb(self.lt)

        if lcb > ucb:
            return True
        return False

    def pull_arm(self) -> (List[int], List[int]):
        m_lt = self.explore(self.lt)
        m_ut = self.explore(self.ut)

        # Handle the fact that elements can be lists...
        if isinstance(m_lt, list):
            a_lt = [self.lt for _ in m_lt]
        else:
            a_lt = [self.lt]
            m_lt = [m_lt]

        if isinstance(m_ut, list):
            a_ut = [self.ut for _ in m_ut]
        else:
            a_ut = [self.ut]
            m_ut = [m_ut]

        return a_lt + a_ut, m_lt + m_ut

    def recommendation(self) -> int:
        print(f"Run {self.cfg.run_id} - Iter {self.t}")
        return self.lt

    def update(self, arm_idxes: List[int], fidelity: List[int], rewards: List):
        # Update time counter
        self.t += len(rewards)

        if self.cfg.save_weights:
            self.w_T_seq.append(self.compute_w_T())

        # Update statistics
        for a, m, r in zip(arm_idxes, fidelity, rewards):
            self._mean_hat[m][a] = update_mean_online(
                self._arm_count[m][a],
                self._mean_hat[m][a],
                r
            )
            self._arm_count[m][a] += 1

        # Update lt and ut
        ucb_vals = np.array([self.compute_ucb(a) for a in range(self.n_arms)])
        ind = np.argpartition(ucb_vals, -2)[-2:]
        ind = ind[np.argsort(ucb_vals[ind])][::-1]

        self.lt = ind[0]
        self.ut = ind[1]

    def compute_ucb(self, arm_idx):
        ucb_vals = [self._mean_hat[m][arm_idx] + self._precisions[m] + self.compute_beta(arm_idx, m)
                    for m in range(self.m_fidelity)]
        return min(ucb_vals)

    def compute_lcb(self, arm_idx):
        lcb_vals = [self._mean_hat[m][arm_idx] - self._precisions[m] - self.compute_beta(arm_idx, m)
                    for m in range(self.m_fidelity)]
        return max(lcb_vals)

    def compute_beta(self, a_idx: int, m_idx: int):
        return np.sqrt(2 * self.cfg.variance_proxy *
            np.log(4 * self.n_arms * self.m_fidelity * self.t ** 4 / self.cfg.delta) / self._arm_count[m_idx][a_idx])

    @abstractmethod
    def explore(self, arm_idx):
        pass


class LUCBExploreA(LUCB):
    NAME = "LUCBExploreA"

    def __init__(self, cfg: BAIConfig):
        super(LUCBExploreA, self).__init__(cfg)

    def explore(self, arm_idx):
        return np.argmax(np.array([self.f_ucb(arm_idx, m) for m in range(self.m_fidelity)]))

    def f_ucb(self, a, m):
        tot_a = sum([self._arm_count[fid][a] for fid in range(self.m_fidelity)])

        if a != self.lt:
            t1 = (self.mu_1_tilde - (self._mean_hat[m][a] + self._precisions[m])) / (self._costs[m] ** (1 / 2))
        else:
            t1 = ((self._mean_hat[m][a] - self._precisions[m]) - self.mu_2_tilde) / (self._costs[m] ** (1 / 2))

        t2 = np.sqrt(2 * np.log(tot_a) / (self._costs[m] * self._arm_count[m][a]))
        return t1 + t2


class LUCBExploreB(LUCB):
    NAME = "LUCBExploreB"

    def __init__(self, cfg: BAIConfig):
        super(LUCBExploreB, self).__init__(cfg)

        self.is_fixed_k = [False for _ in range(self.n_arms)]
        self.m_star = [-1 for _ in range(self.n_arms)]

    def explore(self, arm_idx):
        if not self.is_fixed_k[arm_idx]:
            return [m for m in range(
                self.m_fidelity)]

        return self.m_star[arm_idx]

    def update(self, arm_idxes: List[int], fidelity: List[int], rewards: List):
        super().update(arm_idxes, fidelity, rewards)

        for a in range(self.n_arms):
            if not self.is_fixed_k[a]:
                # Compute \hat{m}^*_k and eventually update is_fixed
                delta_hat_list = []
                for m in range(self.m_fidelity):
                    if a != self.lt:
                        delta_hat = (self.mu_1_tilde - (self._mean_hat[m][a] + self._precisions[m]))
                    else:
                        delta_hat = ((self._mean_hat[m][a] - self._precisions[m]) - self.mu_2_tilde)
                    delta_hat_list.append(delta_hat)
                delta_hat_list = np.array(delta_hat_list)

                vals = delta_hat_list / np.sqrt(np.array(self._costs))

                max_m = np.argmax(vals)
                if vals[max_m] >= 3 * np.sqrt(self.cfg.variance_proxy * np.log(4 * self.n_arms * self.m_fidelity / self.cfg.delta) / (
                                self._costs[0] * self._arm_count[max_m][a])):
                    self.is_fixed_k[a] = True
                    self.m_star[a] = max_m
                    print(f"{max_m} is fixed for {a}")
