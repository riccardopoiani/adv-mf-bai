from abc import ABC
from typing import List

import numpy as np

from algorithms.bai import BaiAlgorithm, BAIConfig
from utils.math import update_mean_online, hoeffding_anytime

"""
Algorithms from the Paper
"Multi-Fidelity Best-Arm Identification", Poiani et al., (NeurIPS 2022)
"""


class IISE(BaiAlgorithm, ABC):
    NAME = "IISE"

    def __init__(self, cfg: BAIConfig):
        super(IISE, self).__init__(cfg)

        self._curr_phase = 0
        self._thresholds = self.compute_thresholds()
        self._init_curr_phase()

        self.t = 0

    def stopping_condition(self) -> bool:
        if 0 < self.cfg.stop_at_n < self.t:
            return True

        if len(self.get_active_set()) == 1:
            return True
        return False

    def pull_arm(self) -> (List[int], List[int]):
        if self.cfg.save_weights:
            self.w_T_seq.append(self.compute_w_T())

        # Pull all the active arms at the best fidelity
        return list(self.get_active_set()), [self._curr_phase for _ in self.get_active_set()]

    def recommendation(self) -> int:
        # Best arm identified. The method is available only when we have stopped
        assert self.stopping_condition()
        return list(self.get_active_set())[0]

    def update(self, arm_idxes: List[int], fidelity: List[int], rewards: List):
        self.t += len(rewards)

        # Mean and confidence intervals
        self._mean_hat[self._curr_phase][arm_idxes] = update_mean_online(
            self._arm_count[self._curr_phase][arm_idxes],
            self._mean_hat[self._curr_phase][arm_idxes],
            np.array(rewards))
        self._arm_count[self._curr_phase][arm_idxes] += 1

        # Compute confidence intervals
        conf = self.compute_ci(arm_idxes, self._curr_phase)

        # Eliminate arms
        max_lb = np.max(self._mean_hat[self._curr_phase][arm_idxes] - conf - self._precisions[self._curr_phase])
        eliminated_arms = \
            np.where(max_lb >= self._mean_hat[self._curr_phase][arm_idxes] + conf + self._precisions[self._curr_phase])[
                0]
        eliminated_arms = [arm_idxes[e] for e in eliminated_arms]
        for a in eliminated_arms:
            self._active_set.remove(a)

        # Switch phase
        self.switch_phase()

    def compute_ci(self, arm_idxes, fid):
        # if self.cfg.use_prac_th:
        t = np.array(self._arm_count).sum()
        num_1 = self.cfg.variance_proxy * np.log(np.log(t))
        num_2 = self.cfg.variance_proxy * np.log(self.n_arms * self.m_fidelity / self.cfg.delta)
        den = self._arm_count[fid]
        return np.sqrt(2 * (num_1 + num_2) / den)[arm_idxes]

        # return hoeffding_anytime(self._arm_count[self._curr_phase][arm_idxes],
        #                         self.delta / (self.n_arms * self.m_fidelity),
        #                         self._variance_proxy)

    def switch_phase(self):
        curr_t = np.max(self._arm_count[self._curr_phase])
        conf = hoeffding_anytime(curr_t, self.delta / (self.n_arms * self.m_fidelity), self._variance_proxy)
        if self._thresholds[self._curr_phase] - 4 * self._precisions[self._curr_phase] >= 4 * conf:
            self._curr_phase += 1
            self._init_curr_phase()

    def compute_thresholds(self) -> List[float]:
        th = []
        for m in range(self.m_fidelity - 1):
            all_vals = [4 * self._precisions[m] + 4 * (self._precisions[m] - self._precisions[k]) * np.sqrt(
                self._costs[m]) / (
                                np.sqrt(self._costs[k]) - np.sqrt(self._costs[m]))
                        for k in range(m + 1, self.m_fidelity)]
            curr_th = max(all_vals)
            assert curr_th > 0
            th.append(curr_th)
        th.append(0)
        return th

    def _init_curr_phase(self):
        while self._curr_phase < self.m_fidelity and self._thresholds[self._curr_phase] == np.inf:
            self._curr_phase += 1
