from abc import ABC, abstractmethod
from typing import List, Set

import numpy as np


class BAIConfig:
    __slots__ = ['n_arms',
                 'm_fidelity',
                 'delta',
                 'precisions',
                 'costs',
                 'variance_proxy',
                 'kl_f',
                 'hyper_params',
                 'discard_fidelity',
                 'run_id',
                 'use_prac_th',
                 'save_weights',
                 'stop_at_n',
                 'save_every_x']

    def __init__(self,
                 n_arms,
                 m_fidelity,
                 delta,
                 precisions,
                 costs,
                 variance_proxy,
                 kl_f,
                 hyper_params,
                 run_id,
                 use_prac_th,
                 save_weights,
                 stop_at_n,
                 save_every_x):
        self.n_arms = n_arms
        self.m_fidelity = m_fidelity
        self.delta = delta
        self.precisions = precisions
        self.costs = costs
        self.variance_proxy = variance_proxy
        self.kl_f = kl_f
        self.hyper_params = hyper_params
        self.run_id = run_id
        self.use_prac_th = use_prac_th
        self.save_weights = save_weights
        self.stop_at_n = stop_at_n
        self.save_every_x = save_every_x


class BaiAlgorithm(ABC):

    def __init__(self, cfg: BAIConfig):
        assert len(cfg.precisions) == cfg.m_fidelity

        self.n_arms = cfg.n_arms
        self.m_fidelity = cfg.m_fidelity
        self._active_set = set([i for i in range(self.n_arms)])
        self.delta = cfg.delta
        self._precisions = cfg.precisions
        self._costs = cfg.costs
        self._variance_proxy = cfg.variance_proxy

        self._mean_hat = [np.zeros(self.n_arms) for _ in range(cfg.m_fidelity)]
        self._arm_count = [np.zeros(self.n_arms) for _ in range(cfg.m_fidelity)]

        self.cfg = cfg

        self.w_T_seq = []
        self.grad_seq = []

    @abstractmethod
    def stopping_condition(self) -> bool:
        """
        :return: True if the algorithm needs to stop, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def pull_arm(self) -> (List[int], List[int]):
        """
        Returns which arm to pull at which fidelity level

        :return: (list of arm_idx to be pulled, fidelity)
        """
        raise NotImplementedError

    @abstractmethod
    def recommendation(self) -> int:
        """
        :return: which is the best arm
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, arm_idxes: List[int], fidelity: List[int], rewards: List):
        raise NotImplementedError

    def get_active_set(self) -> Set:
        return self._active_set.copy()

    def get_arm_count(self) -> List:
        """
        :return: number of pulls for each arm
        """
        return self._arm_count

    def get_cost(self) -> float:
        tot = 0
        for a in range(self.n_arms):
            for m in range(self.m_fidelity):
                tot += self._arm_count[m][a] * self._costs[m]

        return tot

    def get_cost_per_arm_and_fidelity(self) -> List[List[float]]:
        mat = np.zeros((self.n_arms, self.m_fidelity))
        for a in range(self.n_arms):
            for m in range(self.m_fidelity):
                mat[a, m] = self._arm_count[m][a] * self._costs[m]
        return list(mat)

    def compute_w_T(self):
        mat = np.zeros((self.m_fidelity, self.n_arms))
        for a in range(self.n_arms):
            mat[:, a] = self._arm_count[:, a] * np.array(self._costs) / self.get_cost()
        return mat
