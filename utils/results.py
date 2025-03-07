from typing import List
import numpy as np

from envs.multi_fidelity_env import MultiFidelityBanditModel


class ResultItem:
    __slots__ = ["best_arm",
                 "cost_complexity",
                 "cost_per_arm_and_fidelity",
                 "arm_count"]

    def __init__(self,
                 best_arm,
                 cost_complexity,
                 cost_per_arm_and_fidelity,
                 arm_count):
        self.best_arm = best_arm
        self.cost_complexity = cost_complexity
        self.cost_per_arm_and_fidelity = cost_per_arm_and_fidelity
        self.arm_count = arm_count


class ResultSummary:

    def __init__(self, res_list: List[ResultItem], bandit_model: MultiFidelityBanditModel):
        self.model = bandit_model
        self._res_list = res_list
        self._num_res = len(self._res_list)

    @property
    def num_run(self):
        return self._num_res

    def best_arm_stats(self):
        """
        :return: (percentage of right identifications)
        """
        true_best_arm = self.model.get_best_arm()
        count = 0
        for res in self._res_list:
            if res.best_arm == true_best_arm:
                count += 1
        return count / self._num_res * 100

    def cost_complexity_stats(self):
        """
        :return: (mean, std, all_vals) of cost complexity required to identify the best arm
        """
        all_vals = np.array([res.cost_complexity for res in self._res_list])
        return all_vals.mean(), all_vals.std(), all_vals

    def get_cost_array(self):
        res = [res.cost_complexity for res in self._res_list]
        return np.array(res)

    def get_avg_matrix_pulls(self):
        res = [res.cost_per_arm_and_fidelity / res.cost_complexity for res in self._res_list]
        return np.array(res)


class ResultItemWeights:
    __slots__ = ["best_arm",
                 "cost_complexity",
                 "cost_per_arm_and_fidelity",
                 "arm_count",
                 "grad_weight_seq",
                 "w_T_seq"]

    def __init__(self,
                 best_arm,
                 cost_complexity,
                 cost_per_arm_and_fidelity,
                 arm_count,
                 grad_weight_seq,
                 w_T_seq
                 ):
        self.best_arm = best_arm
        self.cost_complexity = cost_complexity
        self.cost_per_arm_and_fidelity = cost_per_arm_and_fidelity
        self.arm_count = arm_count
        self.w_T_seq = w_T_seq
        self.grad_weight_seq = grad_weight_seq


class ResultSummaryWeights:

    def __init__(self, res_list: List[ResultItemWeights], bandit_model: MultiFidelityBanditModel):
        self.model = bandit_model
        self._res_list = res_list
        self._num_res = len(self._res_list)

    @property
    def num_run(self):
        return self._num_res

    def best_arm_stats(self):
        """
        :return: (percentage of right identifications)
        """
        true_best_arm = self.model.get_best_arm()
        count = 0
        for res in self._res_list:
            if res.best_arm == true_best_arm:
                count += 1
        return count / self._num_res * 100

    def cost_complexity_stats(self):
        """
        :return: (mean, std, all_vals) of cost complexity required to identify the best arm
        """
        all_vals = np.array([res.cost_complexity for res in self._res_list])
        return all_vals.mean(), all_vals.std(), all_vals

    def get_cost_array(self):
        res = [res.cost_complexity for res in self._res_list]
        return np.array(res)

    def get_avg_matrix_pulls(self):
        res = [res.cost_per_arm_and_fidelity / res.cost_complexity for res in self._res_list]
        return np.array(res)

    def get_grad_seq(self):
        res = [res.grad_weight_seq for res in self._res_list]
        return np.array(res)

    def get_w_T_seq(self):
        res = [res.w_T_seq for res in self._res_list]
        return np.array(res)
