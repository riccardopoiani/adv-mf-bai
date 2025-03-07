from abc import abstractmethod
from typing import List

import numpy as np

from algorithms.bai import BaiAlgorithm
from utils.lb_tool_fast_stopping import solve_stopping
from utils.lb_tools import bai
from utils.math import update_mean_online
from utils.regret_minimizer import GradientAscent


def from_w_to_pi(w, costs, n_arms):
    norm = 0
    for a in range(n_arms):
        norm += np.sum(w[:, a] / costs)

    pi = np.zeros(w.shape)
    for a in range(n_arms):
        pi[:, a] = (w[:, a] / costs) * (1 / norm)
    return pi


class GradientLearner(BaiAlgorithm):

    def __init__(self, cfg):
        super(GradientLearner, self).__init__(cfg)
        self.t = 0
        self.lr = self.cfg.hyper_params['lr']
        self.kl_f = self.cfg.kl_f
        self.verbosity = self.cfg.hyper_params['verbosity']

    def stopping_condition(self) -> bool:
        if self.cfg.stop_at_n > 0:
            if self.t >= self.cfg.stop_at_n:
                return True
            return False

        beta = self.compute_beta()
        alt_value = self.compute_glrt()
        if alt_value >= beta:
            return True
        return False

    @abstractmethod
    def compute_beta(self):
        raise NotImplementedError

    @abstractmethod
    def compute_glrt(self):
        raise NotImplementedError

    @abstractmethod
    def compute_gradient(self) -> np.array:
        raise NotImplementedError

    @abstractmethod
    def pull_arm(self) -> (List[int], List[int]):
        raise NotImplementedError

    @abstractmethod
    def recommendation(self) -> int:
        raise NotImplementedError

    def update(self, arm_idxes: List[int], fidelity: List[int], rewards: List):
        # Update statistics
        for a, m, r in zip(arm_idxes, fidelity, rewards):
            self._mean_hat[m][a] = update_mean_online(
                self._arm_count[m][a],
                self._mean_hat[m][a],
                r
            )
            self._arm_count[m][a] += 1

    def get_gamma(self):
        return 1 / (4 * np.sqrt(self.t))


class BAIGradientLearner(GradientLearner):
    NAME = "GRAD"

    def __init__(self, cfg):
        super(BAIGradientLearner, self).__init__(cfg)
        self.cum_w_forced = np.zeros(self.n_arms)
        self.gradient_ascent = GradientAscent(self.n_arms, lr=self.lr)
        self.uniform_vector = np.ones(self.n_arms) / self.n_arms
        self.curr_w_tilde = np.ones(self.n_arms) / self.n_arms

    def pull_arm(self) -> (List[int], List[int]):
        self.t += 1

        # Initialization
        if self._arm_count[self.m_fidelity - 1].min() == 0:
            for a in range(self.n_arms):
                if self._arm_count[self.m_fidelity - 1][a] == 0:
                    return [a], [self.m_fidelity - 1]

        # Arm pull
        gradient = self.compute_gradient()
        self.gradient_ascent.feed(gradient)
        self.curr_w_tilde = self.gradient_ascent.get_action()

        # Add forced exploration
        w_prime = (1 - self.get_gamma()) * self.curr_w_tilde + self.get_gamma() * self.uniform_vector
        self.cum_w_forced += w_prime

        # Tracking
        a = np.argmax(self.cum_w_forced - self._arm_count[self.m_fidelity - 1])
        return [a], [self.m_fidelity - 1]

    def compute_gradient(self) -> np.array:
        grad = np.zeros(self.n_arms)

        # Solve lower bound for w_tilde and mu_hat
        _, min_idx, alt = bai(self.curr_w_tilde, self._mean_hat[self.m_fidelity - 1], self.kl_f)

        # Compute subgradient
        best_arm_idx = np.argmax(self._mean_hat[self.m_fidelity - 1])
        grad[best_arm_idx] = self.kl_f(self._mean_hat[self.m_fidelity - 1][best_arm_idx], alt)
        grad[min_idx] = self.kl_f(self._mean_hat[self.m_fidelity - 1][min_idx], alt)

        return grad

    def compute_glrt(self):
        if self._arm_count[self.m_fidelity - 1].min() == 0:
            return 0

        return bai(self._arm_count[self.m_fidelity - 1], self._mean_hat[self.m_fidelity - 1], self.kl_f)[0]

    def recommendation(self) -> int:
        return np.argmax(self._mean_hat[self.m_fidelity - 1])

    def compute_beta(self):
        # if self.cfg.use_prac_th:
        return np.log(self.n_arms / self.delta) + np.log(np.log(self.t) + 1)

        # t1 = np.log(1 / self.cfg.delta)
        # t2 = self.n_arms * np.log(4 * np.log(1 / self.delta) + 1)
        # t3 = 12 * np.log(np.log(self.t) + 3)
        # t4 = self.n_arms
        # return t1 + t2 + t3 + t4


class MFBAIGradientLearner(GradientLearner):
    NAME = "MF-GRAD"

    def __init__(self, cfg):
        super(MFBAIGradientLearner, self).__init__(cfg)

        self._mean_hat = np.array(self._mean_hat)
        self._arm_count = np.array(self._arm_count)

        self.cum_pi_forced = np.zeros(self.m_fidelity * self.n_arms)
        self.gradient_ascent = GradientAscent(self.m_fidelity * self.n_arms, lr=self.lr)
        self.uniform_vector = np.ones(self.m_fidelity * self.n_arms) / (self.n_arms * self.m_fidelity)
        self.curr_w_tilde = np.ones(self.m_fidelity * self.n_arms) / (self.n_arms * self.m_fidelity)
        self.curr_pi_tilde = from_w_to_pi(self.curr_w_tilde.reshape(self.m_fidelity,
                                                                    self.n_arms
                                                                    ),
                                          np.array(self._costs),
                                          self.n_arms)

    def pull_arm(self) -> (List[int], List[int]):
        self.t += 1

        if self.verbosity is not None and self.t % self.verbosity == 0:
            print(f"Run {self.cfg.run_id} - Iter {self.t}")

        # Initialization
        if sum([int(self._arm_count[m].min() == 0) for m in range(self.m_fidelity)]) > 0:
            for a in range(self.n_arms):
                for m in range(self.m_fidelity):
                    if self._arm_count[m][a] == 0:
                        return [a], [m]

        # Arm pull
        gradient = self.compute_gradient()
        self.gradient_ascent.feed(gradient)
        self.curr_w_tilde = self.gradient_ascent.get_action()

        if self.cfg.save_weights and self.t % self.cfg.save_every_x == 0:
            self.grad_seq.append(self.curr_w_tilde.reshape((self.m_fidelity, self.n_arms)).copy())
            self.w_T_seq.append(self.compute_w_T())

        # Add forced exploration
        self.curr_pi_tilde = from_w_to_pi(self.curr_w_tilde.reshape((self.m_fidelity, self.n_arms)),
                                          np.array(self._costs),
                                          self.n_arms
                                          )
        self.curr_pi_tilde = self.curr_pi_tilde.reshape(self.m_fidelity * self.n_arms)

        # Add forced exploration
        pi_prime = (1 - self.get_gamma()) * self.curr_pi_tilde + self.get_gamma() * self.uniform_vector
        self.cum_pi_forced += pi_prime

        # Tracking
        mat_cum_w_force = self.cum_pi_forced.reshape((self.m_fidelity, self.n_arms))

        idx = np.unravel_index(np.argmax(mat_cum_w_force - self._arm_count), mat_cum_w_force.shape)

        return [idx[1]], [idx[0]]

    def compute_beta(self):
        # if self.cfg.use_prac_th:
        return np.log(self.n_arms / self.delta) + self.m_fidelity * np.log(np.log(self.t) + 1)

        # t1 = np.log(self.n_arms / self.delta)
        # t2 = 2 * self.m_fidelity * np.log(np.log(self.n_arms / self.delta) + 1)
        # t3 = 12 * self.m_fidelity * np.log(np.log(self.t) + 3)
        # t4 = 2 * self.m_fidelity
        # return t1 + t2 + t3 + t4

    def compute_gradient(self) -> np.array:
        grad = np.zeros((self.m_fidelity, self.n_arms))

        # Solve lower bound for w_tilde and mu_hat
        weights = self.curr_w_tilde.reshape((self.m_fidelity, self.n_arms)).copy()
        for a in range(self.n_arms):
            weights[:, a] = weights[:, a] / np.array(self._costs)
        weights = weights.reshape(self.m_fidelity * self.n_arms)
        _, best_arm_idx, min_idx, alt_best_vec, alt_min_idx_vec = solve_stopping(weights,
                                                                                 self._mean_hat.copy(),
                                                                                 np.array(self._precisions),
                                                                                 self.n_arms,
                                                                                 self.m_fidelity,
                                                                                 self.cfg.kl_f)

        # Compute subgradient
        grad[:, best_arm_idx] = self.kl_f(self._mean_hat[:, best_arm_idx], alt_best_vec) / self._costs
        grad[:, min_idx] = self.kl_f(self._mean_hat[:, min_idx], alt_min_idx_vec) / self._costs

        for m in range(self.m_fidelity - 1):
            if alt_min_idx_vec[-1] <= self._mean_hat[m, min_idx] + self._precisions[m]:
                grad[m, min_idx] = 0
            if alt_best_vec[-1] >= self._mean_hat[m, best_arm_idx] - self._precisions[m]:
                grad[m, best_arm_idx] = 0

        # Add correction
        correction = 0
        self.curr_pi_tilde = self.curr_pi_tilde.reshape((self.m_fidelity, self.n_arms))
        for a in range(self.n_arms):
            correction += np.sum((self.curr_pi_tilde[:, a] * np.array(self._costs)))
        self.curr_pi_tilde = self.curr_pi_tilde.reshape(self.n_arms * self.m_fidelity)
        grad *= correction

        # Flat gradient
        grad = grad.flatten()

        return grad

    def compute_glrt(self):
        if sum([int(self._arm_count[m].min() == 0) for m in range(self.m_fidelity)]) > 0:
            return 0

        fast = solve_stopping(self._arm_count,
                              self._mean_hat,
                              np.array(self._precisions),
                              self.n_arms,
                              self.m_fidelity,
                              self.cfg.kl_f)[0]
        return fast

    def recommendation(self) -> int:
        return solve_stopping(self._arm_count,
                              self._mean_hat,
                              np.array(self._precisions),
                              self.n_arms,
                              self.m_fidelity,
                              self.cfg.kl_f,
                              debug=False)[1]
