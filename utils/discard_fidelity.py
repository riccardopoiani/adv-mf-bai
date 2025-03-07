import cvxpy as cp
import numpy as np


def solve_for_fixed_idx(fid_idx,
                        w_arm_idx,
                        w_a,
                        mu_arm_idx,
                        mu_a,
                        kl_f,
                        xi):
    theta_arm_idx = cp.Variable(mu_arm_idx.size)
    theta_a = cp.Variable(mu_a.size)

    constraints = []

    # Fidelity constraints
    for m_idx in range(mu_arm_idx.size - 1):
        constraints.append(theta_a[m_idx] - theta_a[-1] <= xi[m_idx])
        constraints.append(theta_a[-1] - theta_a[m_idx] <= xi[m_idx])

        constraints.append(theta_arm_idx[m_idx] - theta_arm_idx[-1] <= xi[m_idx])
        constraints.append(theta_arm_idx[-1] - theta_arm_idx[m_idx] <= xi[m_idx])

    # Optimal arms
    constraints.append(theta_a[-1] >= theta_arm_idx[-1])
    constraints.append(theta_a[-1] >= theta_arm_idx[fid_idx] + xi[m])

    obj = cp.sum(cp.multiply(w_a, kl_f(mu_a, theta_a))) + cp.sum(
        cp.multiply(w_arm_idx, kl_f(mu_arm_idx, theta_arm_idx)))

    # Solving the problem
    prob = cp.Problem(cp.Minimize(obj), constraints)
    sol = prob.solve()

    return sol


def glrt_discard(n_arms: int, arm_idx: int, fid_idx: int, weights: np.array, mu_vec: np.array, kl_f, xi):
    best_val = None
    for a in range(n_arms):
        if a != arm_idx:
            curr_val = solve_for_fixed_idx(fid_idx,
                                           weights[:, arm_idx],
                                           weights[:, a],
                                           mu_vec[:, arm_idx],
                                           mu_vec[:, a],
                                           kl_f,
                                           xi)

            if best_val is None or curr_val < best_val:
                best_val = curr_val

    return best_val
