import numpy as np


def evaluate_objective(eta, w_1, w_a, mu_1, mu_a, k_1, k_a, xi, costs, kl_f):
    ta = np.dot(k_a * w_a, kl_f(mu_a, eta - xi) / costs)
    t1 = np.dot(k_1 * w_1, kl_f(mu_1, eta + xi) / costs)

    return t1 + ta


def evaluate_eta(eta, w_1, w_a, mu_1, mu_a, xi, costs):
    m_fid = xi.size

    rw_1 = w_1 / costs
    rw_a = w_a / costs

    k_a = np.array([1 if eta >= mu_a[m] + xi[m] else 0 for m in range(m_fid)])
    k_1 = np.array([1 if eta <= mu_1[m] - xi[m] else 0 for m in range(m_fid)])

    num = np.dot(rw_1 * k_1, mu_1 - xi) + np.dot(rw_a * k_a, mu_a + xi)
    den = np.dot(rw_1, k_1) + np.dot(rw_a, k_a)

    return num / den, k_1, k_a


def solve_mf_opt_problem_fast(w_1, w_a, mu_1, mu_a, xi, costs, kl_f):
    """
    it should return the value of the objective function, together with the values for
    arm 1 and arm a
    """
    points_1 = (mu_1 - xi).tolist()
    points_a = (mu_a + xi).tolist()
    points = np.array(points_1 + points_a)
    points = np.sort(points)

    fixed_point_list = []
    k_1_list = []
    k_a_list = []
    for i in range(len(points) - 1):
        eta = (points[i] + points[i + 1]) / 2
        res, k_1, k_a = evaluate_eta(eta, w_1, w_a, mu_1, mu_a, xi, costs)
        if points[i] <= res <= points[i + 1]:
            fixed_point_list.append(res)
            k_1_list.append(k_1)
            k_a_list.append(k_a)

    best_obj = None
    alt_1 = None
    alt_a = None
    for eta, k_1, k_a in zip(fixed_point_list, k_1_list, k_a_list):
        obj = evaluate_objective(eta, w_1, w_a, mu_1, mu_a, k_1, k_a, xi, costs, kl_f)
        if best_obj is None or obj < best_obj:
            best_obj = obj
            alt_1 = mu_1.copy()
            alt_a = mu_a.copy()
            for m in range(costs.size):
                if k_1[m] == 1:
                    alt_1[m] = eta + xi[m]
                if k_a[m] == 1:
                    alt_a[m] = eta - xi[m]

    return best_obj, alt_1, alt_a


def mf_bai_fast_old(weights: np.array,
                    mu_vec: np.array,
                    n_arms: int,
                    m_fid: int,
                    xi: np.array,
                    costs: np.array,
                    kl_f):
    """
    This routine can only be used when mu_vec is a multi-fidelity bandit!
    """
    weights = weights.reshape(m_fid, n_arms)
    mu_vec = mu_vec.reshape(m_fid, n_arms)

    best_arm_idx = np.argmax(mu_vec[m_fid - 1])

    # First try to solve using eta
    f_list = []
    best_val = None
    best_idx = None
    best_alt_1_vec = None
    best_alt_a_vec = None
    for a in range(n_arms):
        if a != best_arm_idx:
            f_a, alt_a_1_vec, alt_a_a_vec = solve_mf_opt_problem_fast(w_1=weights[:, best_arm_idx],
                                                                      w_a=weights[:, a],
                                                                      mu_1=mu_vec[:, best_arm_idx],
                                                                      mu_a=mu_vec[:, a],
                                                                      xi=xi,
                                                                      costs=costs,
                                                                      kl_f=kl_f
                                                                      )
            f_list.append(f_a)
            if best_idx is None or f_a < best_val:
                best_idx = a
                best_val = f_a
                best_alt_1_vec = alt_a_1_vec
                best_alt_a_vec = alt_a_a_vec

    return best_val, best_idx, best_alt_1_vec, best_alt_a_vec


def mf_bai_fast(weights: np.array,
                mu_vec: np.array,
                n_arms: int,
                m_fid: int,
                xi: np.array,
                costs: np.array,
                kl_f):
    """
    This routine can only be used when mu_vec is a multi-fidelity bandit!
    """
    weights = weights.reshape(m_fid, n_arms)
    mu_vec = mu_vec.reshape(m_fid, n_arms)

    # First try to solve using eta
    best_val = None
    best_idx = None
    best_alt_idx = None
    best_alt_i_star_vec = None
    best_alt_b_vec = None
    for i_star in range(n_arms):
        best_val_curr_arm = None
        best_alt_idx_curr_arm = None
        best_alt_i_star_vec_curr_arm = None
        best_alt_b_vec_curr_arm = None

        for b in range(n_arms):
            if b != i_star and mu_vec[-1, i_star] > mu_vec[-1, b]:
                f_i_b, alt_i_vec, alt_b_vec = solve_mf_opt_problem_fast(w_1=weights[:, i_star],
                                                                        w_a=weights[:, b],
                                                                        mu_1=mu_vec[:, i_star],
                                                                        mu_a=mu_vec[:, b],
                                                                        xi=xi,
                                                                        costs=costs,
                                                                        kl_f=kl_f
                                                                        )
                if best_val_curr_arm is None or f_i_b < best_val_curr_arm:
                    best_val_curr_arm = f_i_b
                    best_alt_idx_curr_arm = b
                    best_alt_i_star_vec_curr_arm = alt_i_vec
                    best_alt_b_vec_curr_arm = alt_b_vec

        if best_val is None or (best_val_curr_arm is not None and best_val_curr_arm > best_val):
            best_idx = i_star
            best_alt_idx = best_alt_idx_curr_arm
            best_val = best_val_curr_arm

            best_alt_i_star_vec = best_alt_i_star_vec_curr_arm
            best_alt_b_vec = best_alt_b_vec_curr_arm

    return best_val, best_idx, best_alt_idx, best_alt_i_star_vec, best_alt_b_vec
