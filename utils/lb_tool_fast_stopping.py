import numpy as np

from utils.lb_tools_fast import solve_mf_opt_problem_fast
from utils.weighted_proj_solver import satisfy_mf_constraints, solve_proj_fast


def evaluate_objective_stop(eta_1, eta_a, w_1, w_a, mu_1, mu_a, xi, kl_f):
    k_a_over = np.array([1 if eta_a >= mu_a[m] + xi[m] else 0 for m in range(xi.size)])
    k_a_under = np.array([1 if eta_a <= mu_a[m] - xi[m] else 0 for m in range(xi.size)])

    k_1_over = np.array([1 if eta_1 >= mu_1[m] + xi[m] else 0 for m in range(xi.size)])
    k_1_under = np.array([1 if eta_1 <= mu_1[m] - xi[m] else 0 for m in range(xi.size)])

    if k_1_under[-1] == k_1_over[-1]:
        k_1_under[-1] = 0
    if k_a_under[-1] == k_a_over[-1]:
        k_a_over[-1] = 0

    t1 = np.dot(k_a_over * w_a, kl_f(mu_a, eta_a - xi))
    t2 = np.dot(k_a_under * w_a, kl_f(mu_a, eta_a + xi))

    t3 = np.dot(k_1_under * w_1, kl_f(mu_1, eta_1 + xi))
    t4 = np.dot(k_1_over * w_1, kl_f(mu_1, eta_1 - xi))

    return t1 + t2 + t3 + t4


def evaluate_eta_stop(eta, w_1, w_a, mu_1, mu_a, xi):
    m_fid = xi.size

    k_a_over = np.array([1 if eta >= mu_a[m] + xi[m] else 0 for m in range(m_fid)])
    k_a_under = np.array([1 if eta <= mu_a[m] - xi[m] else 0 for m in range(m_fid)])

    k_1_over = np.array([1 if eta >= mu_1[m] + xi[m] else 0 for m in range(m_fid)])
    k_1_under = np.array([1 if eta <= mu_1[m] - xi[m] else 0 for m in range(m_fid)])

    if k_1_under[-1] == k_1_over[-1]:
        k_1_under[-1] = 0
    if k_a_under[-1] == k_a_over[-1]:
        k_a_over[-1] = 0
    num_1 = np.dot(w_1 * k_1_under, mu_1 - xi) + np.dot(w_1 * k_1_over, mu_1 + xi)
    num_a = np.dot(w_a * k_a_under, mu_a - xi) + np.dot(w_a * k_a_over, mu_a + xi)
    den = np.dot(w_1, k_1_over) + np.dot(w_1, k_1_under) + np.dot(w_a, k_a_under) + np.dot(w_a, k_a_over)

    return (num_1 + num_a) / den


def solve_mf_opt_problem_fast_not_model(w_1, w_a, mu_1, mu_a, xi, kl_f, eps=1e-5):
    """
    it should return the value of the objective function, together with the values for
    arm 1 and arm a
    """
    points_1m = (mu_1 - xi).tolist()
    points_1p = (mu_1 + xi).tolist()
    points_ap = (mu_a + xi).tolist()
    points_am = (mu_a - xi).tolist()

    points = np.array(points_1m + points_am + points_1p + points_ap)
    points = np.sort(points)

    fixed_point_list = []

    # Test initial and final point
    eta = points[0] - 0.5
    res = evaluate_eta_stop(eta, w_1, w_a, mu_1, mu_a, xi)
    if res <= points[0]:
        fixed_point_list.append(eta)
    eta = points[-1] + 0.5
    res = evaluate_eta_stop(eta, w_1, w_a, mu_1, mu_a, xi)
    if res >= points[-1]:
        fixed_point_list.append(eta)

    for i in range(len(points) - 1):
        eta = (points[i] + points[i + 1]) / 2
        if points[i] == points[i + 1]:
            res = evaluate_eta_stop(points[i], w_1, w_a, mu_1, mu_a, xi)
            if res == points[i]:
                fixed_point_list.append(res)
        else:
            res = evaluate_eta_stop(eta, w_1, w_a, mu_1, mu_a, xi)
            if points[i] <= res <= points[i + 1]:
                fixed_point_list.append(res)

    best_obj = None
    alt_1 = None
    alt_a = None
    for eta in fixed_point_list:
        obj = evaluate_objective_stop(eta, eta, w_1, w_a, mu_1, mu_a, xi, kl_f)
        if best_obj is None or obj < best_obj:
            best_obj = obj

            alt_1 = mu_1.copy()
            alt_a = mu_a.copy()
            for m in range(xi.size):
                if eta >= mu_1[m] + xi[m]:
                    alt_1[m] = eta - xi[m]
                if eta >= mu_a[m] + xi[m]:
                    alt_a[m] = eta - xi[m]
                if eta <= mu_1[m] - xi[m]:
                    alt_1[m] = eta + xi[m]
                if eta <= mu_a[m] - xi[m]:
                    alt_a[m] = eta + xi[m]

    return best_obj, alt_1, alt_a


def solve_stop_problem_fast(w_1, w_a, mu_1, mu_a, xi, kl_f):
    # If arms are MF we are ok
    if satisfy_mf_constraints(mu_1, xi) and satisfy_mf_constraints(mu_a, xi):
        return solve_mf_opt_problem_fast(w_1, w_a, mu_1, mu_a, xi, np.ones(xi.size), kl_f)

    # Then, we compute the value at the non-equality by minimizing the individual projection of mu_1, mu_a
    val_1 = 0
    val_a = 0
    alt_1 = mu_1.copy()
    alt_a = mu_a.copy()
    if not satisfy_mf_constraints(mu_1, xi):
        alt_1, val_1 = solve_proj_fast(w_1, mu_1.copy(), xi, kl_f)
    if not satisfy_mf_constraints(mu_a, xi):
        alt_a, val_a = solve_proj_fast(w_a, mu_a.copy(), xi, kl_f)
    if alt_a is not None and alt_1 is not None and alt_a[-1] > alt_1[-1]:
        return val_1 + val_a, alt_1, alt_a

    # Otherwise, we need to devise a new tool to solve the problem
    val_equality, alt_1_eq, alt_a_eq = solve_mf_opt_problem_fast_not_model(w_1, w_a, mu_1.copy(), mu_a.copy(), xi, kl_f)

    if val_equality is None and alt_a is not None and alt_1 is not None and alt_a[-1] < alt_1[-1]:
        solve_mf_opt_problem_fast_not_model(w_1, w_a, mu_1, mu_a, xi, kl_f)
        return solve_mf_opt_problem_fast(w_1, w_a, mu_1, mu_a, xi, np.ones(xi.size), kl_f)

    if alt_a is not None and alt_1 is not None and alt_a[-1] > alt_1[-1]:
        if val_a + val_1 < val_equality:
            return val_1 + val_a, alt_1, alt_a

    return val_equality, alt_1_eq, alt_a_eq


def solve_stopping(weights, mu_vec, xi, n_arms, m_fid, kl_f, debug=False):
    weights = weights.reshape((m_fid, n_arms))
    mu_vec = mu_vec.reshape((m_fid, n_arms))

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
            if b != i_star:
                f_i_b, alt_i_vec, alt_b_vec = solve_stop_problem_fast(w_1=weights[:, i_star],
                                                                      w_a=weights[:, b],
                                                                      mu_1=mu_vec[:, i_star],
                                                                      mu_a=mu_vec[:, b],
                                                                      xi=xi,
                                                                      kl_f=kl_f
                                                                      )
                if debug:
                    print(f"f({i_star}, {b}) = {f_i_b}")
                if best_val_curr_arm is None or (f_i_b is not None and f_i_b < best_val_curr_arm):
                    best_val_curr_arm = f_i_b
                    best_alt_idx_curr_arm = b
                    best_alt_i_star_vec_curr_arm = alt_i_vec
                    best_alt_b_vec_curr_arm = alt_b_vec

        if best_val is None or (best_val_curr_arm is not None and best_val_curr_arm > best_val):
            if debug:
                print(f"Best is {i_star} with {best_val_curr_arm}")
            best_idx = i_star
            best_alt_idx = best_alt_idx_curr_arm
            best_val = best_val_curr_arm

            best_alt_i_star_vec = best_alt_i_star_vec_curr_arm
            best_alt_b_vec = best_alt_b_vec_curr_arm

    return best_val, best_idx, best_alt_idx, best_alt_i_star_vec, best_alt_b_vec
