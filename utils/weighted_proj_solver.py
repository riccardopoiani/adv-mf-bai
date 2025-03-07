import numpy as np
import cvxpy as cp


def evaluate_proj_objective(eta, n_vec, mean_vec, k_over, k_under, xi, kl_f):
    t_under = np.dot(n_vec * k_under, kl_f(mean_vec, eta + xi))
    t_over = np.dot(n_vec * k_over, kl_f(mean_vec, eta - xi))
    return t_over + t_under


def evaluate_eta_proj(eta, n_vec, mean_vec, xi, m_fid):
    k_over = np.array([1 if eta >= mean_vec[m] + xi[m] else 0 for m in range(m_fid)])
    k_under = np.array([1 if eta <= mean_vec[m] - xi[m] else 0 for m in range(m_fid)])

    num = np.dot(n_vec * k_over, mean_vec + xi) + np.dot(n_vec * k_under, mean_vec - xi)
    den = np.dot(n_vec, k_over) + np.dot(n_vec, k_under)
    return num / den, k_over, k_under


def solve_proj_fast(n_vec, mean_vec, xi_vec, kl_f):
    points_1 = (mean_vec - xi_vec).tolist()
    points_a = (mean_vec + xi_vec).tolist()
    points = np.array(points_1 + points_a)
    points = np.sort(points)

    fixed_point_list = []
    k_over_list = []
    k_under_list = []

    # Find fixed points
    for i in range(len(points) - 1):
        eta = (points[i] + points[i + 1]) / 2
        res, k_1, k_a = evaluate_eta_proj(eta, n_vec, mean_vec, xi_vec, xi_vec.size)
        if points[i] <= res <= points[i + 1]:
            fixed_point_list.append(res)
            k_over_list.append(k_1)
            k_under_list.append(k_a)

    # Evaluate objective
    best_obj = None
    alt = None
    for eta, k_over, k_under in zip(fixed_point_list, k_over_list, k_under_list):
        obj = evaluate_proj_objective(eta, n_vec, mean_vec, k_over, k_under, xi_vec, kl_f)
        if best_obj is None or obj < best_obj:
            best_obj = obj
            alt = mean_vec.copy()
            for m in range(xi_vec.size):
                if k_over[m] == 1:
                    alt[m] = eta - xi_vec[m]
                if k_under[m] == 1:
                    alt[m] = eta + xi_vec[m]

    return alt, best_obj


def satisfy_mf_constraints(mean_vec, xi_vec):
    for m in range(mean_vec.size):
        if not abs(mean_vec[m] - mean_vec[-1]) <= xi_vec[m]:
            return False
    return True


def weighted_proj_opt(na_vec, mu_vec, n_arms, m_fid, xi_vec, kl_f):
    proj_vec = np.zeros((m_fid, n_arms))
    for a in range(n_arms):
        # check if model satisfy MF constraints
        if satisfy_mf_constraints(mu_vec[:, a], xi_vec):
            proj_vec[:, a] = mu_vec[:, a]
        else:
            proj_vec[:, a], _ = solve_proj_fast(na_vec[:, a], mu_vec[:, a], xi_vec, kl_f)

    return proj_vec


def weighted_proj_optim(na_vec, mu_vec, n_arms, m_fid, xi_vec, kl_f):
    opt_proj_var = [cp.Variable(m_fid) for _ in range(n_arms)]
    opt_constraints = []

    # Fidelity constraints among optimization variables
    for a in range(n_arms):
        for m_idx in range(m_fid - 1):
            opt_constraints.append(
                opt_proj_var[a][m_idx] - opt_proj_var[a][-1] <= xi_vec[m_idx])
            opt_constraints.append(
                opt_proj_var[a][-1] - opt_proj_var[a][m_idx] <= xi_vec[m_idx])

    # Objective function
    terms = []
    for a in range(n_arms):
        terms.append(
            cp.sum(cp.multiply(na_vec[:, a], kl_f(opt_proj_var[a], mu_vec[:, a]))))
    obj = cp.sum(terms)
    problem = cp.Problem(cp.Minimize(obj), opt_constraints)
    try:
        problem.solve()
    except:
        problem.solve(cp.CVXOPT)

    model = np.array([opt_proj_var[a].value for a in range(n_arms)]).transpose()
    return model
