import numpy as np
import cvxpy as cp


def bai(weights: np.array, mu_vec: np.array, kl_f):
    # Standard tools to solve the alt problem for an unstructured BAI problem
    best_arm_idx = np.argmax(mu_vec)
    f_list = []
    best_val = None
    best_idx = None
    best_alt = None
    for a in range(mu_vec.size):
        if a != best_arm_idx:
            mu_best = mu_vec[best_arm_idx]
            mu_a = mu_vec[a]

            # Add computations
            p1 = weights[best_arm_idx] / (weights[best_arm_idx] + weights[a]) * mu_best
            pa = weights[a] / (weights[best_arm_idx] + weights[a]) * mu_a
            alt_a = p1 + pa

            f_a = weights[best_arm_idx] * kl_f(mu_best, alt_a) + weights[a] * kl_f(mu_a, alt_a)
            f_list.append(f_a)
            if best_idx is None or f_a < best_val:
                best_idx = a
                best_val = f_a
                best_alt = alt_a

    return best_val, best_idx, best_alt


def solve_mf_opt_problem(w_1, w_a, mu_1, mu_a, xi, costs, kl_f):
    M = len(xi)

    # Rescale weights
    rw_1 = w_1 / costs
    rw_a = w_a / costs

    # Init constraints list
    constraints = []

    # Build F_a and optimize
    theta_1 = cp.Variable(M)
    theta_a = cp.Variable(M)

    # Swap optimal arm
    constraints.append(theta_a[-1] >= theta_1[-1])

    # Fidelity constraints among optimization variables
    for m_idx in range(M - 1):
        constraints.append(theta_a[m_idx] - theta_a[-1] <= xi[m_idx])
        constraints.append(theta_a[-1] - theta_a[m_idx] <= xi[m_idx])

        constraints.append(theta_1[m_idx] - theta_1[-1] <= xi[m_idx])
        constraints.append(theta_1[-1] - theta_1[m_idx] <= xi[m_idx])

    # Objective function
    obj = cp.sum(cp.multiply(rw_1, kl_f(theta_1, mu_1))) + cp.sum(
        cp.multiply(rw_a, kl_f(theta_a, mu_a)))

    prob = cp.Problem(cp.Minimize(obj), constraints)

    try:
        sol = prob.solve()
    except:
        try:
            sol = prob.solve(solver=cp.CVXOPT)
        except:
            sol = prob.solve(solver=cp.CVXOPT, verbose=True)
            print("Error")
            return None

    return sol, theta_1.value, theta_a.value


def solve_mf_opt_problem_complete(opt_idx, weights, mu_vec, xi, costs, n_arms, m_fid, kl_f):
    M = len(xi)

    # Weights rescaling
    rw = np.zeros(weights.shape)
    for m in range(m_fid):
        for a in range(n_arms):
            rw[m, a] = weights[m, a] / costs[m]

    opt_vars = [cp.Variable(M) for _ in range(n_arms)]

    # Constraints
    constraints = []

    # Opt idx is the optimal arm
    for a in range(n_arms):
        if a != opt_idx:
            constraints.append(opt_vars[opt_idx][M - 1] >= opt_vars[a][M - 1])

        # Fidelity constraints among optimization variables
    for a in range(n_arms):
        for m_idx in range(M - 1):
            constraints.append(opt_vars[a][m_idx] - opt_vars[a][-1] <= xi[m_idx])
            constraints.append(opt_vars[a][-1] - opt_vars[a][m_idx] <= xi[m_idx])

    # Objective function
    terms = []
    for a in range(n_arms):
        terms.append(cp.sum(cp.multiply(rw[:, a], kl_f(opt_vars[a], mu_vec[:, a]))))

    obj = cp.sum(terms)

    # Solving the problem
    prob = cp.Problem(cp.Minimize(obj), constraints)

    try:
        sol = prob.solve()
    except:
        try:
            sol = prob.solve(solver=cp.CVXOPT)
        except:
            sol = prob.solve(solver=cp.CVXOPT, verbose=True)
            print("Error")
            return None

    return sol


def mf_bai(weights: np.array,
           mu_vec: np.array,
           n_arms: int,
           m_fid: int,
           xi: np.array,
           costs: np.array,
           kl_f):
    weights = weights.reshape(m_fid, n_arms)
    mu_vec = mu_vec.reshape(m_fid, n_arms)

    best_arm_idx = np.argmax(mu_vec[m_fid-1])

    # First try to solve using eta
    f_list = []
    best_val = None
    best_idx = None
    best_alt_1_vec = None
    best_alt_a_vec = None
    for a in range(n_arms):
        if a != best_arm_idx:
            f_a, alt_a_1_vec, alt_a_a_vec = solve_mf_opt_problem(w_1=weights[:, best_arm_idx],
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
