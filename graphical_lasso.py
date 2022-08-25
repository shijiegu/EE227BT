r'''
This module contains some implementations for the Graphical Lasso and for the
Joint Graphical Lasso.
'''
from inverse_covariance import QuicGraphicalLasso
import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import cd_fast
from sklearn.utils.validation import check_random_state
from typing import List, Tuple


def graphical_lasso(
        S: np.array,
        l: float,
        threshold: float = 1e-4,
        penalize_diagonal: bool = True)\
        -> Tuple[np.array, np.array]:
    r'''
    Solves the L1-regularized maximum likelihood objective:
    max_{Theta} logdet(Theta) - tr(Theta @ S) - lambda * |Theta|_1
    Implements the Graphical Lasso algorithm from the paper:
        `Sparse inverse covariance estimation with the graphical LASSO`
        Friedman et al. (2008)
        https://www.ncbi.nlm.nih.gov/pubmed/18079126
    Uses cd_fast.enet_coordinate_descent_gram to solve the inner L1 lasso
    regression type problem.
    :param S: Empirical covariance matrix
    :param l: L1 regularization strength
    :param threshold: Convergence is declared when the mean absolute change in
        the inferred covariance matrix W is less than threshold * |S|_{1, off}
        (the L1 norm of the off-diagonal elements of S)
    :param penalize_diagonal: If True, diagonal elements of the precision
        matrix are penalized. More recently, it has become popular to not
        penalize the diagonal.
    :return W: The inferred covariance matrix
    :return Theta: The inferred precision matrix
    '''
    p = S.shape[0]  # a.k.a. n_features
    W = S + l * np.identity(p) if penalize_diagonal else np.copy(S)
    beta = np.zeros(shape=(p, p - 1), dtype=float)
    while True:
        W_old = np.copy(W)
        for j in range(p):
            minus_j = [idx for idx in range(p) if idx != j]
            V = W[np.ix_(minus_j, minus_j)]
            s_j = S[minus_j, j]
            beta[j, :], _, _, _ = cd_fast.enet_coordinate_descent_gram(
                w=beta[j, :], alpha=l, beta=0.0, Q=V, q=s_j, y=s_j,
                max_iter=1000, tol=1e-3, rng=check_random_state(None))
            w_j = V @ beta[j]
            W[minus_j, j] = w_j
            W[j, minus_j] = w_j
        if np.mean(np.abs(W - W_old)) <= \
                threshold * np.mean(np.abs(S - np.diag(np.diag(S)))):
            break
    theta = np.zeros_like(W)
    for j in range(p):
        minus_j = [idx for idx in range(p) if idx != j]
        w_j = W[j, minus_j]
        theta[j, j] = 1.0 / (W[j, j] - np.dot(w_j, beta[j]))
        theta[minus_j, j] = -beta[j] * theta[j, j]
    return W, theta


def graphical_lasso_fast(
        S: np.array,
        l: float,
        threshold: float = 1e-4,
        penalize_diagonal: bool = True,
        verbose: float = False)\
        -> Tuple[np.array, np.array]:
    r'''
    Wrapper around the graphical_lasso. First finds the block diagonal
    structure of the solution and then solves the Graphical Lasso for each
    component, greatly speeding up the algorithm while producing the exact
    same results.
    Implements the paper:
        `New Insights and Faster Computations for the Graphical Lasso`
        Witten et al. (2011)
        https://projecteuclid.org/euclid.ejs/1352470831
    :param S: Empirical covariance matrix
    :param l: L1 regularization strength
    :param threshold: Convergence is declared when the mean absolute change in
        the inferred covariance matrix W is less than threshold * |S|_{1, off}
        (the L1 norm of the off-diagonal elements of S)
    :param penalize_diagonal: If True, diagonal elements of the precision
        matrix are penalized. More recently, it has become popular to not
        penalize the diagonal.
    :param verbose: If True, will print the number of components found and the
        size of the largest component
    :return W: The inferred covariance matrix
    :return Theta: The inferred precision matrix
    '''
    W = np.zeros_like(S, dtype=float)
    theta = np.zeros_like(S, dtype=float)
    adj_mat = np.abs(S) > l
    n_components, labels = connected_components(
        csgraph=adj_mat, directed=False, return_labels=True)
    component_indices = [[] for _ in range(n_components)]
    for vertex_idx, component_index in enumerate(labels):
        component_indices[component_index].append(vertex_idx)
    if verbose:
        largest_component_size = max([len(idxs) for idxs in component_indices])
        print(f"graphical_lasso_fast:\n\tn_components found = {n_components}\n"
              f"\tlargest component found = {largest_component_size} nodes")
    for idxs in component_indices:
        W[np.ix_(idxs, idxs)], theta[np.ix_(idxs, idxs)] = graphical_lasso(
            S[np.ix_(idxs, idxs)], l, threshold, penalize_diagonal)
    return W, theta


def quic_graphical_lasso_fast(
        X: np.array,
        l: np.array,
        S: np.array,
        verbose: bool = False)\
        -> Tuple[np.array, np.array]:
    r'''
    Wrapper around the QUIC graphical_lasso. First finds the block diagonal
    structure of the solution and then solves the QUIC Graphical Lasso for each
    component, greatly speeding up the algorithm while producing the exact
    same results. Since QUIC Graphical Lasso takes as parameter the data
    matrix X, it is also required.
    Implements the paper:
        `New Insights and Faster Computations for the Graphical Lasso`
        Witten et al. (2011)
        https://projecteuclid.org/euclid.ejs/1352470831
    :param X: The data matrix, of size n_samples x n_features
    :param l: L1 regularization strength matrix, with penalty for each
        entry in the precision matrix.
    :param S: Empirical covariance matrix
    :param verbose: If True, will print the number of components found and the
        size of the largest component
    :return W: The inferred covariance matrix
    :return Theta: The inferred precision matrix
    '''
    p = X.shape[1]  # a.k.a. n_features
    if l.shape != (p, p):
        raise ValueError(f"l must be a n_features X n_features matrix")
    W = np.zeros_like(S, dtype=float)
    theta = np.zeros_like(S, dtype=float)
    adj_mat = np.abs(S) > l
    n_components, labels = connected_components(
        csgraph=adj_mat, directed=False, return_labels=True)
    component_indices = [[] for _ in range(n_components)]
    for vertex_idx, component_index in enumerate(labels):
        component_indices[component_index].append(vertex_idx)
    if verbose:
        largest_component_size = max([len(idxs) for idxs in component_indices])
        print(f"graphical_lasso_fast:\n\tn_components found = {n_components}\n"
              f"\tlargest component found = {largest_component_size} nodes")
    for idxs in component_indices:
        if len(idxs) == 1:
            # Cannot use QuicGlasso because it explodes, so just use closed
            # form solution
            W[np.ix_(idxs, idxs)] = \
                S[np.ix_(idxs, idxs)] + l[np.ix_(idxs, idxs)]
            theta[np.ix_(idxs, idxs)] = 1.0 / W[np.ix_(idxs, idxs)]
        else:
            quic_results = QuicGraphicalLasso(lam=l[np.ix_(idxs, idxs)])\
                .fit(X[:, idxs])
            W[np.ix_(idxs, idxs)], theta[np.ix_(idxs, idxs)] = \
                quic_results.covariance_, quic_results.precision_
    return W, theta


def joint_graphical_lasso_guo(
        Xs: List[np.array],
        l: float,
        fast: bool = False,
        threshold: float = 1e-5,
        smart_init: bool = True,
        verbose: bool = False) -> Tuple[List[np.array], List[np.array]]:
    r'''
    Solves the L1-regularized maximum likelihood objective:
    max_{Theta} sum_k [ logdet(Theta^(k)) - tr(Theta^(k) @ S^(k))
                        - lambda * |Theta^(k)|_1 ]
                - sum_ij (sum_k |Theta^(k)_ij|)^0.5
    where S^(k) is the empirical covariance matrix for the k-th class.
    This is the Joint Graphical Lasso by Guo et al.
        `Joint estimation of multiple graphical models`
        Guo et al. (2011)
        https://www.researchgate.net/publication/232226235_Joint_estimation_of_multiple_graphical_models
    :param Xs: Data matrices
    :param l: Regularization strength (float)
    :param fast: If to use the fast Graphical Lasso solver as the underlying
        one. The `fast` solver first finds connected components. Use this for
        high dimensional data.
    :param threshold: Convergence is declared when the relative change in
        Thetas is less that threshold
    :param verbose: If True, will show training progress statistics.
    :return Ws: The inferred covariance matrices
    :return Ts: The inferred precision matrices
    '''
    Xs = [np.copy(X) for X in Xs]  # We are going to scale these, so copy
    Ns = [X.shape[0] for X in Xs]
    P = Xs[0].shape[1]  # a.k.a. n_features
    K = len(Xs)
    assert(all([X.shape == (Ns[k], P) for k, X in enumerate(Xs)]))

    # 0-center and scale the Xs
    means = [np.mean(X, axis=0) for X in Xs]
    variances = [np.var(X, axis=0) for X in Xs]
    for k in range(K):
        for p in range(P):
            Xs[k][:, p] = Xs[k][:, p] - means[k][p]
            Xs[k][:, p] = Xs[k][:, p] / np.sqrt(variances[k][p])

    # Caching some computation
    Ss = [X.T @ X / Ns[k] for k, X in enumerate(Xs)]
    # Initialize Ts
    if smart_init:
        Ts = [np.linalg.inv(S + np.eye(P)) for S in Ss]
    else:
        Ts = [np.ones(shape=(P, P)) for _ in range(K)]
    Ws = [np.eye(P) for _ in range(K)]  # Doesn't really matter
    it = 0
    while True:
        it += 1
        if verbose:
            print(f"joint_graphical_lasso_guo iteration # {it}")
        Ts_old = [np.copy(T) for T in Ts]
        # Approximate penalty, split the problem, and solve with weighted
        # Graphical Lasso
        # 1) Approximate penalty (thresholing with 1e-10 for numerical
        # stability)
        Tau = 1.0 / np.maximum(np.sqrt(sum([np.abs(T) for T in Ts])), 1e-10)
        np.fill_diagonal(Tau, 0)  # Do not penalize the diagonal
        # 2) Solve each weighted Graphical Lasso subproblem
        for k in range(K):
            if fast:
                Ws[k], Ts[k] = quic_graphical_lasso_fast(Xs[k], l * Tau, Ss[k],
                                                         verbose=False)
            else:
                quic_results = QuicGraphicalLasso(lam=l * Tau).fit(Xs[k])
                Ws[k], Ts[k] = \
                    quic_results.covariance_, quic_results.precision_
        # Check convergence
        numerator = sum([np.sum(np.abs(Ts[k] - Ts_old[k])) for k in range(K)])
        denominator = sum([np.sum(np.abs(Ts_old[k])) for k in range(K)])
        relative_change_in_thetas = numerator / denominator
        if verbose:
            tot_edges = sum([(np.sum(Ts[i] != 0) -
                              np.sum(Ts[i].diagonal() != 0)) /
                            2.0 for i in range(len(Ts))])
            print(f"\tTotal number of edges inferred = {tot_edges}")
            print(f"\tRelative change in Thetas = {relative_change_in_thetas}")
        if relative_change_in_thetas < threshold:
            break

    # Rescale results
    for k in range(K):
        for p in range(P):
            Ws[k][:, p] = Ws[k][:, p] * np.sqrt(variances[k][p])
            Ws[k][p, :] = Ws[k][p, :] * np.sqrt(variances[k][p])
            Ts[k][:, p] = Ts[k][:, p] / np.sqrt(variances[k][p])
            Ts[k][p, :] = Ts[k][p, :] / np.sqrt(variances[k][p])

    return Ws, Ts
