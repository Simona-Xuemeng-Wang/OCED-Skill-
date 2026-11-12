import numpy as np
import warnings


def calculate_factor_scores(X, L, Phi):
    """
    Parameters:
    X: standardized raw data matrix, shape (n, p)
    L: pattern matrix after oblique rotation, shape (p, m)
    Phi: factor correlation matrix, shape (m, m)

    Notes:
    The column order of X must match the row order of L;
    the row/column order of Phi must match the column order of L.

    Returns:
    Factor score matrix, shape (n, m)
    """

    # Check inputs
    X = np.asarray(X, dtype=float)
    L = np.asarray(L, dtype=float)
    Phi = np.asarray(Phi, dtype=float)

    # Correlation matrix of the observed variables
    R = np.corrcoef(X.T)

    # Structure matrix under oblique rotation
    structure = L @ Phi

    # Regression method: compute weights W = Phi @ L.T @ R^(-1)
    try:
        W = np.linalg.solve(R, structure).T

    except np.linalg.LinAlgError:
        warnings.warn("Correlation matrix is singular; using the pseudo-inverse")
        R_pinv = np.linalg.pinv(R)
        W = structure.T @ R_pinv

    # Compute factor scores (no orthogonalization)
    factor_scores = X @ W.T

    return factor_scores