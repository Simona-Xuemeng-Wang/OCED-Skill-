import pandas as pd
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from factor_analyzer.rotator import Rotator
from factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from scipy.linalg import eigvals


def parallel_analysis(df):
    # Eigenvalues of the correlation matrix (for EFA)
    correlation_matrix = df.corr().values
    eigenvalues = eigvals(correlation_matrix)
    eigenvalues = np.real(eigenvalues)  # take the real part
    eigenvalues = np.sort(eigenvalues)[::-1]  # sort in descending order

    # Simple parallel analysis: eigenvalues of random data as a benchmark
    n_vars, n_obs = df.shape[1], df.shape[0]
    random_eigenvalues = []

    # Eigenvalues from 100 random datasets
    for _ in range(100):  # use _ as the throwaway loop variable
        random_data = np.random.normal(0, 1, size=(n_obs, n_vars))
        random_corr = np.corrcoef(random_data.T)
        random_eig = eigvals(random_corr)
        random_eig = np.real(random_eig)
        random_eig = np.sort(random_eig)[::-1]
        random_eigenvalues.append(random_eig)

    # 99th percentile of the random-data eigenvalues
    random_99th = np.percentile(random_eigenvalues, 99, axis=0)

    # Suggested number of factors (computed before printing)
    suggested_factors = sum(eigenvalues > random_99th)
    kaiser_factors = sum(eigenvalues > 1)

    # Scree plot
    plt.figure(figsize=(10, 6))
    plt.plot (range(1, len(eigenvalues) + 1), eigenvalues, 'bo-',
             linewidth=2, markersize=8, label='real_data')
    plt.plot(range(1, len(random_99th) + 1), random_99th, 'r--',
             linewidth=2, label='random data 99% Percentile')


    plt.title('EFA Scree Plot with Parallel Analysis')
    plt.xlabel('Factor Number')
    plt.ylabel('Eigenvalue')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print results
    print(f"Parallel suggested: {suggested_factors} factors")
    print(f"Using {suggested_factors} factors for EFA (based on parallel analysis)")

    # Return the suggested number of factors
    return suggested_factors