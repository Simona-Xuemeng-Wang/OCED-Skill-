import pandas as pd
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from factor_analyzer.rotator import Rotator
from factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from scipy.linalg import eigvals

def run_sklearn_efa_with_rotation(df, n_factors=3 , threshold=0.41):
    # step 1 get the Kaiser-Meyer-Olking Measure of Sampling Adequacy and Bartlett's Test of Sphericity
        # KMO 检验
    kmo_all, kmo_model = calculate_kmo(df)
    print("\nKMO 检验结果:")
    print(f"KMO 总体指标: {kmo_model:.3f}")

    # Bartlett's 球形检验
    chi_square_value, p_value = calculate_bartlett_sphericity(df)
    print("\nBartlett's 球形检验:")
    print(f"卡方值: {chi_square_value:.3f}")
    print(f"p 值: {p_value:.4f}")

    # KMO 解释
    if kmo_model < 0.5:
        print("警告：数据不适合进行因子分析（KMO < 0.5）")
    elif kmo_model < 0.6:
        print("KMO: 不可接受")
    elif kmo_model < 0.7:
         print("KMO: 一般")
    elif kmo_model < 0.8:
        print("KMO: 良好")
    else:
        print("KMO: 非常好")

        # Bartlett's 检验解释
    if p_value > 0.05:
        print("警告：变量间相关性不显著，不适合因子分析")

    # Step 3: 运行 EFA（sklearn 版本）
    fa = FactorAnalysis(n_components=n_factors)
    fa.fit(df)

    # Step 4: raw factor loading matrix
    raw_loadings = fa.components_.T  # shape: [n_features, n_factors]

    # raw pattern matrix
    raw_pattern_matrix = pd.DataFrame(raw_loadings, index=df.columns,
                                      columns=[f'Factor{i + 1}' for i in range(n_factors)])
    print("Raw Factor Loadings (Before Rotation):")
    print(raw_pattern_matrix.round(3))

    # Step 5:  quartimin Rotation
    rotator = Rotator(method='quartimin')
    rotated_loadings = rotator.fit_transform(raw_loadings)
    # Get the communality
    communality = np.sum(rotated_loadings** 2, axis=1)
    communality_df = pd.DataFrame({
    'Variable': df.columns,
    'Communality': communality.round(3)})
    # Step 6: 整理模式矩阵（Pattern Matrix）
    pattern_matrix = pd.DataFrame(rotated_loadings, index=df.columns,
                                  columns=[f'Factor{i + 1}' for i in range(n_factors)])
    print("Pattern Matrix (Rotated Factor Loadings):")
    print(pattern_matrix.round(3))
    # 打印因子相关矩阵
    factor_correlation_matrix = rotator.phi_
    factor_corr_df = pd.DataFrame(factor_correlation_matrix,
                                  index=[f'Factor{i + 1}' for i in range(n_factors)],
                                  columns=[f'Factor{i + 1}' for i in range(n_factors)])
    print("\nFactor Correlation Matrix:")
    print(factor_corr_df.round(3))
    # CLuster explaination
    factor_loadings = []
    for i in range(n_factors):
        # Get high loading variables for each factor
        high_loadings = pattern_matrix[pattern_matrix[f'Factor{i + 1}'].abs() >= threshold].index.tolist()

        # Create a list of dictionaries with factor details
        for var in high_loadings:
            value = pattern_matrix.loc[var, f'Factor{i + 1}']
            factor_loadings.append({
                'Factor': f'Factor {i + 1}',
                'Variable': var,
                'Loading': value
            })

    # Convert to DataFrame
    factor_loadings_df = pd.DataFrame(factor_loadings)
    # Get explained variance
    # Sum squared loadings for each factor
    factor_variances = np.sum(rotated_loadings ** 2, axis=0)
    n_variables = rotated_loadings.shape[0]  # 变量数，而不是样本数
    variance_percentage = (factor_variances / n_variables) * 100

    # Create DataFrame for results
    variance_df = pd.DataFrame({
        'Factor': [f'Factor {i + 1}' for i in range(len(factor_variances))],
        'Variance Explained (%)': variance_percentage,
        'Cumulative Variance (%)': np.cumsum(variance_percentage).round(2)
    })


    return {
            'pattern_matrix' : pattern_matrix.round(2),
            'factor_correlation_matrix': factor_corr_df,
             'communalities': communality_df,
            'high_loadings': factor_loadings_df,
            'Explained Variance': variance_df
            }








