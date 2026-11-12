from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd



def perform_pca_analysis(df, threshold):
    # ---------------------------------------------------------
    # step 1 set up the correlation matrix
    # Set up the correlation matrx
    # filter the low correlation

    corr_matrix = df.corr(method='spearman')
    # Threshold
    avg_corr = corr_matrix.abs().mean(numeric_only=True)  # Calculate the average variables
    selected = avg_corr[avg_corr > threshold].index.tolist()  # Filter the variables

    # Filter dataset
    df_selected = df[selected].copy()
    # --------------------------------------------------------
    # Step 2:Missing value percentage
    missing_percent = (df_selected.isnull().sum() / df.shape[0] * 100).rename("Missing (%)")

    # --------------------------------------------------------
    # A. Dropna // Maybe apply the median imputation
    # --------------------------------------------------------
    # Standarized
    df_clean = df_selected.dropna().copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    # PCA
    pca = PCA(n_components=2)
    pca.fit(X_scaled)

    # PCA Score
    X_pca_score = pca.transform(X_scaled)
    pc1_score = X_pca_score[:, 0]
    pca_scores = pd.DataFrame(pc1_score, columns=["PC1 Score"])
    # PCA loadings
    pc1_loadings = pd.DataFrame(
        pca.components_[0],
        index=df_clean.columns,
        columns=["PC1 (Original)"])

    # The number of observation
    obs = df_selected.count().rename("Obs.")

    # Explained variance ratio table
    pca_explained = pd.Series(
        pca.explained_variance_ratio_,
        index=[f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))]).round(3)

    pca_eigenvalues = pd.Series(pca.explained_variance_,index=[f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))]).round(3)
    print("\n=== PCA Eigenvalues")
    print(pca_eigenvalues)

    merged_table = pd.concat([
        missing_percent,
        obs,
        pc1_loadings], axis=1)

    return { "PCA Eigenvale": pca_eigenvalues , "PCA_explained": pca_explained, "PC1_loadings": pc1_loadings, "merged_table": merged_table,
             "pca_score": pca_scores }