
import pandas as pd
def cronbach_alpha(data):
    k = data.shape[1]
    item_var_sum = data.var().sum()
    total_var = data.sum(axis=1).var()
    # Compute alpha
    alpha = (k / (k - 1)) * (1 - item_var_sum / total_var)
    return alpha


def analyze_alpha_if_deleted(data):
    original_alpha = cronbach_alpha(data)
    print(f"Original Cronbach's Alpha: {original_alpha:.4f}")
    print("-" * 50)
    results = []
    for col in data.columns:

        data_without_col = data.drop(columns=[col])
        new_alpha = cronbach_alpha(data_without_col)
        change = new_alpha - original_alpha

        results.append({
            'Dropped item': col,
            'New alpha': new_alpha,
            'Alpha change': change,
            'Improved': '✓' if change > 0 else '✗'
        })

        print(f"Drop {col:10}: Alpha = {new_alpha:.4f} (change: {change:+.4f}) {'✓' if change > 0 else '✗'}")
    results_df = pd.DataFrame(results)
    best_to_remove = results_df.loc[results_df['Alpha change'].idxmax()]

    print("\n" + "=" * 50)
    print("Recommendation:")
    if best_to_remove['Alpha change'] > 0:
        print(f"Dropping '{best_to_remove['Dropped item']}' raises alpha the most")
        print(f"Increase: {best_to_remove['Alpha change']:.4f}")
        print(f"New alpha: {best_to_remove['New alpha']:.4f}")
    else:
        print("Dropping any item does not raise alpha; keep all items")

    return results_df
