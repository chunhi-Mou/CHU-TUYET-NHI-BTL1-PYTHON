import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# --- Configuration ---
INPUT_CSV = 'Report/OUTPUT/OUTPUT_PART1/results.csv'
OUTPUT_FOLDER = 'Report/OUTPUT/OUTPUT_PART3'
ELBOW_PLOT_PATH = os.path.join(OUTPUT_FOLDER, 'kmeans_elbow_plot.png')
PCA_PLOT_PATH = os.path.join(OUTPUT_FOLDER, 'pca_kmeans_cluster_plot.png')

IDENTIFIER_COLS = ['Player', 'Nation', 'Team', 'Position', 'Age']

K_RANGE = range(2, 16)

# <-----CHOSEN K----->
OPTIMAL_K = 6

KEY_STATS_FOR_INTERPRETATION = [
    'Goalkeeping: Performance: Save%', 'Goalkeeping: Performance: GA90', 'Defensive Actions: Tackles: TklW',
    'Defensive Actions: Blocks: Int', 'Defensive Actions: Pressures: Press', 'Miscellaneous Stats Aerial Duels: Won%',
    'Miscellaneous Stats Performance: Recov', 'Passing: Total: Cmp%', 'Possession: Touches: Mid 3rd',
    'Possession: Touches: Att Pen', 'Passing: Expected: PrgP', 'Possession: Carries: PrgC',
    'Possession: Take-Ons: Succ%', 'Passing: Expected: KP', 'Passing: Expected: PPA',
    'Miscellaneous Stats Performance: Crs', 'Goal and Shot Creation: SCA: SCA90', 'Possession: Receiving: PrgR',
    'Per 90 minutes: Ast', 'Per 90 minutes: Gls', 'Shooting: Standard: SoT/90', 'Expected: xG'
]

def load_data(filepath):
    try: 
        df = pd.read_csv(filepath)
        return df
    except: print(f"Error load file: {filepath}")

def preprocess_data(df, id_cols):
    ids = [c for c in id_cols if c in df.columns]
    df_ids = df[ids].copy()

    feats = df.drop(columns=ids).apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    if feats.empty:
        print("No usable features")
        sys.exit(1)

    if feats.isnull().values.any():
        feats = pd.DataFrame(SimpleImputer(strategy='median').fit_transform(feats), columns=feats.columns)
    
    scaled = pd.DataFrame(StandardScaler().fit_transform(feats), columns=feats.columns)
    return scaled, feats, df_ids

def plot_elbow_method(df_scaled, k_range, output_path):
    inertia = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init='auto')
        model.fit(df_scaled); inertia.append(model.inertia_)

    plt.plot(k_range, inertia, 'o--'); plt.xlabel('k'); plt.ylabel('Inertia')
    plt.title('Elbow Method'); plt.xticks(list(k_range)); plt.grid(True)
    plt.tight_layout(); plt.savefig(output_path); plt.close()
    print(f"Elbow plot saved to: {output_path}")

def run_kmeans(df_scaled, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(df_scaled)
    print(f"K-means DONE. Players assigned to {n_clusters} clusters.")
    return cluster_labels

def get_comparison_label(stat_name, cluster_val, overall_val):
    diff_ratio = (cluster_val - overall_val) / overall_val
    is_lower_better = stat_name == 'Goalkeeping: Performance: GA90'

    if diff_ratio > 0.20:
        return "High"
    elif diff_ratio < -0.20:
        return "Low (Good)" if is_lower_better else "Low"
    elif diff_ratio > 0.05:
         return "Slightly High"
    elif diff_ratio < -0.05:
        return "Slightly Low (Good)" if is_lower_better else "Slightly Low"
    else:
        return "Average"

def analyze_clusters(df_original, df_imputed, k, key_stats):
    stats = [s for s in key_stats if s in df_imputed.columns]
    overall_means = df_imputed.drop(columns='Cluster')[stats].mean()
    cluster_means = df_imputed.groupby('Cluster')[stats].mean()
    total_players = len(df_original)

    print(f"\nK={k} | Stats: {len(stats)} used\n")

    for i in range(k):
        cluster_data = df_original[df_original['Cluster'] == i]
        n = len(cluster_data)
        print(f"Cluster {i}: {n} players ({n/total_players:.1%})")

        for id_col in ['Position', 'Team']:
            if id_col in cluster_data.columns:
                top = cluster_data[id_col].value_counts().head(1)
                if not top.empty:
                    print(f"  Top {id_col}: {top.index[0]} ({top.iloc[0]})")

        hints = []
        for stat in stats:
            cm, om = cluster_means.loc[i][stat], overall_means[stat]
            label = get_comparison_label(stat, cm, om)
            if label not in ["Average", "Slightly High", "Slightly Low", "Data Missing"]:
                hints.append(f"{label.lower()} {stat.split(':')[-1].strip()}")

        hint = ", ".join(hints) if hints else "near average"
        print(f"Hint: {hint}\n")

def visualize_pca(df_scaled_with_clusters, df_identifiers, k, output_path):
    pca = PCA(n_components=2)
    comps = pca.fit_transform(df_scaled_with_clusters.drop(columns='Cluster'))
    df_pca = pd.DataFrame(comps, columns=['PC1', 'PC2'], index=df_scaled_with_clusters.index)
    df_pca['Cluster'] = df_scaled_with_clusters['Cluster']
    if 'Player' in df_identifiers: df_pca['Player'] = df_identifiers['Player']
    
    evr = pca.explained_variance_ratio_
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster',
                    palette=sns.color_palette('viridis', k), alpha=0.75)
    plt.title(f'Clusters (K={k}) via PCA')
    plt.xlabel(f'PC1 ({evr[0]:.1%})'); plt.ylabel(f'PC2 ({evr[1]:.1%})')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title=f'Cluster (K={k})', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.figtext(0.5, 0.01, f"Total Variance Explained: {evr.sum():.1%}", ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.7, "pad":4})
    plt.tight_layout(); plt.savefig(output_path); plt.close()

if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    # 1. Load Data
    df_original = load_data(INPUT_CSV)
    # 2. Preprocess Data
    df_scaled, df_imputed, df_identifiers = preprocess_data(df_original, IDENTIFIER_COLS)
    # 3. Elbow Method
    plot_elbow_method(df_scaled, K_RANGE, ELBOW_PLOT_PATH)
    # 4. K-Means Clustering
    cluster_labels = run_kmeans(df_scaled, OPTIMAL_K)
    # 5. Add Cluster Labels back to DataFrames
    df_original['Cluster'] = cluster_labels
    df_scaled['Cluster'] = cluster_labels
    df_imputed['Cluster'] = cluster_labels
    print("\nCluster distribution:")
    print(df_original['Cluster'].value_counts().sort_index())
    # 6. Cluster Analysis
    analyze_clusters(df_original, df_imputed, OPTIMAL_K, KEY_STATS_FOR_INTERPRETATION)
    # 7. PCA Visualization
    visualize_pca(df_scaled, df_identifiers, OPTIMAL_K, PCA_PLOT_PATH)
    print("\n--- Finished ---")