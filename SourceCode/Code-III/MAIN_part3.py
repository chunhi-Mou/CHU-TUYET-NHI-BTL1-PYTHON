import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# --- Configuration ---
INPUT_CSV = 'Report/OUTPUT_PART1/results.csv'
OUTPUT_FOLDER = 'Report/OUTPUT_PART3'
ELBOW_PLOT_PATH = os.path.join(OUTPUT_FOLDER, 'kmeans_elbow_plot.png')
PCA_PLOT_PATH = os.path.join(OUTPUT_FOLDER, 'pca_kmeans_cluster_plot.png')
CLUSTER_ANALYSIS_FILE_PATH = os.path.join(OUTPUT_FOLDER, 'cluster_analysis.txt')

IDENTIFIER_COLS = ['Player', 'Nation', 'Team', 'Position', 'Age']

K_RANGE = range(2, 16)

# <-----CHOSEN K----->
OPTIMAL_K = 8

KEY_STATS_FOR_INTERPRETATION = [
    # --- Thủ môn (Goalkeeping) ---
    # Đánh giá khả năng cản phá và hiệu quả chung
    'Goalkeeping: Performance: Save%',  # Tỷ lệ cứu thua
    'Goalkeeping: Performance: GA90',    # Bàn thua trung bình mỗi 90 phút

    # --- Phòng ngự (Defending) ---
    # Các hành động phòng ngự chủ động và thu hồi bóng
    'Defensive Actions: Tackles: TklW',        # Số lần tắc bóng thành công (Ball Winning roles)
    'Defensive Actions: Blocks: Int',          # Số lần cắt bóng (Stoppers)
    'Defensive Actions: Pressures: Press',     # Tổng số lần gây áp lực (Pressing roles, Ball Winning Mid) - Cần kiểm tra tên cột chính xác từ results.csv
    'Miscellaneous Stats Aerial Duels: Won%',  # Tỷ lệ thắng không chiến (Stoppers, Target Man)
    'Miscellaneous Stats Performance: Recov',  # Số lần thu hồi bóng (Ball Winning roles, Defensive Backs)

    # --- Kiểm soát & Xây dựng lối chơi (Possession & Build-up) ---
    # Khả năng chuyền bóng và sự tham gia vào lối chơi
    'Passing: Total: Cmp%',             # Tỷ lệ chuyền chính xác tổng thể (Playmakers, Ball Playing Defenders)
    'Possession: Touches: Mid 3rd',     # Số lần chạm bóng ở 1/3 giữa sân (Midfield involvement)
    'Possession: Touches: Att Pen',     # Số lần chạm bóng trong vòng cấm đối phương (Poachers, Inside Forwards)

    # --- Phát triển bóng (Progression) ---
    # Khả năng đưa bóng lên phía trước bằng chuyền hoặc rê dắt
    'Passing: Expected: PrgP',          # Số đường chuyền tịnh tiến (Playmakers, Ball Playing Defenders)
    'Possession: Carries: PrgC',        # Số lần rê bóng tịnh tiến (Wing Backs, Mobile Strikers)
    'Possession: Take-Ons: Succ%',      # Tỷ lệ qua người thành công (Dribblers, Wingers, Inside Forwards) - Cần kiểm tra tên cột chính xác từ results.csv

    # --- Sáng tạo (Creation) ---
    # Tạo cơ hội cho đồng đội qua nhiều cách khác nhau
    'Passing: Expected: KP',            # Số đường chuyền quyết định (Key Passes - Playmakers)
    'Passing: Expected: PPA',           # Số đường chuyền vào vòng cấm (Playmakers, Attacking Midfielders)
    'Miscellaneous Stats Performance: Crs', # Số lần tạt bóng (Wingers, Wing Backs) - Cần kiểm tra tên cột chính xác từ results.csv
    'Goal and Shot Creation: SCA: SCA90', # Số hành động tạo cú sút mỗi 90 phút (Creative players)
    'Possession: Receiving: PrgR',      # Số lần nhận đường chuyền tịnh tiến (Advanced players, Forwards)
    'Per 90 minutes: Ast',              # Số kiến tạo mỗi 90 phút

    # --- Dứt điểm (Finishing) ---
    # Khả năng ghi bàn và chất lượng cú sút
    'Per 90 minutes: Gls',              # Số bàn thắng mỗi 90 phút (Forwards)
    'Shooting: Standard: SoT/90',       # Số cú sút trúng đích mỗi 90 phút (Forwards)
    'Expected: xG'                      # Bàn thắng kỳ vọng (Chất lượng cơ hội)
]

def load_data(filepath):
    try: 
        df = pd.read_csv(filepath)
        return df
    except: print(f"Error load file: {filepath}")

def preprocess_data(df, id_cols):
    ids = [c for c in id_cols if c in df.columns]
    df_ids = df[ids].copy()

    df_features = df.drop(columns=ids)
    for col in df_features.select_dtypes(include='object'):
        df_features[col] = df_features[col].astype(str).str.replace(',', '', regex=False)

    feats = df.drop(columns=ids).apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    if feats.empty: print("No usable features."); sys.exit(1)

    if feats.isnull().values.any():
        feats = pd.DataFrame(SimpleImputer(strategy='median').fit_transform(feats), columns=feats.columns)
    
    scaled = pd.DataFrame(StandardScaler().fit_transform(feats), columns=feats.columns)
    return scaled, feats, df_ids

def plot_elbow_method(df_scaled, k_range, output_path):
    inertia = []
    for k in k_range:
        try: model = KMeans(n_clusters=k, random_state=42, n_init='auto')
        except: model = KMeans(n_clusters=k, random_state=42, n_init=10)
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
    if overall_val is None or cluster_val is None or pd.isna(overall_val) or pd.isna(cluster_val):
        return "Data Missing"
    if abs(overall_val) < 1e-9:
         return "Average (Overall Mean Near Zero)"

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

def analyze_clusters(df_original, df_imputed, k, key_stats, output_file):
    print("\nCluster Analysis")

    stats = [s for s in key_stats if s in df_imputed.columns]
    print(f"\nUsing {len(stats)} key stats:\n{stats}")

    overall_means = df_imputed.drop(columns='Cluster')[stats].mean()
    cluster_means = df_imputed.groupby('Cluster')[stats].mean()
    total_players = len(df_original)

    with open(output_file, 'w', encoding='utf-8-sig') as f:
        f.write(f"Cluster Report (K={k})\n{'='*50}\n")
        f.write(f"From: {INPUT_CSV}\nK chosen: {k} ({os.path.basename(ELBOW_PLOT_PATH)})\n{'='*50}\n\n")

        f.write("Overall Averages:\n")
        for stat in stats:
            f.write(f"- {stat}: {overall_means[stat]:.2f}\n")
        f.write(f"\n{'='*50}\n\n")

        for i in range(k):
            cluster_data = df_original[df_original['Cluster'] == i]
            num_players = len(cluster_data)
            f.write(f"--- Cluster {i} ---\n")
            f.write(f"{num_players} players ({num_players / total_players:.1%})\n")

            for id_col in ['Position', 'Team']:
                if id_col in cluster_data.columns:
                    common = cluster_data[id_col].value_counts().head(3)
                    f.write(f"Top {id_col}s:\n")
                    for item, count in common.items():
                        f.write(f"  - {item}: {count} ({count / num_players:.1%})\n")
                else:
                    f.write(f"{id_col} not found\n")

            f.write("\nKey Stats:\n")
            hints = []
            for stat in stats:
                cm, om = cluster_means.loc[i][stat], overall_means[stat]
                label = get_comparison_label(stat, cm, om)
                if label != "Data Missing":
                    f.write(f"- {stat}: {cm:.2f} (vs. {om:.2f}) → [{label}]\n")
                    if label not in ["Average", "Slightly High", "Slightly Low"]:
                        hints.append(f"{label.lower()} {stat.split(':')[-1].strip()}")
                else:
                    f.write(f"- {stat}: Data Missing\n")

            f.write("\nHint:\n")
            hint = f"Cluster {i} tends to have: {', '.join(hints)}." if hints else \
                   f"Cluster {i} is near the overall average."
            if 'Position' in cluster_data.columns and 'common' in locals() and not common.empty:
                hint += f" Common positions: {', '.join(common.index)}."
            f.write(hint + "\n" + "-"*40 + "\n\n")

    print(f"\nSaved to: {output_file}")

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

# --- Main Execution ---
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
    analyze_clusters(df_original, df_imputed, OPTIMAL_K, KEY_STATS_FOR_INTERPRETATION, CLUSTER_ANALYSIS_FILE_PATH)

    # 7. PCA Visualization
    visualize_pca(df_scaled, df_identifiers, OPTIMAL_K, PCA_PLOT_PATH)

    print("\n--- Script Finished ---")