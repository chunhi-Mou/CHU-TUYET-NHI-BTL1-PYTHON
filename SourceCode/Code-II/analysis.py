import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

def get_top_bottom_players(df, stat_col, n=3):
    df_valid = df[['Player', 'Team', stat_col]].copy()
    df_valid[stat_col] = pd.to_numeric(df_valid[stat_col], errors='coerce')
    df_valid = df_valid.dropna(subset=[stat_col])

    if df_valid.empty:
        print(f"Warn: No valid data of '{stat_col}'.")
        return None, None

    df_sorted = df_valid.sort_values(stat_col, ascending=False)
    return df_sorted.head(n), df_sorted.tail(n).sort_values(stat_col)

def calculate_stats_summary(df, stats_cols):
    valid_cols = []
    df_copy = df.copy()
    for col in stats_cols:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            valid_cols.append(col)
        else:
            print(f"Warn: Col '{col}' not found.")

    if not valid_cols:
        print("Error: No valid cols")
        return None

    team_stats = pd.DataFrame()
    if 'Team' in df_copy.columns and not df_copy['Team'].isnull().all():
        grouped = df_copy.groupby('Team')
        team_stats_agg = grouped[valid_cols].agg(['median', 'mean', 'std'])

        if isinstance(team_stats_agg.columns, pd.MultiIndex):
            team_stats_agg.columns = ['_'.join(col).strip() for col in team_stats_agg.columns.values]
            team_stats = team_stats_agg.reset_index()

            new_colnames = {'Team': 'Team'}
            for col in valid_cols:
                if f'{col}_median' in team_stats.columns:
                     new_colnames[f'{col}_median'] = f'Median of {col}'
                if f'{col}_mean' in team_stats.columns:
                     new_colnames[f'{col}_mean'] = f'Mean of {col}'
                if f'{col}_std' in team_stats.columns:
                     new_colnames[f'{col}_std'] = f'Std of {col}'
            team_stats = team_stats.rename(columns=new_colnames)
        else:
            team_stats = team_stats_agg.reset_index()

    else:
        print("Warn: 'Team' col is null")
        team_stats = pd.DataFrame(columns=['Team'] + [f'{stat} of {col}' for col in valid_cols for stat in ['Median', 'Mean', 'Std']])


    overall_stats_data = {'Team': 'all'}
    for col in valid_cols:
        col_median = df_copy[col].median()
        col_mean = df_copy[col].mean()
        col_std = df_copy[col].std()
        overall_stats_data[f'Median of {col}'] = col_median
        overall_stats_data[f'Mean of {col}'] = col_mean
        overall_stats_data[f'Std of {col}'] = col_std

    overall_df = pd.DataFrame([overall_stats_data])

    final_cols_order = list(team_stats.columns)
    overall_df_ordered = overall_df[final_cols_order]
    summary = pd.concat([overall_df_ordered, team_stats], ignore_index=True)

    return summary

def find_top_performing_teams_pca(df, pca_cols, negative_cols, n_top=5):

    df = df[df['Team'] != 'all'].copy()
    cols = [f'Median of {col}' for col in pca_cols if f'Median of {col}' in df.columns]
    if not cols:
        return None

    data = df[['Team'] + cols].copy()
    data[cols] = data[cols].apply(pd.to_numeric, errors='coerce')

    for col in [f'Median of {c}' for c in negative_cols if f'Median of {c}' in cols]:
        data[col] *= -1

    imputed = SimpleImputer(strategy='median').fit_transform(data[cols])
    scaled = StandardScaler().fit_transform(imputed)
    if np.isnan(scaled).all() or scaled.shape[1] == 0:
        return None

    try:
        pc1 = PCA(n_components=1).fit_transform(scaled).flatten()
    except Exception:
        return None

    return pd.DataFrame({'Team': data['Team'], 'PCA_Score': pc1}) \
            .sort_values('PCA_Score', ascending=False) \
            .reset_index(drop=True).head(n_top)
