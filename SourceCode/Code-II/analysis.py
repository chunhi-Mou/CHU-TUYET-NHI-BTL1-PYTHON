import pandas as pd
from config_part2 import *

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

def print_top_team_per_statistic(summary_df, relevant_stats):
    print("\n--- The team with the highest scores for each statistic ---")
    teams_df = summary_df[summary_df['Team'] != 'all'].copy()
    if teams_df.empty:
        print("No team data")
        return

    top_teams_for_positive_stats = []
    positive_stats_considered = []

    for stat in relevant_stats:
        col = f'Mean of {stat}'
        if col not in teams_df.columns:
            print(f"Col '{col}' not found.")
            continue
        teams_df[col] = pd.to_numeric(teams_df[col], errors='coerce')
        valid_df = teams_df.dropna(subset=[col])
        if valid_df.empty:
            print(f"No valid data for '{col}'.")
            continue
        top_row = valid_df.loc[valid_df[col].idxmax()]
        top_team = top_row['Team']
        top_value = top_row[col]
        print(f"Top team for '{stat}': {top_team} (mean = {top_value:.3f})")
        
        if stat not in NEGATIVE_STATS:
            top_teams_for_positive_stats.append(top_team)
            positive_stats_considered.append(stat)

    if not top_teams_for_positive_stats:
        print("No top teams")
        return

    top_team_series = pd.Series(top_teams_for_positive_stats)
    most_common_team = top_team_series.value_counts().idxmax()
    most_common_count = top_team_series.value_counts().max()
    print(f"\nTop: {most_common_team} (appeared {most_common_count} times / {len(positive_stats_considered)} non-negative statistics considered)")


