import pandas as pd
from config_part2 import *
from analysis import *
from plotting import *

def load_data():
    try:
        df = pd.read_csv(INPUT_CSV)
        numeric_cols = [col for col in df.columns if col not in EXCLUDED_COLUMNS_FROM_TOP3]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '', regex=False), errors='coerce')
            else:
                print(f"Warn: Col '{col}' not found")
        return df, numeric_cols
    except FileNotFoundError:
        print(f"Error: {INPUT_CSV} not found")
    except Exception as e:
        print(f"Error loading data: {e}")
    return None, []

def find_top_bottom_players_all_stats(df, numeric_cols):
    try:
        with open(TOP_3_FILE, 'w', encoding='utf-8-sig') as f:
            f.write("--- Top and Bottom 3 Players per Statistic ---\n")
            for stat in numeric_cols:
                top, bottom = get_top_bottom_players(df, stat, n=3)
                f.write(f"\n\n--- {stat} ---\n")
                for label, group in [("Top 3", top), ("Bottom 3", bottom)]:
                    data = group.to_string(index=False) if group is not None and not group.empty else "No valid data"
                    f.write(f"\n{label}:\n{data}\n")
        print(f"Saved to {TOP_3_FILE}")
    except IOError as e:
        print(f"Error writing file: {e}")

def generate_statistics_summary(df, numeric_cols):
    summary_df = calculate_stats_summary(df, numeric_cols)
    if summary_df is not None:
        try:
            if 'Team' in summary_df.columns:
                cols = ['Team'] + [col for col in summary_df.columns if col != 'Team']
                summary_df = summary_df[cols]
            summary_df.to_csv(STATS_SUMMARY_FILE, index=False, float_format='%.3f')
            print(f"Saved summary to {STATS_SUMMARY_FILE}")
        except Exception as e:
            print(f"Error saving summary: {e}")
    else:
        print("Failed to calculate summary")
    return summary_df

def generate_histograms(df):
    for stat in SELECTED_STATS:
        if stat in df.columns:
            print(f"Plot {stat}")
            plot_histogram_all_players(df, stat)
            plot_histograms_per_team_facet(df, stat)
        else:
            print(f"Warn: {stat} not found")
    print(f"Plots saved in {PLOTS_FOLDER}")

def main():
    print("--- Start Part II ---")
    df, numeric_cols = load_data()
    if df is None: return
    find_top_bottom_players_all_stats(df, numeric_cols)
    summary_df = generate_statistics_summary(df, numeric_cols)
    print_top_team_per_statistic(summary_df, numeric_cols) 
    generate_histograms(df)
    print("--- Part II Complete ---")

if __name__ == "__main__":
    main()
