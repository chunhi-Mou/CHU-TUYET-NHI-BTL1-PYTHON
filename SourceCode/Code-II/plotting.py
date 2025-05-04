import os
import seaborn as sns
import matplotlib.pyplot as plt
from config_part2 import PLOTS_FOLDER

def _safe_filename(col_name):
    return "".join(c if c.isalnum() else "_" for c in col_name)

def _save_plot(fig, filename_base, fmt='png'):
    filename = f"{filename_base}.{fmt}"
    path = os.path.join(PLOTS_FOLDER, filename)
    try:
        fig.savefig(path, format=fmt)
        print(f"Saved: {path}")
    except Exception as e:
        print(f"Error saving {path}: {e}")
    finally:
        plt.close(fig)

def plot_histogram_all_players(df, stat_col, bins=20, fmt='png', xlim=None, ylim=None):

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[stat_col].dropna(), bins=bins, kde=True, ax=ax)
    ax.set(title=f'Distribution of {stat_col} (All Players)', xlabel=stat_col, ylabel='Frequency')
    
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)

    fig.tight_layout()
    _save_plot(fig, f'hist_all_{_safe_filename(stat_col)}', fmt)

def plot_histograms_per_team_facet(df, stat_col, col_wrap=4, bins=15, fmt='png', xlim=None, ylim=None):

    df_valid = df.dropna(subset=[stat_col, 'Team'])
    if df_valid.empty or df_valid['Team'].nunique() == 0:
        print(f"Warn: No valid data for '{stat_col}' and 'Team'")
        return

    print(f"Plotting '{stat_col}' for {df_valid['Team'].nunique()} teams")

    try:
        g = sns.FacetGrid(df_valid, col="Team", col_wrap=col_wrap, sharex=True, sharey=False, height=3, aspect=1.2)
        g.map(sns.histplot, stat_col, bins=bins)

        if xlim or ylim:
            for ax in g.axes.flatten():
                if xlim: ax.set_xlim(xlim)
                if ylim: ax.set_ylim(ylim)

        g.set_titles("{col_name}")
        g.set_axis_labels(stat_col, "Frequency")
        plt.suptitle(f'Distribution of {stat_col} per Team', y=1.02)
        g.tight_layout(rect=[0, 0.03, 1, 0.98])

        _save_plot(g.fig, f'hist_facet_team_{_safe_filename(stat_col)}', fmt)

    except Exception as e:
        print(f"Error: {e}")
        plt.close()