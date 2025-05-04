import os

INPUT_CSV = 'Report/OUTPUT_PART1/results.csv'
OUTPUT_FOLDER = 'Report/OUTPUT_PART2'
TOP_3_FILE = os.path.join(OUTPUT_FOLDER, 'top_3.txt')
STATS_SUMMARY_FILE = os.path.join(OUTPUT_FOLDER, 'results2.csv')
PLOTS_FOLDER = os.path.join(OUTPUT_FOLDER, 'plots')


ATTACKING_STATS = [
    'Performance: goals',
    'Performance: assists',
    'Shooting: Standard: SoT/90'
]

DEFENSIVE_STATS = [
    'Defensive Actions: Tackles: TklW',
    'Defensive Actions: Blocks: Int',
    'Miscellaneous: Aerial Duels: Won%'
]

SELECTED_STATS = ATTACKING_STATS + DEFENSIVE_STATS

EXCLUDED_COLUMNS_FROM_TOP3 = [
    'Player',
    'Nation',
    'Team',
    'Position',
    'Age'
]

NEGATIVE_STATS = [
    'Performance: yellow cards',
    'Performance: red cards',
    'Goalkeeping: Performance: GA90',
    'Defensive Actions: Challenges: Lost',
    'Possession: Carries: Mis',
    'Possession: Carries: Dis',
    'Possession: Take-Ons: Tkld%',
    'Miscellaneous: Performance: Fls',
    'Miscellaneous: Performance: Off',
    'Miscellaneous: Aerial Duels: Lost',
]

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)