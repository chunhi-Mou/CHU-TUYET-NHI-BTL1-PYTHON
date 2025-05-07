import os

OUTPUT_FOLDER = 'Report/OUTPUT/OUTPUT_PART1'
OUTPUT_FILENAME = os.path.join(OUTPUT_FOLDER, 'results.csv')
WAIT_TIME = 15 # seconds

URL_CONFIG = {
    'standard': ('https://fbref.com/en/comps/9/stats/Premier-League-Stats', '#stats_standard'),
    'keeper': ('https://fbref.com/en/comps/9/keepers/Premier-League-Stats', '#stats_keeper'),
    'shooting': ('https://fbref.com/en/comps/9/shooting/Premier-League-Stats', '#stats_shooting'),
    'passing': ('https://fbref.com/en/comps/9/passing/Premier-League-Stats', '#stats_passing'),
    'gca': ('https://fbref.com/en/comps/9/gca/Premier-League-Stats', '#stats_gca'),
    'defense': ('https://fbref.com/en/comps/9/defense/Premier-League-Stats', '#stats_defense'),
    'possession': ('https://fbref.com/en/comps/9/possession/Premier-League-Stats', '#stats_possession'),
    'misc': ('https://fbref.com/en/comps/9/misc/Premier-League-Stats', '#stats_misc'),
}

# Mapping: csv-col-name : data-stat
STATS_MAP = {
    # Basic Info
    'Player': 'player',
    'Nation': 'nationality',
    'Team': 'team',
    'Position': 'position',
    'Age': 'age',

    # Playing Time
    'Playing Time: matches played': 'games',
    'Playing Time: starts': 'games_starts',
    'Playing Time: minutes': 'minutes',

    # Performance
    'Performance: goals': 'goals',
    'Performance: assists': 'assists',
    'Performance: yellow cards': 'cards_yellow',
    'Performance: red cards': 'cards_red',

    # Expected
    'Expected: xG': 'xg',
    'Expected: xAG': 'xg_assist',

    # Progression
    'Progression: PrgC': 'progressive_carries',
    'Progression: PrgP': 'progressive_passes',
    'Progression: PrgR': 'progressive_passes_received',

    # Per 90 minutes
    'Per 90 minutes: Gls': 'goals_per90',
    'Per 90 minutes: Ast': 'assists_per90',
    'Per 90 minutes: xG': 'xg_per90',
    'Per 90 minutes: xAG': 'xg_assist_per90',

    # Goalkeeping
    # Performance
    'Goalkeeping: Performance: GA90': 'gk_goals_against_per90',
    'Goalkeeping: Performance: Save%': 'gk_save_pct',
    'Goalkeeping: Performance: CS%': 'gk_clean_sheets_pct',
    # Penalty Kicks
    'Goalkeeping: Penalty Kicks: Save%': 'gk_pens_save_pct',

    # Shooting
    'Shooting: Standard: SoT%': 'shots_on_target_pct',
    'Shooting: Standard: SoT/90': 'shots_on_target_per90',
    'Shooting: Standard: G/Sh': 'goals_per_shot',
    'Shooting: Standard: Dist': 'average_shot_distance',

    # Passing
    # Total
    'Passing: Total: Cmp': 'passes_completed',
    'Passing: Total: Cmp%': 'passes_pct',
    'Passing: Total: TotDist': 'passes_total_distance',
    # Short
    'Passing: Short: Cmp%': 'passes_pct_short',
    'Passing: Medium: Cmp%': 'passes_pct_medium',
    'Passing: Long: Cmp%': 'passes_pct_long',
    # Expected
    'Passing: Expected: KP': 'assisted_shots',
    'Passing: Expected: 1/3': 'passes_into_final_third',
    'Passing: Expected: PPA': 'passes_into_penalty_area',
    'Passing: Expected: CrsPA': 'crosses_into_penalty_area',
    'Passing: Expected: PrgP': 'progressive_passes',

    # Goal and Shot Creation
    # SCA
    'Goal and Shot Creation: SCA: SCA': 'sca',
    'Goal and Shot Creation: SCA: SCA90': 'sca_per90',
    # GCA
    'Goal and Shot Creation: GCA: GCA': 'gca',
    'Goal and Shot Creation: GCA: GCA90': 'gca_per90',

    # Defensive Actions
    # Tackles
    'Defensive Actions: Tackles: Tkl': 'tackles',
    'Defensive Actions: Tackles: TklW': 'tackles_won',
    # Challenges
    'Defensive Actions: Challenges: Att': 'challenges',
    'Defensive Actions: Challenges: Lost': 'challenges_lost',
    # Blocks
    'Defensive Actions: Blocks: Blocks': 'blocks',
    'Defensive Actions: Blocks: Sh': 'blocked_shots',
    'Defensive Actions: Blocks: Pass': 'blocked_passes',
    'Defensive Actions: Blocks: Int': 'interceptions',

    # Possession
    # Touches
    'Possession: Touches: Touches': 'touches',
    'Possession: Touches: Def Pen': 'touches_def_pen_area',
    'Possession: Touches: Def 3rd': 'touches_def_3rd',
    'Possession: Touches: Mid 3rd': 'touches_mid_3rd',
    'Possession: Touches: Att 3rd': 'touches_att_3rd',
    'Possession: Touches: Att Pen': 'touches_att_pen_area',
    # Take-Ons
    'Possession: Take-Ons: Att': 'take_ons',
    'Possession: Take-Ons: Succ%': 'take_ons_won_pct',
    'Possession: Take-Ons: Tkld%': 'take_ons_tackled_pct',
    # Carries
    'Possession: Carries: Carries': 'carries',
    'Possession: Carries: PrgDist': 'carries_progressive_distance',
    'Possession: Carries: PrgC': 'progressive_carries',
    'Possession: Carries: 1/3': 'carries_into_final_third',
    'Possession: Carries: CPA': 'carries_into_penalty_area',
    'Possession: Carries: Mis': 'miscontrols',
    'Possession: Carries: Dis': 'dispossessed',
    #Receiving
    'Possession: Receiving: Rec': 'passes_received',
    'Possession: Receiving: PrgR': 'dispossessed',
    
    # Miscellaneous
    # Performance
    'Miscellaneous: Performance: Fls': 'fouls',
    'Miscellaneous: Performance: Fld': 'fouled',
    'Miscellaneous: Performance: Off': 'offsides',
    'Miscellaneous: Performance: Crs': 'crosses',
    'Miscellaneous: Performance: Recov': 'ball_recoveries',
    # Aerial Duels
    'Miscellaneous: Aerial Duels: Won': 'aerials_won',
    'Miscellaneous: Aerial Duels: Lost': 'aerials_lost',
    'Miscellaneous: Aerial Duels: Won%': 'aerials_won_pct',
}