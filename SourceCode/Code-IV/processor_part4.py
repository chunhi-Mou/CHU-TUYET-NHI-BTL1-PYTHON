import pandas as pd
import os
from config_part4 import *

def get_valid_players_from_part1() -> set:
    try:
        df = pd.read_csv(PART1_RESULTS_FILE)
        if not all(col in df.columns for col in [PART1_MINUTES_COLUMN, PART1_PLAYER_COLUMN]):
            print(f"Missing cols in {PART1_RESULTS_FILE}")
            return set()

        df['minutes'] = pd.to_numeric(df[PART1_MINUTES_COLUMN].astype(str).str.replace(',', ''), errors='coerce')
        players = set(df.loc[df['minutes'] > MIN_MINUTES_THRESHOLD, PART1_PLAYER_COLUMN]
                      .dropna().astype(str).str.strip())
        players.discard('')
        print(f"{len(players)} valid players found (> {MIN_MINUTES_THRESHOLD} mins).")
        return players

    except FileNotFoundError:
        print(f"File not found: {PART1_RESULTS_FILE}")
    except Exception as e:
        print(f"Error: {e}")
    return set()

def process_transfer_data() -> pd.DataFrame | None:
    valid_players = get_valid_players_from_part1()
    if not valid_players:
        print("No valid players found")
        return None

    try:
        df = pd.read_csv(RAW_DATA_FILENAME)
        if df.empty or 'Player' not in df.columns:
            print(f"Invalid raw data: {'empty file' if df.empty else 'missing Player column'}.")
            return None

        df['Player'] = df['Player'].astype(str).str.strip()
        filtered = df[df['Player'].isin(valid_players)].copy()

        if filtered.empty:
            print("No matching players")
            return filtered

        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        filtered.to_csv(ESTIMATION_READY_DATA_FILENAME, index=False, encoding='utf-8-sig')
        print(f"Filtered data saved: {len(filtered)} rows -> {ESTIMATION_READY_DATA_FILENAME}")
        return filtered

    except FileNotFoundError:
        print(f"File not found: {RAW_DATA_FILENAME}")
    except Exception as e:
        print(f"Error in process_transfer_data(): {e}")
    return None
