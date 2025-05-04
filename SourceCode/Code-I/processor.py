import pandas as pd

def process_data(raw_data: dict, final_columns: list) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(raw_data, orient='index')
    # Filter by Minutes Played (> 90)
    minutes_column_name = 'Playing Time: minutes'
    if minutes_column_name in df.columns:
        df['Min_numeric'] = pd.to_numeric(df[minutes_column_name].astype(str).str.replace(',', '', regex=False), errors='coerce')
        df = df[df['Min_numeric'] > 90].copy()
        df.drop(columns=['Min_numeric'], inplace=True)
        print(f"Cleaned to {len(df)} players > 90 min")
    else:
        print("Warn: 'Min' col not found")
    df = df[final_columns]
    df.fillna('N/a', inplace=True)
    # Sort by Player's First Name
    if 'Player' in df.columns:
        df.sort_values(by='Player', key=lambda x: x.str.split().str[0].str.lower(), inplace=True)
        print("Data sorted by Player's first name")
    else:
        print("Warn: NOT FOUND 'Player' col")
    return df