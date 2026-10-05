import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Clean column names and date reformatting
def clean_data(
        data, 
        timestamp_col='Timestamp', 
        format='%d/%m/%Y %I:%M:%S %p', 
        target_col='Normal/Attack'):
    
    df = data.copy()

    # Reformat time for convenience
    df[timestamp_col] = df[timestamp_col].str.strip()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], format=format)

    # Encode target feature
    df[target_col] = (df[target_col]
                      .str.replace(" ","", regex=False)
                      .map({'Normal':0, 'Attack':1}))

    return df


# Fill missing timestamps from the sequence
def fill_missing_timestamps(
        data, actuators, sensors, target,
        timestamp_col='Timestamp', freq='s'):
    
    df = data.copy()

    # Get the missing timestamps within the range of min and max
    full_range = pd.date_range(
        start=df[timestamp_col].min(),
        end=df[timestamp_col].max(),
        freq=freq)

    missing_timestamps = full_range[~full_range.isin(df[timestamp_col])]
    print(f"Missing timestamp: {len(missing_timestamps)}")
    print(f"Dataset length: {len(df)}")

    df_filled = df.set_index(timestamp_col).reindex(full_range).rename_axis(timestamp_col).reset_index()

    df_filled[actuators] = df_filled[actuators].fillna(0)
    df_filled[sensors] = df_filled[sensors].interpolate(method='linear', limit_direction="both")
    df_filled[target] = df_filled[target].fillna(0)

    print(f"Dataset length after filling: {len(df_filled)}")
    print(f"Missing value remaining: {df_filled.isna().sum().sum()}")

    return  df_filled


# Filter out the initial operating state of the system
def filter_operating_state(data, target, hours):
    df = data.copy()

    samples_to_drop = 60 * 60 * hours
    df_filtered = df.loc[samples_to_drop:].reset_index(drop=True)

    print(f"Initial Length: {df.shape[0]}")
    print(f"Dataset after removed:  {df_filtered.shape[0]}")

    return df_filtered


