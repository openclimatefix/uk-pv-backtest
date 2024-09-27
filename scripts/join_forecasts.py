"""Join forecasts based on differing init times"""

import pandas as pd

df_1 = pd.read_csv(
    "../data/pvnet_2019-2023_backtest_300724/final/forecast_v=9a_model_name_1=pvnet_app_v__model_version_1=2.3.19__start_date=2019-01-01__end_date=2023-01-01.csv.gz"
)
df_2 = pd.read_csv(
    "../data/pvnet_2023-2024_backtest_230924/forecast_v=9b__model_name_1=pvnet_app_v__model_version_1=2.3.19__start_date=2023-01-01__end_date=2024-01-01.csv.gz"
)


# Check for overlapping values in "forecasting_creation_datetime_utc" columns
# Using set, which automatically removes duplicates
df_1_dates = set(df_1["forecasting_creation_datetime_utc"])
df_2_dates = set(df_2["forecasting_creation_datetime_utc"])

overlapping_dates = df_1_dates.intersection(df_2_dates)

if overlapping_dates:
    print("Warning: Overlapping dates found in 'forecasting_creation_datetime_utc' columns:")
    print(overlapping_dates)
    raise ValueError("Overlapping dates found. Please check your input data.")
else:
    print("No overlapping dates found. Proceeding with concatenation.")

df_combined = pd.concat([df_1, df_2], ignore_index=True)

# Sort the combined DataFrame by 'forecasting_creation_datetime_utc' in ascending order
df_combined = df_combined.sort_values("forecasting_creation_datetime_utc", ascending=True)

# Reset the index after sorting
df_combined = df_combined.reset_index(drop=True)

df_combined.to_csv(
    "../data/pvnet_2019-2024_backtest_230924/pvnet_2019-2024_240924_combined.csv.gz",
    index=False,
    compression="gzip",
)
