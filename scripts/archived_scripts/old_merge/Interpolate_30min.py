"""
Script designed to interpolate hourly forecast data to generate half-hourly forecasts.
It reads a CSV file containing hourly forecast data, interpolates to create half-hourly forecasts,
and saves the resulting data to a new CSV file. The interpolation is a simple average between
consecutive hourly forecasts.
"""

import pandas as pd

df_xg = pd.read_csv("../../data/full_predictions_cross_validation_v4_without_prob_full.csv")


# Generate half-hourly forecasts by interpolating between each hourly forecast
def generate_half_hourly_forecasts(df):
    hourly_columns = [col for col in df.columns if "Hour Forecast" in col]
    interpolated_columns = []
    for i in range(len(hourly_columns) - 1):
        current_hour = hourly_columns[i]
        next_hour = hourly_columns[i + 1]
        half_hour_column = f"{i + 0.5} Hour Forecast"
        df[half_hour_column] = (df[current_hour] + df[next_hour]) / 2
        interpolated_columns.extend([current_hour, half_hour_column])

    # Include the last hour forecast column
    interpolated_columns.append(hourly_columns[-1])

    # Reorder columns to include 'Init Time' at the beginning and maintain chronological order of forecasts
    ordered_columns = ["Init Time"] + interpolated_columns
    return df.loc[:, ordered_columns]


df_xg_temp = df_xg.copy()
df_xg_30 = generate_half_hourly_forecasts(df_xg_temp)
df_xg_30.to_csv(
    "../../data/full_predictions_cross_validation_v4_without_prob_with_30min.csv",
    index=False,
)
