"""
Interpolate hourly forecast data to generate half-hourly forecasts.
It reads a CSV file containing hourly forecast data, interpolates to create half-hourly forecasts,
and saves the resulting data to a new CSV file. The interpolation is a simple average between
consecutive hourly forecasts.
"""

import pandas as pd

df_xg = pd.read_csv("../../data/full_predictions_cross_validation_v4_prob_full.csv")


# Generate half-hourly forecasts by interpolating between each hourly forecast
def generate_half_hourly_forecasts(df):
    base_hourly_columns = [
        col for col in df.columns if "Hour Forecast" in col and "p10" not in col and "p90" not in col
    ]
    p10_hourly_columns = [col for col in df.columns if "p10" in col]
    p90_hourly_columns = [col for col in df.columns if "p90" in col]
    interpolated_columns = []

    for i in range(len(base_hourly_columns) - 1):
        # Base forecast interpolation
        current_hour = base_hourly_columns[i]
        next_hour = base_hourly_columns[i + 1]
        half_hour_column = f"{i + 0.5} Hour Forecast"
        df[half_hour_column] = (df[current_hour] + df[next_hour]) / 2

        # p10 forecast interpolation
        current_hour_p10 = p10_hourly_columns[i]
        next_hour_p10 = p10_hourly_columns[i + 1]
        half_hour_column_p10 = f"p10 {i + 0.5} Hour Forecast"
        df[half_hour_column_p10] = (df[current_hour_p10] + df[next_hour_p10]) / 2

        # p90 forecast interpolation
        current_hour_p90 = p90_hourly_columns[i]
        next_hour_p90 = p90_hourly_columns[i + 1]
        half_hour_column_p90 = f"p90 {i + 0.5} Hour Forecast"
        df[half_hour_column_p90] = (df[current_hour_p90] + df[next_hour_p90]) / 2

        interpolated_columns.extend(
            [
                current_hour,
                half_hour_column,
                current_hour_p10,
                half_hour_column_p10,
                current_hour_p90,
                half_hour_column_p90,
            ]
        )

    # Include the last hour forecast column and its p10 and p90
    interpolated_columns.extend([base_hourly_columns[-1], p10_hourly_columns[-1], p90_hourly_columns[-1]])

    # Reorder columns to ensure 'Init Time' is at the beginning, followed by the base, p10, and p90 forecasts in chronological order
    def sort_key(col):
        if col == "Init Time":
            return -1  # Ensure 'Init Time' is always first
        parts = col.split()
        hour = float(parts[0]) if parts[0].replace(".", "", 1).isdigit() else float("inf")
        forecast_type = 0 if len(parts) == 2 else (1 if "p10" in col else 2)
        return (hour, forecast_type)

    ordered_columns = sorted(interpolated_columns, key=sort_key)
    ordered_columns.insert(0, "Init Time")  # Insert 'Init Time' at the beginning after sorting
    return df.loc[:, ordered_columns]


print("Copying the original DataFrame...")
df_xg_temp = df_xg.copy()
print("Generating half-hourly forecasts...")
df_xg_30 = generate_half_hourly_forecasts(df_xg_temp)
print("Saving the interpolated forecasts to CSV...")
df_xg_30.to_csv(
    "../../data/full_predictions_cross_validation_v8_(from_v4)_prob_full_formated_xg.csv",
    index=False,
)
print("Interpolated forecasts saved successfully.")
