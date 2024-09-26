"""
Process the output from the PVNet model to format it for merging with XGBoost predictions. It performs the following steps:

1. Opens the PVNet dataset using xarray.
2. Filters the dataset to keep only the GSP ID of 0 and the expected generation output.
3. Converts the filtered dataset to a pandas DataFrame for easier manipulation.
4. Pivots the DataFrame to have forecast initialization times as rows and forecast horizons as columns.
5. Renames the columns to match the format used in the XGBoost forecasts, which uses "Hour Forecast" naming.
6. Adjusts the column names to ensure whole hour values are integers instead of having a decimal place.
7. Resets the index to bring the forecast initialization time back as a column and renames it to "Init Time".
8. Saves the formatted DataFrame to a CSV file for further processing.
"""

import xarray as xr

print("Opening PVNet dataset...")
pvnet_ds = xr.open_zarr("../../data/pvnet_2023-2024_backtest_230924/pvnet_2023-2024_DA_240924_a4.zarr")
print("PVNet dataset opened successfully.")

# Check if 'forecast_init_time' exists, if not, rename 'init_time_utc' to 'forecast_init_time'
if "forecast_init_time" not in pvnet_ds.dims and "init_time_utc" in pvnet_ds.dims:
    print("Renaming 'init_time_utc' to 'forecast_init_time'...")
    pvnet_ds = pvnet_ds.rename({"init_time_utc": "forecast_init_time"})
    print("Renaming completed.")

# Keep the GSP ID of 0 and include the expected generation, plevel10, and plevel90
print("Filtering dataset for GSP ID 0 and required output labels...")
filtered_pvnet_ds = pvnet_ds.sel(
    gsp_id=0, output_label=["forecast_mw", "forecast_mw_plevel_10", "forecast_mw_plevel_90"]
)
print("Filtering completed.")

# Convert the filtered dataset to pandas DataFrame for easier manipulation
print("Converting filtered dataset to pandas DataFrame...")
filtered_pvnet_df = filtered_pvnet_ds.to_dataframe().reset_index()
print("Conversion to DataFrame completed.")

# Transforming filtered_pvnet_df to match the format of xgb_df
# Pivot the dataframe to have forecast_init_time as rows and horizon_mins as columns for each output_label
print("Pivoting DataFrame...")
pvnet_pivot_df = filtered_pvnet_df.pivot_table(
    index="forecast_init_time", columns=["output_label", "horizon_mins"], values="hindcast", aggfunc="first"
).reset_index()
print("Pivoting completed.")

# Flatten the MultiIndex columns and format them as instructed
print("Flattening and formatting columns...")
pvnet_pivot_df.columns = [
    " ".join(map(str, col)).strip() if col[1] else col[0] for col in pvnet_pivot_df.columns.values
]

# Rename the columns to match the desired format, including "Hour Forecast" naming, and prefixing with p10 or p90 where applicable
print("Renaming columns...")
new_columns = {}
for col in pvnet_pivot_df.columns:
    parts = col.split()
    if parts[0] == "forecast_init_time":
        new_columns[col] = "Init Time"
    else:
        label = ""
        if "plevel_10" in parts[0]:
            label = "p10 "
        elif "plevel_90" in parts[0]:
            label = "p90 "
        hour = int(parts[1]) / 60 if (int(parts[1]) / 60).is_integer() else int(parts[1]) / 60
        hour_label = f"{hour} Hour Forecast" if hour != 0 else "0.5 Hour Forecast"
        new_columns[col] = f"{label}{hour_label}".strip()

pvnet_pivot_df.rename(columns=new_columns, inplace=True)
print("Column renaming completed.")

print("Saving DataFrame to CSV...")
pvnet_pivot_df.to_csv("../../data/pvnet_2023-2024_backtest_230924/pvnet_2023-2024_DA_240924_a4.csv", index=False)
print("CSV file saved successfully.")
