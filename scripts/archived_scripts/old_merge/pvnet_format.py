"""
Script to process the output from the PVNet model to format it for merging with XGBoost predictions. It performs the following steps:

1. Opens the PVNet dataset using xarray.
2. Filters the dataset to keep only the GSP ID of 0 and the expected generation output.
3. Converts the filtered dataset to a pandas DataFrame for easier manipulation.
4. Pivots the DataFrame to have forecast initialization times as rows and forecast horizons as columns.
5. Renames the columns to match the format used in the XGBoost DataFrame, which uses "Hour Forecast" naming.
6. Adjusts the column names to ensure whole hour values are integers instead of having a decimal place.
7. Resets the index to bring the forecast initialization time back as a column and renames it to "Init Time".
8. Saves the formatted DataFrame to a CSV file for further processing.
"""

import xarray as xr

pvnet_ds = xr.open_dataset("../../data/model_ensemble.nc")

# just want to keep the gsp id of 0 and just the expected generation
filtered_pvnet_ds = pvnet_ds.sel(gsp_id=0, output_label=[b"forecast_mw"])

# Convert the filtered dataset to pandas DataFrame for easier manipulation
filtered_pvnet_df = filtered_pvnet_ds.to_dataframe().reset_index()

# Transforming filtered_pvnet_df to match the format of xgb_df
# First, we need to pivot the dataframe to have forecast_init_time as rows and horizon_mins as columns
pvnet_pivot_df = filtered_pvnet_df.pivot(index="forecast_init_time", columns="horizon_mins", values="hindcast")

# Rename the columns to match the format in xgb_df, which uses "Hour Forecast" naming
pvnet_pivot_df.columns = [f"{(col/60)} Hour Forecast" for col in pvnet_pivot_df.columns]

# Ensure whole hour values are integers instead of 1 decimal place
pvnet_pivot_df.columns = [
    f"{int(col)} Hour Forecast" if col.is_integer() else f"{col} Hour Forecast"
    for col in (pvnet_pivot_df.columns.str.replace(" Hour Forecast", "").astype(float))
]

# Reset the index to bring forecast_init_time back as a column and rename it to "Init Time"
pvnet_pivot_df.reset_index(inplace=True)
pvnet_pivot_df.rename(columns={"forecast_init_time": "Init Time"}, inplace=True)

pvnet_pivot_df.to_csv("../../data/pvnet_predicitons_2021-2023_preformat_v2.csv", index=False)
