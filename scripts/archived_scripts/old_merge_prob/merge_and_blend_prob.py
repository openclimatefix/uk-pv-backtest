"""
Script designed to merge and blend forecast data from two different models, XGBoost (XG) and PVNet,
to create a combined forecast dataset. The blending process is optional and can be controlled through a parameter.
The script performs several key operations:

- It first converts the 'Init Time' column in both datasets to datetime format, ensuring that the data is aligned
  on the same temporal scale for accurate merging.
- The data from both models is then merged based on the 'Init Time' column. This operation combines the forecasts
  from both models into a single dataset, with PVNet's forecasts being appended as new columns with a suffix to
  distinguish them from XGBoost's forecasts.
- If blending is enabled, the script calculates blended forecasts for specific forecast horizons based on predefined
  blending ratios. These ratios determine the weight of each model's forecast in the final blended output.
- For columns present in both datasets (excluding 'Init Time'), the script either blends the forecasts (if blending
  is enabled and the column is eligible for blending) or replaces XGBoost's forecasts with PVNet's forecasts.
- The combined dataset is then filtered to include only data points starting from January 1, 2020, up to August 8, 2022.
- A copy of the combined dataset is created to adjust the '0 Hour Forecast' by using PVNet's '0.5 Hour Forecast' for
  the next initiation time, compensating for XGBoost's lack of a '0 Hour Forecast'.
- The script also prepares a separate dataset containing XGBoost's forecasts prior to January 1, 2020.
- Finally, the pre-2020 XGBoost dataset and the adjusted combined dataset are concatenated to form the final merged
  and blended dataset, which is then returned.
"""

import pandas as pd

# Two step merging process for the datasets
data_xg = pd.read_csv("../../data/full_predictions_cross_validation_v8_(from_v4)_prob_full_formated_xg_unnorm.csv")


def format_column_headers_directly(df):
    new_columns = []
    for col in df.columns:
        # Check for the specific patterns and format accordingly
        if "Hour Forecast" in col:
            parts = col.split()
            # Identify the part of the column name that represents the hour value
            for i, part in enumerate(parts):
                if part.replace(".", "", 1).isdigit() and "hour" in parts[i + 1].lower():
                    # Convert the hour value to float and format to one decimal place
                    formatted_hour = f"{float(part):.1f}"
                    parts[i] = formatted_hour
                    break
            new_col = " ".join(parts)
            new_columns.append(new_col)
        else:
            new_columns.append(col)
    df.columns = new_columns
    return df


# Apply the function to your DataFrame
data_xg = format_column_headers_directly(data_xg)

data_pvnet = pd.read_csv("../../data/pvnet_sum_model_combined_270324.csv")

blend = True


data_xg["Init Time"] = pd.to_datetime(data_xg["Init Time"], utc=True)
data_pvnet["Init Time"] = pd.to_datetime(data_pvnet["Init Time"], utc=True)
# Convert whole integer headers in xg to .0
data_combined = pd.merge(data_xg, data_pvnet, on="Init Time", how="left", suffixes=("", "_pvnet"))
data_combined

print(data_combined.isna().sum())

blend_ratios = {
    "7.0 Hour Forecast": (0.75, 0.25),
    "7.5 Hour Forecast": (0.5, 0.5),
    "8.0 Hour Forecast": (0.25, 0.75),
    "p90 7.0 Hour Forecast": (0.75, 0.25),
    "p90 7.5 Hour Forecast": (0.5, 0.5),
    "p90 8.0 Hour Forecast": (0.25, 0.75),
    "p10 7.0 Hour Forecast": (0.75, 0.25),
    "p10 7.5 Hour Forecast": (0.5, 0.5),
    "p10 8.0 Hour Forecast": (0.25, 0.75),
}

for col in data_pvnet.columns:
    if col in data_xg.columns and col != "Init Time":
        if blend and col in blend_ratios.keys():
            pvnet_ratio, xgb_ratio = blend_ratios[col]
            mask = ~pd.isna(data_combined[col + "_pvnet"])
            data_combined.loc[mask, col] = (data_combined.loc[mask, col + "_pvnet"] * pvnet_ratio) + (
                data_combined.loc[mask, col] * xgb_ratio
            )
        else:
            mask = ~pd.isna(data_combined[col + "_pvnet"])
            data_combined.loc[mask, col] = data_combined.loc[mask, col + "_pvnet"]
        data_combined.drop(columns=[col + "_pvnet"], inplace=True)

data_combined = data_combined[data_combined["Init Time"] >= "2018-01-01 03:00:00+00:00"]
data_combined = data_combined[data_combined["Init Time"] <= "2022-08-08 08:00:00+00:00"]


data_combined_shift = data_combined.copy()
# As XGb makes a 0 hour forecast but PVNet does not, its better to use the forecast from PVnet 30 mins ahead
# for the successive init time, 30 mins later.
data_combined_shift.loc[data_combined_shift["0.5 Hour Forecast"].shift(1).notna(), "0.0 Hour Forecast"] = (
    data_combined_shift["0.5 Hour Forecast"].shift(1)
)

data_combined_shift.loc[data_combined_shift["p10 0.5 Hour Forecast"].shift(1).notna(), "p10 0.0 Hour Forecast"] = (
    data_combined_shift["p10 0.5 Hour Forecast"].shift(1)
)

data_combined_shift.loc[data_combined_shift["p90 0.5 Hour Forecast"].shift(1).notna(), "p90 0.0 Hour Forecast"] = (
    data_combined_shift["p90 0.5 Hour Forecast"].shift(1)
)
# Now to create the other bit of the dataset
xgb_p2 = data_xg.copy()
# Needs to be less than but not including this date
xgb_p2 = xgb_p2[xgb_p2["Init Time"] < "2018-01-01 03:00:00+00:00"]
merged_data = pd.DataFrame()
merged_data = pd.concat([xgb_p2, data_combined_shift])


print("Merging")
# merged_data = blend_data(data_xg, data_pvnet, blend=False)
merged_data.to_csv("../../data/full_pred_v8_5_xgb_pvnet_with_blend_p10p90_0hourfix_v3.csv.gz", index=False)
print("Save")
