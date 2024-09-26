"""
Merge forecast data from the Intraday and Dayahead PVNet models to create a combined forecast dataset.
The script performs the following key operations:

- Merges the data from both models based on the 'Init Time' column.
- Blends the forecasts for specific forecast horizons based on predefined ratios.
- Filters the combined dataset to include data points from a specific range.
"""

import pandas as pd

data_pvnet_ID = pd.read_csv("../../data/pvnet_2023-2024_backtest_230924/pvnet_2023-2024_ID_240924_a4.csv")
data_pvnet_DA = pd.read_csv("../../data/pvnet_2023-2024_backtest_230924/pvnet_2023-2024_DA_240924_a4.csv")

blend = True

data_pvnet_ID["Init Time"] = pd.to_datetime(data_pvnet_ID["Init Time"], utc=True)
data_pvnet_DA["Init Time"] = pd.to_datetime(data_pvnet_DA["Init Time"], utc=True)


data_combined = pd.merge(data_pvnet_DA, data_pvnet_ID, on="Init Time", how="left", suffixes=("", "_ID"))
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

for col in data_pvnet_ID.columns:
    if col in data_pvnet_DA.columns and col != "Init Time":
        if blend and col in blend_ratios.keys():
            pvnet_ratio, xgb_ratio = blend_ratios[col]
            mask = ~pd.isna(data_combined[col + "_ID"])
            data_combined.loc[mask, col] = (data_combined.loc[mask, col + "_ID"] * pvnet_ratio) + (
                data_combined.loc[mask, col] * xgb_ratio
            )
        else:
            mask = ~pd.isna(data_combined[col + "_ID"])
            data_combined.loc[mask, col] = data_combined.loc[mask, col + "_ID"]

        data_combined.drop(columns=[col + "_ID"], inplace=True)

data_combined = data_combined[data_combined["Init Time"] >= "2019-01-01 00:00:00+00:00"]
data_combined = data_combined[data_combined["Init Time"] < "2024-01-01 00:00:00+00:00"]


data_combined_shift = data_combined.copy()
# Use the forecast from PVnet 30 mins ahead
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


print("Saving dataset")
data_combined_shift.to_csv("../../data/pvnet_2023-2024_backtest_230924/pvnet_2023-2024_240924_a4.csv.gz", index=False)
print("Done")
