"""
Unnormalise the forecast using PVLive capacity. The process involves several key steps:

- Load forecast and PVLive capacity data
- Merge datasets based on datetime
- Unnormalise forecasts using installed capacity
- Clean up data by removing negative values and unnecessary columns
- Save unnormalised forecast to CSV
"""

import pandas as pd

# load data
print("Loading Forecast data")
data = pd.read_csv("../../data/full_predictions_cross_validation_v8_(from_v4)_prob_full_formated_xg.csv")

# join with pvlive capacity data
print("Loading PVLive data")
pvlive = pd.read_csv("../../pvlive_2016_2023.csv")
pvlive["end_datetime_utc"] = pd.to_datetime(pvlive["end_datetime_utc"])

# unnormalise data
print("Unnormalising data")
data["Init Time"] = pd.to_datetime(data["Init Time"])
data["Init Time"] = data["Init Time"].dt.tz_localize("UTC")
data = data.merge(pvlive, left_on="Init Time", right_on="end_datetime_utc")
for c in data.columns:
    if "Hour Forecast" in c:
        data[c] = data[c].astype(float)
        idx_night = data[c] <= 0.000234
        data.loc[idx_night, c] = 0
        data[c] = data[c] * data["installedcapacity_mwp"]

# drop columns
data.drop(
    columns=[
        "installedcapacity_mwp",
        "capacity_mwp",
        "start_datetime_utc",
        "end_datetime_utc",
    ],
    inplace=True,
)

data.to_csv(
    "../../data/full_predictions_cross_validation_v8_(from_v4)_prob_full_formated_xg_unnorm.csv",
    index=False,
)
