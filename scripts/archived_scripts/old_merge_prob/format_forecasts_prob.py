"""
Script formats the forecasts

The inputs csv file should have the following columns
- Init Time
- 0 Hour Forecast
- 1 Hour Forecast
....
- 40 Hour Forecast
The values are normalized by pvlive installed capacity mw, so need to un normalize these

The output csv file will have the following columns:
- start_datetime_utc: datetime - the start datetime of the period
- end_datetime_utc: datetime - the end datetime of the period
- forecasting_creation_datetime_utc: datetime, when the forecast is made
- generation_mw: float - the solar generation value
- generation_mw_p10: float - the 10th percentile of the solar generation value
- generation_mw_p90: float - the 90th percentile of the solar generation value
"""

import pandas as pd

# load data
print("Loading Forecast data")
data = pd.read_csv("../data/full_predictions_cross_validation_v6_prob_full.csv")

# join with pvlive capacity data
print("Loading PVLive data")
pvlive = pd.read_csv("../pvlive_2016_2023.csv")
pvlive["end_datetime_utc"] = pd.to_datetime(pvlive["end_datetime_utc"])

# unnormalize data
print("Unnormalizing data")
data["Init Time"] = pd.to_datetime(data["Init Time"])
data["Init Time"] = data["Init Time"].dt.tz_localize("UTC")
data = data.merge(pvlive, left_on="Init Time", right_on="end_datetime_utc")
for c in data.columns:
    if "Hour Forecast" in c:
        data[c] = data[c].astype(float)

        # Night limit required to avoid p90 generation values at night
        idx_night = data[c] <= 0.000244
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

# rename columns
print("Renaming columns")
new_cols = {c: c.strip(" Hour Forecast") for c in data.columns if "Hour Forecast" in c}
data.rename(columns=new_cols, inplace=True)

print("Stacking rows")
data_stack = pd.melt(data, id_vars=["Init Time", "generation_mw"], var_name="combined")

# Identify forecast type
data_stack["forecast"] = data_stack["combined"].str.extract("(p10|p90)")
data_stack["forecast"].fillna("pred", inplace=True)

is_pred = data_stack["forecast"] == "pred"
data_stack.loc[is_pred, "horizon"] = data_stack["combined"].str.extract("(\d+)")[0]
data_stack.loc[~is_pred, "horizon"] = data_stack["combined"].str.replace("p10|p90", "")


# Drop the original combined column
data_stack.drop(columns=["combined"], inplace=True)

data_stack["horizon"] = data_stack["horizon"].astype(int)
data_stack = data_stack.sort_values(by=["Init Time", "horizon"])

data_stack.drop(columns=["generation_mw"], inplace=True)
data_stack.rename(
    columns={
        "Init Time": "forecasting_creation_datetime_utc",
        "value": "generation_mw",
    },
    inplace=True,
)

# Create start and end time
data_stack["start_datetime_utc"] = data_stack["forecasting_creation_datetime_utc"] + pd.to_timedelta(
    data_stack["horizon"], "h"
)

data_stack["end_datetime_utc"] = data_stack["start_datetime_utc"] + pd.Timedelta(minutes=30)
data_stack.drop(columns=["horizon"], inplace=True)

# Pivot the DataFrame
pivot_df = data_stack.pivot_table(
    index=[
        "forecasting_creation_datetime_utc",
        "start_datetime_utc",
        "end_datetime_utc",
    ],
    columns="forecast",
    values="generation_mw",
    aggfunc="first",
)

# Rename the columns if necessary
pivot_df.rename(
    columns={
        "pred": "generation_mw",
        "p10": "generation_mw_p10",
        "p90": "generation_mw_p90",
    },
    inplace=True,
)

# Reset index to flatten the DataFrame
pivot_df.reset_index(inplace=True)

print("Save to csv")

print(pivot_df.head())

pivot_df.to_csv("../data/formatted_forecasts_v6_prob.csv.gz", index=False, compression="gzip")
