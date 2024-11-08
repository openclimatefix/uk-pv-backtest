"""
Script formats the forecasts

The inputs csv file should have the following columns
- Init Time
- 0 Hour Forecast
- 1 Hour Forecast
....
- 40 Hour Forecast
The values are normalized by pvlive installed capacity mw, so need to un normalize this


The output csv file will have the following columns:
- start_datetime_utc: datetime - the start datetime of the period
- end_datetime_utc: datetime - the end datetime of the period
- forecasting_creation_datetime_utc: datetime, when the forecast is made
- generation_mw: float - the solar generation value
"""

import pandas as pd
import plotly.graph_objs as go

# load data
print("Loading Forecast data")
data = pd.read_csv("../data/full_predictions_cross_validation_v4_without_prob_full.csv")

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

# rename columns
print("Renaming columns")
new_cols = {c: int(c.strip(" Hour Forecast")) for c in data.columns if "Hour Forecast" in c}
data.rename(columns=new_cols, inplace=True)

# stack rows
print("Stacking rows")
data_stack = pd.melt(data, id_vars=["Init Time", "generation_mw"])
data_stack = data_stack.sort_values(by=["Init Time", "variable"])

# for plotting
for horizon in [0, 1, 2, 4, 8, 12, 24, 36]:
    d = data_stack[data_stack["variable"] == horizon]
    mae = (d["generation_mw"] - d["value"].shift(horizon * 2)).abs().mean()
    print(f"MAE: {mae:.2f} MW for {horizon}")

go.Figure(
    data=[
        go.Scatter(x=d["Init Time"], y=d["generation_mw"], name="generation_mw"),
        go.Scatter(x=d["Init Time"], y=d["value"].shift(horizon * 2), name="value"),
    ]
).show()


# format rows
print("Formatting rows")
data_stack["variable"] = data_stack["variable"].astype(int)
data_stack.drop(columns=["generation_mw"], inplace=True)
data_stack.rename(
    columns={
        "Init Time": "forecasting_creation_datetime_utc",
        "value": "generation_mw",
    },
    inplace=True,
)

# create start and end time
data_stack["end_datetime_utc"] = data_stack["forecasting_creation_datetime_utc"] + pd.to_timedelta(
    data_stack["variable"], "h"
)

data_stack["start_datetime_utc"] = data_stack["end_datetime_utc"] - pd.Timedelta(minutes=30)
data_stack.drop(columns=["variable"], inplace=True)

print("Save to csv")
data_stack.to_csv("../data/formatted_forecasts_v4.csv.gz", index=False, compression="gzip")
