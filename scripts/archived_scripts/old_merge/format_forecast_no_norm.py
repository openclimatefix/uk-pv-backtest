"""
Script formats the forecasts, it expects the the data to be already normalised.

The inputs csv file should have the following columns
- Init Time
- 0 Hour Forecast
- 1 Hour Forecast
....
- 40 Hour Forecast

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
data = pd.read_csv("../../data/full_pred_v6_3_xgb_pvnet_blended.csv")


# drop previously generated capacity that i carried over...
data.drop(columns=["generation_mw"], inplace=True)

# join with pvlive capacity data
print("Loading PVLive data")
pvlive = pd.read_csv("/home/zak/projects/DRS/uk-nia-drs/pvlive_2016_2023.csv")
pvlive["end_datetime_utc"] = pd.to_datetime(pvlive["end_datetime_utc"])

data["Init Time"] = pd.to_datetime(data["Init Time"])
data = data.merge(pvlive, left_on="Init Time", right_on="end_datetime_utc")

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
new_cols = {c: float(c.strip(" Hour Forecast")) for c in data.columns if "Hour Forecast" in c}
data.rename(columns=new_cols, inplace=True)

# stack rows
print("Stacking rows")
data_stack = pd.melt(data, id_vars=["Init Time", "generation_mw"])
data_stack = data_stack.sort_values(by=["Init Time", "variable"])

# for plotting
# for horizon in [0, 1, 2, 4, 8, 12, 24, 36]:
for horizon in [0, 1, 2, 4, 8]:
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
data_stack["variable"] = data_stack["variable"].astype(float)
data_stack.drop(columns=["generation_mw"], inplace=True)
data_stack.rename(
    columns={
        "Init Time": "forecasting_creation_datetime_utc",
        "value": "generation_mw",
    },
    inplace=True,
)

# create start and end time handling non-integer hours
data_stack["end_datetime_utc"] = data_stack.apply(
    lambda row: row["forecasting_creation_datetime_utc"] + pd.to_timedelta(float(row["variable"]), unit="h"),
    axis=1,
)

data_stack["start_datetime_utc"] = data_stack["end_datetime_utc"] - pd.Timedelta(minutes=30)
data_stack.drop(columns=["variable"], inplace=True)

print("Save to csv")
data_stack.to_csv("../../data/full_pred_v6_3_xgb_pvnet_blend.csv.gz", index=False, compression="gzip")
