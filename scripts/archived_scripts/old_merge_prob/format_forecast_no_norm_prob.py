# Formatting forecast no norm prob

import pandas as pd

# load data
print("Loading Forecast data")
data = pd.read_csv("../../data/filtered_pvnet_sum_model_2022_2023.csv")

data["Init Time"] = pd.to_datetime(data["Init Time"], utc=True)


# drop previously generated capacity that i carried over...
# data.drop(columns=["generation_mw"], inplace=True)

# join with pvlive capacity data
print("Loading PVLive data")
pvlive = pd.read_csv("../../pvlive_2016_2023.csv")
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
new_cols = {c: c.strip(" Hour Forecast") for c in data.columns if "Hour Forecast" in c}
data.rename(columns=new_cols, inplace=True)

# stack rows
print("Stacking rows")
data_stack = pd.melt(data, id_vars=["Init Time", "generation_mw"], var_name="combined")

# data_stack = data_stack.sort_values(by=["Init Time", "horizon"])


# for plotting
# for horizon in [0, 1, 2, 4, 8, 12, 24, 36]:
# for horizon in [0.0, 1.0, 2.0, 4.0, 8.0]:
#     d = data_stack[data_stack["variable"] == horizon]
#     mae = (d["generation_mw"] - d["value"].shift(horizon * 2)).abs().mean()
#     print(f"MAE: {mae:.2f} MW for {horizon}")

# go.Figure(
#     data=[
#         go.Scatter(x=d["Init Time"], y=d["generation_mw"], name="generation_mw"),
#         go.Scatter(x=d["Init Time"], y=d["value"].shift(horizon * 2), name="value"),
#     ]
# ).show()


# Identify forecast type
# data_stack["forecast"] = data_stack["combined"].apply(lambda x: "pred" if "p10" not in x and "p90" not in x else x.split()[0])
data_stack["forecast"] = data_stack["combined"].str.extract("(p10|p90)").fillna("pred")

print("Extract forecast type")

is_pred = data_stack["forecast"] == "pred"
data_stack.loc[is_pred, "horizon"] = data_stack["combined"].str.extract("(\d+\.?\d*)")[0].astype(float)
data_stack.loc[~is_pred, "horizon"] = data_stack["combined"].str.replace("p10|p90", "")

data_stack["horizon"] = data_stack["horizon"].astype(float)


# # # Drop the original combined column
data_stack.drop(columns=["combined"], inplace=True)

data_stack = data_stack.sort_values(by=["Init Time", "horizon"])

data_stack

data_stack.drop(columns=["generation_mw"], inplace=True)
data_stack.rename(
    columns={
        "Init Time": "forecasting_creation_datetime_utc",
        "value": "generation_mw",
    },
    inplace=True,
)

print("creating start and end times")

# create start and end time handling non-integer hours
data_stack["end_datetime_utc"] = data_stack.apply(
    lambda row: row["forecasting_creation_datetime_utc"] + pd.to_timedelta(float(row["horizon"]), unit="h"),
    axis=1,
)

data_stack["start_datetime_utc"] = data_stack["end_datetime_utc"] - pd.Timedelta(minutes=30)
data_stack.drop(columns=["horizon"], inplace=True)

data_stack.head(20)


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


pivot_df.to_csv(
    "../../data/intraday_pvnet_sum_model_v2_2022_2023_formated_updated_pvlive.csv.gz", index=False, compression="gzip"
)
