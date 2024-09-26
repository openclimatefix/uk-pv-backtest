# Formatting forecast no norm prob
import pandas as pd

# load data
print("Loading Forecast data")
data = pd.read_csv("../../data/pvnet_2023-2024_backtest_230924/pvnet_2023-2024_240924_a4.csv.gz")

data["Init Time"] = pd.to_datetime(data["Init Time"], utc=True)

# join with pvlive capacity data
print("Loading PVLive data")
pvlive = pd.read_csv("../../data/pvnet_2019-2023_backtest_300724/pvlive_2016_2023.csv")
pvlive["end_datetime_utc"] = pd.to_datetime(pvlive["end_datetime_utc"])

data["Init Time"] = pd.to_datetime(data["Init Time"])
data = data.merge(pvlive, left_on="Init Time", right_on="end_datetime_utc")

data.drop(
    columns=[
        "installedcapacity_mwp",
        # "capacity_mwp",
        "start_datetime_utc",
        "end_datetime_utc",
        "generation_mw",
    ],
    inplace=True,
)

# rename columns
print("Renaming columns")
new_cols = {c: c.strip(" Hour Forecast") for c in data.columns if "Hour Forecast" in c}
data.rename(columns=new_cols, inplace=True)


# Melt the dataframe
print("Melting the dataframe")
id_vars = ["Init Time", "capacity_mwp"]
value_vars = [col for col in data.columns if col not in id_vars]
data_stack = pd.melt(data, id_vars=id_vars, value_vars=value_vars, var_name="prediction_type", value_name="value")

data_stack["pred_name"] = data_stack["prediction_type"].str.extract("(p10|p90)")
data_stack["pred_name"] = data_stack["pred_name"].fillna("pred")

# Remove 'p10' and 'p90' from prediction_type column and convert to float
data_stack["horizon"] = data_stack["prediction_type"].replace({"p10 ": "", "p90 ": ""}, regex=True)
data_stack["horizon"] = pd.to_numeric(data_stack["horizon"], errors="coerce")

data_stack = data_stack.sort_values(by=["Init Time", "horizon"])
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

# Pivot the DataFrame
pivot_df = data_stack.pivot_table(
    index=[
        "forecasting_creation_datetime_utc",
        "start_datetime_utc",
        "end_datetime_utc",
        "capacity_mwp",
    ],
    columns="pred_name",
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
    "../../data/pvnet_2023-2024_backtest_230924/pvnet_2023-2024_240924_a4.csv.gz",
    index=False,
    compression="gzip",
)
