"""
Identify missing dates or gaps in forecast data. It determines the time intervals between forecast
entries and identifies any intervals that are larger than expected, saving the results as a csv.
"""

import pandas as pd

df = pd.read_csv(
    "../data/pvnet_2019-2024_backtest_240924/forecast_v=9c__model_name_1=pvnet_app_v__model_version_1=2.3.19__start_date=2019-01-01__end_date=2024-01-01.csv.gz"
)

# get unique init times
init_times = pd.DataFrame(
    df["forecasting_creation_datetime_utc"].unique(),
    columns=["forecasting_creation_datetime_utc"],
)
init_times["forecasting_creation_datetime_utc"] = pd.to_datetime(init_times["forecasting_creation_datetime_utc"])

# find the differences
init_times["next_forecasting_creation_datetime_utc"] = init_times.shift(-1)
init_times["diff"] = (
    init_times["next_forecasting_creation_datetime_utc"] - init_times["forecasting_creation_datetime_utc"]
)

# remove any 30 minute gaps
gaps = init_times[init_times["diff"] != pd.Timedelta("30T")]

# look at distribution of gaps
gaps["count"] = 1
gaps_sum = gaps[["diff", "count"]].groupby("diff").sum()

# Print summary of gaps
print(gaps_sum)

# Print dates where there are gaps
print("Dates with gaps:")
print(gaps[["forecasting_creation_datetime_utc", "diff"]])

# Save the gaps information to a CSV file
gaps.to_csv("../data/pvnet_2019-2024_backtest_240924/forecast_gaps_v9c.csv", index=False)
print("Gaps information saved")
