"""Rename a forecast file based on specific model details and date range."""

import pandas as pd

dir = "data/pvnet_2019-2024_backtest_240924"
old_file = "pvnet_2019-2024_240924_combined.csv.gz"


v_id = "9c"
model_name_1 = "pvnet_app_v"
model_version_1 = "2.3.19"
start_date = "2019-01-01"
end_date = "2024-01-01"

new_file = (
    f"forecast_v={v_id}__"
    f"model_name_1={model_name_1}__"
    f"model_version_1={model_version_1}__"
    f"start_date={start_date}__"
    f"end_date={end_date}.csv.gz"
)

df = pd.read_csv(f"../{dir}/{old_file}")
df.to_csv(f"../{dir}/{new_file}", index=False)
