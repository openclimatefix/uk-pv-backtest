"""
Get data from PVlive and save it to a csv

This CV will have the following columns
- start_datetime_utc: datetime - the start datetime of the period
- end_datetime_utc: datetime - the end datetime of the period
- generation_mw: float - the solar generation value
- capacity_mwp: float - The estimated capacity of the system
- installedcapacity_mwp: float - The installed capacity (this changes is time)
"""

from datetime import datetime

import pandas as pd
import pytz
from pvlive_api import PVLive

# set up pv live
pvl = PVLive()

# get data, 1 year takes about 5 seconds
data = pvl.between(
    start=datetime(2016, 12, 1, tzinfo=pytz.utc),
    end=datetime(2024, 1, 1, tzinfo=pytz.utc),
    dataframe=True,
    extra_fields="installedcapacity_mwp,capacity_mwp",
)

# order data
data.sort_values("datetime_gmt", inplace=True)

# rename columns
data.rename(columns={"datetime_gmt": "end_datetime_utc"}, inplace=True)
data["start_datetime_utc"] = data["end_datetime_utc"] - pd.Timedelta(minutes=30)

# drop column
data.drop(columns=["gsp_id"], inplace=True)

# order the columns
data = data[
    [
        "start_datetime_utc",
        "end_datetime_utc",
        "generation_mw",
        "capacity_mwp",
        "installedcapacity_mwp",
    ]
]

# save to csv
data.to_csv("pvlive_2016_2023.csv", index=False)
