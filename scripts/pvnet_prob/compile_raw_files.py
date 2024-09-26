"""
Compile multiple raw PVNet forecast files (.nc) into a single Zarr file.
It processes each file, concatenates the data, and saves it in a more accessible format
for further analysis and processing of PVNet forecasts.
"""

import os
from glob import glob

import pandas as pd
import xarray as xr
from tqdm import tqdm

files = glob("../../../projects/PVNet/backtest_results/t8_a1_2023_ID_updated_ukv/*.nc")
ds_pred_list = []
print(len(files))
for f in tqdm(files, desc="Processing files"):
    ds = xr.open_dataset(f)
    ds = ds.assign_coords(
        horizon_mins=((ds.target_datetime_utc - ds.init_time_utc).isel(init_time_utc=0, drop=True) / 60e9).astype(int)
    )
    ds = ds.swap_dims({"target_datetime_utc": "horizon_mins"})
    ds_pred_list.append(ds.compute())


print("Concatenating datasets...")
ds_pred = xr.concat(ds_pred_list, dim="init_time_utc")
print("Sorting dataset by init_time_utc...")
ds_pred = ds_pred.sortby("init_time_utc")
ds_pred["init_time_utc"] = pd.to_datetime(ds_pred["init_time_utc"].values)


# Save the file as zarr
output_file = "../../data/pvnet_2023-2024_backtest_230924/pvnet_2023-2024_ID_240924_a4.zarr"
print(f"Saving compiled predictions to {output_file}...")

try:
    ds_pred.to_zarr(output_file)
    if os.path.exists(output_file):
        print(f"Saved compiled predictions to {output_file}")
    else:
        print(f"Error: File was not created at {output_file}")
except Exception as e:
    print(f"Error saving file: {str(e)}")

print("Process completed.")
