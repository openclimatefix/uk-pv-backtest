"""
Downloads and processes PV generation data from PVLive for national and GSP levels,
then uploads the processed data to GCS.
"""

import logging
import os
import shutil
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytz
import xarray as xr
from google.cloud import storage
from pvlive_api import PVLive
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


GCS_BUCKET = "PLACEHOLDER"
GCS_PATH = "PLACEHOLDER" #.zarr
GCS_NATIONAL_PATH = "PLACEHOLDER" #.csv
TEMP_DIR = tempfile.gettempdir()


def download_zarr_from_gcs(bucket_name, gcs_path, local_path):
    """Downloads a Zarr dataset from GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=gcs_path)
    for blob in blobs:
        relative_path = os.path.relpath(blob.name, gcs_path)
        local_file_path = os.path.join(local_path, relative_path)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        blob.download_to_filename(local_file_path)
    logging.info(f"Downloaded gs://{bucket_name}/{gcs_path} to {local_path}")


def upload_zarr_to_gcs(zarr_path, bucket_name, gcs_path):
    """Uploads a Zarr dataset to GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    for root, _, files in os.walk(zarr_path):
        for file in files:
            local_file = os.path.join(root, file)
            relative_path = os.path.relpath(local_file, zarr_path)
            blob_path = os.path.join(gcs_path, relative_path)
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_file)
    logging.info(f"Uploaded {zarr_path} to gs://{bucket_name}/{gcs_path}")


def upload_national_data(local_path, bucket_name, gcs_path):
    """Uploads the national data CSV to GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    logging.info(f"Uploaded national data from {local_path} to gs://{bucket_name}/{gcs_path}")


def get_national_pvlive_data():
    """Retrieves national PVLive data and saves it as a CSV."""
    logging.info("Fetching national PVLive data...")
    pvl = PVLive()
    data = pvl.between(
        start=datetime(2016, 12, 1, tzinfo=pytz.utc),
        end=datetime(2025, 1, 1, tzinfo=pytz.utc),
        dataframe=True,
        extra_fields="installedcapacity_mwp,capacity_mwp",
    )

    data = data.sort_values("datetime_gmt")
    data = data.rename(columns={"datetime_gmt": "end_datetime_utc"})
    data["start_datetime_utc"] = data["end_datetime_utc"] - pd.Timedelta(minutes=30)
    data = data.drop(columns=["gsp_id"])
    data = data[[
        "start_datetime_utc",
        "end_datetime_utc",
        "generation_mw",
        "capacity_mwp",
        "installedcapacity_mwp",
    ]]

    local_path = os.path.join(TEMP_DIR, "pvlive_national.csv")
    data.to_csv(local_path, index=False)
    logging.info(f"National PVLive data saved to {local_path}")
    upload_national_data(local_path, GCS_BUCKET, GCS_NATIONAL_PATH)
    return local_path


def get_gsp_pvlive_data():
    """Retrieves and formats GSP PV generation data using the PVLive API."""
    logging.info("Fetching GSP PVLive data...")
    local_zarr_path = os.path.join(TEMP_DIR, "pvlive_gsp.zarr")

    target_times = pd.date_range(datetime(2016, 12, 1), datetime(2025, 1, 1), freq="30T")
    gsp_ids = np.arange(0, 318)

    x = xr.DataArray(
        np.zeros((len(target_times), len(gsp_ids)), dtype=np.float64),
        coords={
            "datetime_gmt": target_times,
            "gsp_id": gsp_ids,
        },
        dims=["datetime_gmt", "gsp_id"]
    )

    ds_gsp = xr.Dataset(
        {
            "generation_mw": x,
            "capacity_mwp": xr.zeros_like(x, dtype=np.float64),
            "installedcapacity_mwp": xr.zeros_like(x, dtype=np.float64),
        }
    )

    pvl = PVLive()
    start = pd.Timestamp(ds_gsp.datetime_gmt.min().item()).tz_localize(timezone.utc)
    end = pd.Timestamp(ds_gsp.datetime_gmt.max().item()).tz_localize(timezone.utc)

    for i in tqdm(gsp_ids, desc="Processing GSPs"):
        try:
            df = pvl.between(
                start=start,
                end=end,
                entity_type="gsp",
                entity_id=i,
                extra_fields="installedcapacity_mwp,capacity_mwp",
                dataframe=True,
            )

            if df.empty:
                logging.warning(f"No data returned for GSP ID {i}")
                continue

            df["datetime_gmt"] = df["datetime_gmt"].dt.tz_localize(None)
            df = df.sort_values("datetime_gmt").set_index("datetime_gmt")

            if start.tz_localize(None) not in df.index:
                try:
                    df.loc[start.tz_localize(None)] = df.loc[start.tz_localize(None) + timedelta(minutes=30)]
                except KeyError:
                    logging.warning(f"Could not set initial value for GSP ID {i}")
                    continue

            df = df.sort_index()

            df_xr = df.to_xarray()
            df_xr = df_xr.reindex(datetime_gmt=ds_gsp.datetime_gmt, method="ffill")
            ds_gsp["installedcapacity_mwp"].loc[dict(gsp_id=i)] = df_xr["installedcapacity_mwp"].values
            ds_gsp["capacity_mwp"].loc[dict(gsp_id=i)] = df_xr["capacity_mwp"].values
            ds_gsp["generation_mw"].loc[dict(gsp_id=i)] = df_xr["generation_mw"].values

        except Exception as e:
            logging.error(f"Error processing GSP ID {i}: {e}")
            continue

    logging.info(f"Saving GSP data to {local_zarr_path}")
    ds_gsp.to_zarr(local_zarr_path, consolidated=True)
    upload_zarr_to_gcs(local_zarr_path, GCS_BUCKET, GCS_PATH)
    shutil.rmtree(local_zarr_path)
    logging.info("GSP data processing complete")


if __name__ == "__main__":
    logging.info("Starting PVLive data processing")
    try:
        national_csv_path = get_national_pvlive_data()
        get_gsp_pvlive_data()
        os.remove(national_csv_path)
        logging.info("PVLive data processing complete")
    except Exception as e:
        logging.exception("An error occurred during PVLive data processing:")
