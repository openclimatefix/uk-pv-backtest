import xarray as xr
from pvlive_api import PVLive
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from google.cloud import storage
import tempfile
import shutil

# Configuration
GCS_BUCKET = "solar-pv-nowcasting-data"
GCS_PATH = "pv_gsp/pvlive_gsp.zarr"  # Path *within* the bucket
END_DATE = "2025-01-01"
TEMP_DIR = tempfile.gettempdir() # Use system temp directory

def download_zarr_from_gcs(bucket_name, gcs_path, local_path):
    """Downloads a Zarr dataset (directory) from GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=gcs_path)  # List all objects under the Zarr path
    for blob in blobs:
        # Construct the local file path, maintaining directory structure
        relative_path = os.path.relpath(blob.name, gcs_path)
        local_file_path = os.path.join(local_path, relative_path)

        # Create necessary directories locally
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # Download the file
        blob.download_to_filename(local_file_path)
    print(f"Downloaded gs://{bucket_name}/{gcs_path} to {local_path}")

def upload_zarr_to_gcs(zarr_path, bucket_name, gcs_path):
    """Uploads a Zarr dataset (directory) to GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    for root, _, files in os.walk(zarr_path):
        for file in files:
            local_file = os.path.join(root, file)
            relative_path = os.path.relpath(local_file, zarr_path)
            blob_path = os.path.join(gcs_path, relative_path)
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_file)
    print(f"Uploaded {zarr_path} to gs://{bucket_name}/{gcs_path}")

def copy_blob(bucket_name, blob_name, destination_bucket_name, destination_blob_name):
    """Copies a blob from one bucket to another with a new name."""
    storage_client = storage.Client()
    source_bucket = storage_client.bucket(bucket_name)
    source_blob = source_bucket.blob(blob_name)
    destination_bucket = storage_client.bucket(destination_bucket_name)

    new_blob = source_bucket.copy_blob(
        source_blob, destination_bucket, destination_blob_name
    )
    print(f"Blob {source_blob.name} in bucket {source_bucket.name} copied to blob {new_blob.name} in bucket {destination_bucket.name}.")


def extend_pvlive_gsp_data():
    """Extends the PVLive GSP data, downloading and uploading to GCS."""
    end_date = pd.Timestamp(END_DATE)

    # 1. Download Zarr dataset from GCS to a temporary directory
    local_zarr_path = os.path.join(TEMP_DIR, "pvlive_gsp.zarr")
    print(f"Downloading Zarr dataset from GCS to {local_zarr_path}")
    download_zarr_from_gcs(GCS_BUCKET, GCS_PATH, local_zarr_path)

    # 2. Load the Zarr dataset
    print(f"Loading Zarr dataset from {local_zarr_path}")
    try:
      ds_gsp = xr.open_zarr(local_zarr_path)
    except Exception as e:
      print(f'Could not load zarr file, so making a new one: {e}')
      new_times = pd.date_range('2018-01-01', '2018-01-02', freq="30T") # added start date
      gsp_ids = [i for i in range(0,350)]

      x_new = xr.DataArray(
          np.zeros((len(new_times), len(gsp_ids))),
          coords={
              "datetime_gmt": new_times,
              "gsp_id": gsp_ids,
          },
      )

      ds_gsp = xr.Dataset(dict(
          generation_mw=x_new,
          capacity_mwp=xr.zeros_like(x_new),
          installedcapacity_mwp=xr.zeros_like(x_new)
      ))

    # ---- ADD BACKUP LOGIC HERE ----
    GCS_BACKUP_PATH = GCS_PATH + "_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Creating backup in GCS: {GCS_BACKUP_PATH}")

    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET)
    blobs = bucket.list_blobs(prefix=GCS_PATH)
    for blob in blobs:
        relative_path = os.path.relpath(blob.name, GCS_PATH)
        destination_blob_name = os.path.join(GCS_BACKUP_PATH, relative_path)
        copy_blob(GCS_BUCKET, blob.name, GCS_BUCKET, destination_blob_name)
    # -------------------------------

    # 3.  Get latest date and check if update is needed
    latest_date = pd.Timestamp(ds_gsp.datetime_gmt.max().item())
    if latest_date.minute == 0:
        latest_date += timedelta(minutes=30)
    else:
        latest_date += timedelta(minutes=30)
    print(f"Latest date in dataset: {latest_date}")
    print(f"Target end date: {end_date}")
    if latest_date >= end_date:
        print("Dataset is already up to date. No update needed.")
        return

    # 4. Generate new time range and GSP IDs
    new_times = pd.date_range(latest_date, end_date, freq="30T")
    gsp_ids = ds_gsp.gsp_id.values
    print(f"Adding {len(new_times)} new time points")

    # 5. Create a new dataset for the additional data
    x_new = xr.DataArray(
        np.zeros((len(new_times), len(gsp_ids))),
        coords={
            "datetime_gmt": new_times,
            "gsp_id": gsp_ids,
        },
        dims=["datetime_gmt", "gsp_id"],  # Explicitly specify dimensions
    )
    ds_new = xr.Dataset(
        {
            "generation_mw": x_new,
            "capacity_mwp": xr.zeros_like(x_new),
            "installedcapacity_mwp": xr.zeros_like(x_new),
        }
    )

    # 6. Initialize PVLive API
    pvl = PVLive()
    start = latest_date.tz_localize(timezone.utc)
    end = end_date.tz_localize(timezone.utc)

    # 7. Fetch and update data for each GSP
    for i in tqdm(gsp_ids, desc="Updating GSP data"):
        try:
            df = pvl.between(
                start=start,
                end=end,
                entity_type="gsp",
                entity_id=i,
                extra_fields="installedcapacity_mwp,capacity_mwp",
                dataframe=True,
            )

            if not df.empty:
                df["datetime_gmt"] = df["datetime_gmt"].dt.tz_localize(None)
                df = df.sort_values("datetime_gmt").set_index("datetime_gmt")

                # Fill missing values for the start time if needed
                if start.tz_localize(None) not in df.index and len(df) > 0:
                    first_time = df.index[0]
                    df.loc[start.tz_localize(None)] = df.loc[first_time]
                    df = df.sort_index()

                # Update ds_new using .loc for safer assignment
                for time in df.index:
                    if time in ds_new.datetime_gmt:
                        ds_new.loc[dict(datetime_gmt=time, gsp_id=i)][
                            "installedcapacity_mwp"
                        ] = df.loc[time, "installedcapacity_mwp"]
                        ds_new.loc[dict(datetime_gmt=time, gsp_id=i)][
                            "capacity_mwp"
                        ] = df.loc[time, "capacity_mwp"]
                        ds_new.loc[dict(datetime_gmt=time, gsp_id=i)][
                            "generation_mw"
                        ] = df.loc[time, "generation_mw"]

        except Exception as e:
            print(f"Error updating GSP {i}: {e}")
            continue  # Continue to the next GSP

    # 8. Combine datasets
    ds_combined = xr.concat([ds_gsp, ds_new], dim="datetime_gmt")
    ds_combined = ds_combined.sortby("datetime_gmt")  # Ensure correct order

    # consolidate metadata
    ds_combined = ds_combined.chunk({"datetime_gmt": -1, "gsp_id": -1})
    ds_combined.to_zarr(local_zarr_path, consolidated=True, mode="w")

    # 9. Upload the updated Zarr dataset to GCS
    print(f"Uploading updated dataset to gs://{GCS_BUCKET}/{GCS_PATH}")
    upload_zarr_to_gcs(local_zarr_path, GCS_BUCKET, GCS_PATH)

    # 10. Clean up the temporary directory
    shutil.rmtree(local_zarr_path)
    print(f"Successfully updated PVLive GSP data to {end_date}")

    return ds_combined

if __name__ == "__main__":
    extend_pvlive_gsp_data()
