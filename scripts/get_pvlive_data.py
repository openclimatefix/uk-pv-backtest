"""
Extension to the original get_pvlive_data.py to update national PV data beyond 2024-01-01.

This script loads the existing CSV file (if available), identifies the latest data point,
and extends it with new data from the PVLive API up to 2025-01-01.

This CSV will have the following columns:
- start_datetime_utc: datetime - the start datetime of the period
- end_datetime_utc: datetime - the end datetime of the period
- generation_mw: float - the solar generation value
- capacity_mwp: float - The estimated capacity of the system
- installedcapacity_mwp: float - The installed capacity (this changes is time)
"""

from datetime import datetime
import os
import pandas as pd
import pytz
from pvlive_api import PVLive
from google.cloud import storage

# Configuration
EXISTING_CSV = "pvlive_2016_2023.csv"  # Path to existing CSV file
NEW_CSV = "pvlive_2016_2025.csv"  # Path for the updated CSV file
GCS_BUCKET = "solar-pv-nowcasting-data"
GCS_PATH = "pv_national/pvlive_national.csv"
END_DATE = datetime(2025, 1, 1, tzinfo=pytz.utc)

def extend_pvlive_data():
    """Extend existing PVLive data with new data up to 2025-01-01"""
    
    # Check if existing CSV exists
    if os.path.exists(EXISTING_CSV):
        print(f"Loading existing data from {EXISTING_CSV}")
        data = pd.read_csv(EXISTING_CSV)
        
        # Convert datetime columns
        data["start_datetime_utc"] = pd.to_datetime(data["start_datetime_utc"])
        data["end_datetime_utc"] = pd.to_datetime(data["end_datetime_utc"])
        
        # Get the latest date in the dataset
        latest_date = data["end_datetime_utc"].max()
        
        # Convert to timezone-aware datetime if not already
        if isinstance(latest_date, pd.Timestamp) and latest_date.tzinfo is None:
            latest_date = latest_date.replace(tzinfo=pytz.utc)
    else:
        print(f"CSV file {EXISTING_CSV} not found. Starting from 2016-12-01.")
        latest_date = datetime(2016, 12, 1, tzinfo=pytz.utc)
        data = pd.DataFrame(columns=[
            "start_datetime_utc",
            "end_datetime_utc",
            "generation_mw",
            "capacity_mwp",
            "installedcapacity_mwp",
        ])
    
    print(f"Latest date in dataset: {latest_date}")
    print(f"Target end date: {END_DATE}")
    
    # If the data is already up to date, no need to update
    if latest_date >= END_DATE:
        print("Dataset is already up to date. No update needed.")
        return data
    
    # set up pv live
    pvl = PVLive()
    
    # get new data
    print(f"Fetching new data from {latest_date} to {END_DATE}")
    new_data = pvl.between(
        start=latest_date,
        end=END_DATE,
        dataframe=True,
        extra_fields="installedcapacity_mwp,capacity_mwp",
    )
    
    if new_data.empty:
        print("No new data available from PVLive.")
        return data
    
    # rename columns
    new_data.rename(columns={"datetime_gmt": "end_datetime_utc"}, inplace=True)
    new_data["start_datetime_utc"] = new_data["end_datetime_utc"] - pd.Timedelta(minutes=30)
    
    # drop column
    if "gsp_id" in new_data.columns:
        new_data.drop(columns=["gsp_id"], inplace=True)
    
    # order the columns
    new_data = new_data[
        [
            "start_datetime_utc",
            "end_datetime_utc",
            "generation_mw",
            "capacity_mwp",
            "installedcapacity_mwp",
        ]
    ]
    
    # Filter out data that might already be in the dataset to avoid duplicates
    new_data = new_data[new_data["end_datetime_utc"] > latest_date]
    
    # Combine with existing data
    combined_data = pd.concat([data, new_data], ignore_index=True)
    
    # Sort by date and remove any potential duplicates
    combined_data.sort_values("end_datetime_utc", inplace=True)
    combined_data.drop_duplicates(subset=["end_datetime_utc"], keep="last", inplace=True)
    combined_data.reset_index(drop=True, inplace=True)
    
    print(f"Saving updated data to {NEW_CSV}")
    combined_data.to_csv(NEW_CSV, index=False)
    
    print(f"Uploading to gs://{GCS_BUCKET}/{GCS_PATH}")
    upload_to_gcs(NEW_CSV, GCS_BUCKET, GCS_PATH)
    
    print(f"Successfully updated PVLive data to {combined_data['end_datetime_utc'].max()}")
    return combined_data

def upload_to_gcs(file_path, bucket_name, blob_name):
    """
    Upload a file to Google Cloud Storage.
    
    Args:
        file_path: Path to the file to upload
        bucket_name: GCS bucket name
        blob_name: Name of the blob in the bucket
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    blob.upload_from_filename(file_path)
    
    print(f"Successfully uploaded to gs://{bucket_name}/{blob_name}")

if __name__ == "__main__":
    extend_pvlive_data()
