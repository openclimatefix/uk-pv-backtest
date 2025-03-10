"""Analyses PV generation data from a Zarr dataset.

Loads data, performs analyses, and prints a summary.
"""

import argparse
import logging
import os
import shutil
import sys
import tempfile
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from google.cloud import storage


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def download_zarr_from_gcs(bucket_name: str, gcs_path: str, local_path: str) -> bool:
    """Downloads a Zarr dataset (directory) from GCS.
    
    Args:
        bucket_name: GCS bucket name.
        gcs_path: Path within the bucket.
        local_path: Local path to save the data.
        
    Returns:
        True if successful, False otherwise.
    """
    logging.info(f"Downloading zarr from gs://{bucket_name}/{gcs_path} to {local_path}...")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=gcs_path)
    
    blob_list = list(blobs)
    
    if not blob_list:
        logging.error(f"No files found at gs://{bucket_name}/{gcs_path}")
        return False
        
    for blob in blob_list:
        relative_path = os.path.relpath(blob.name, gcs_path)
        local_file_path = os.path.join(local_path, relative_path)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        blob.download_to_filename(local_file_path)
    
    logging.info(f"Downloaded zarr dataset to {local_path}")
    return True


def load_data(zarr_path: str) -> Optional[xr.Dataset]:
    """Load the GSP data from a zarr file or GCS path.
    
    Args:
        zarr_path: Path to the Zarr dataset (local or GCS).
        
    Returns:
        An xarray Dataset, or None if loading fails.
    """
    logging.info(f"Loading data from {zarr_path}...")
    
    if zarr_path.startswith("gs://"):
        path_parts = zarr_path.replace("gs://", "").split("/")
        bucket_name = path_parts[0]
        blob_path = "/".join(path_parts[1:])
        
        temp_dir = tempfile.mkdtemp()
        local_zarr_path = os.path.join(temp_dir, "temp.zarr")
        
        try:
            success = download_zarr_from_gcs(bucket_name, blob_path, local_zarr_path)
            if not success:
                logging.error(f"Failed to download zarr dataset from {zarr_path}")
                shutil.rmtree(temp_dir)
                return None
            ds = xr.open_zarr(local_zarr_path)
            
        except Exception as e:
            logging.error(f"Error loading zarr dataset: {e}")
            shutil.rmtree(temp_dir)
            return None
            
    else:
        try:
            ds = xr.open_zarr(zarr_path)
        except Exception as e:
            logging.error(f"Error loading zarr dataset: {e}")
            return None
    
    logging.info(f"Loaded zarr dataset with dimensions: {dict(ds.dims)}")
    logging.info(f"Variables: {list(ds.data_vars)}")
    logging.info(f"Time range: {ds.datetime_gmt.values.min()} to {ds.datetime_gmt.values.max()}")
    logging.info(f"GSP IDs: {len(ds.gsp_id)} GSPs (0 to {ds.gsp_id.values.max()})")
    
    return ds


def analyse_data(ds: xr.Dataset) -> None:
    """Analyses data and prints key findings.
    
    Args:
        ds: The xarray Dataset containing the PV data.
        
    Returns:
        None (prints the analysis results).
    """
    if ds is None:
        logging.error("Dataset is empty. Exiting.")
        return

    logging.info("=== PV Data Analysis ===")
    logging.info(f"Data from {ds.datetime_gmt.values.min()} to {ds.datetime_gmt.values.max()}")
    logging.info(f"Total Data Points: {len(ds.datetime_gmt) * len(ds.gsp_id):,}")
    
    logging.info("")
    logging.info("--- Basic Stats ---")
    gen_mean = float(np.nanmean(ds.generation_mw.values))
    gen_max = float(np.nanmax(ds.generation_mw.values))
    cap_mean = float(np.nanmean(ds.capacity_mwp.values))
    cap_max = float(np.nanmax(ds.capacity_mwp.values))
    installed_mean = float(np.nanmean(ds.installedcapacity_mwp.values))
    installed_max = float(np.nanmax(ds.installedcapacity_mwp.values))
    
    logging.info("Generation (MW):")
    logging.info(f"  Mean: {gen_mean:.2f}")
    logging.info(f"  Max: {gen_max:.2f}")
    
    logging.info("Capacity (MWp):")
    logging.info(f"  Mean: {cap_mean:.2f}")
    logging.info(f"  Max: {cap_max:.2f}")
    
    logging.info("Installed Capacity (MWp):")
    logging.info(f"  Mean: {installed_mean:.2f}")
    logging.info(f"  Max: {installed_max:.2f}")
    
    missing_gen = np.isnan(ds.generation_mw.values).sum()
    missing_cap = np.isnan(ds.capacity_mwp.values).sum()
    missing_installed = np.isnan(ds.installedcapacity_mwp.values).sum()
    
    total_points = len(ds.datetime_gmt) * len(ds.gsp_id)
    
    logging.info("")
    logging.info("Missing Values:")
    logging.info(f"  generation_mw: {missing_gen:,} ({missing_gen/total_points*100:.2f}%)")
    logging.info(f"  capacity_mwp: {missing_cap:,} ({missing_cap/total_points*100:.2f}%)")
    logging.info(f"  installedcapacity_mwp: {missing_installed:,} ({missing_installed/total_points*100:.2f}%)")
    
    logging.info("")
    logging.info("--- GSP Analysis ---")
    gsp_gen_mean = np.nanmean(ds.generation_mw.values, axis=0)
    gsp_cap_mean = np.nanmean(ds.capacity_mwp.values, axis=0)
    gsp_installed_mean = np.nanmean(ds.installedcapacity_mwp.values, axis=0)

    gsp_cf_mean = np.divide(gsp_gen_mean, gsp_cap_mean, out=np.zeros_like(gsp_gen_mean), where=gsp_cap_mean!=0) * 100
    
    gsp_df = pd.DataFrame({
        'gsp_id': ds.gsp_id.values,
        'mean_generation_mw': gsp_gen_mean,
        'mean_capacity_mwp': gsp_cap_mean,
        'mean_installedcapacity_mwp': gsp_installed_mean,
        'mean_capacity_factor': gsp_cf_mean
    })
    
    gsp_df_sorted = gsp_df.sort_values('mean_generation_mw', ascending=False)
    logging.info("Top 10 GSPs by mean generation (MW):")
    for idx, row in gsp_df_sorted.head(10).iterrows():
        logging.info(f"  GSP {int(row['gsp_id'])}: {row['mean_generation_mw']:.2f} MW (CF: {row['mean_capacity_factor']:.2f}%)")
    
    total_gen_capacity = np.sum(gsp_gen_mean)
    gsp_contribution = (gsp_gen_mean / total_gen_capacity) * 100
    top_contributors = np.argsort(gsp_contribution)[::-1]
    
    logging.info("")
    logging.info("GSP Contribution:")
    logging.info(f"  Top 10 GSPs contribute: {np.sum(gsp_contribution[top_contributors[:10]]):.2f}%")
    logging.info(f"  Top 50 GSPs contribute: {np.sum(gsp_contribution[top_contributors[:50]]):.2f}%")
    logging.info(f"  Top 100 GSPs contribute: {np.sum(gsp_contribution[top_contributors[:100]]):.2f}%")
    
    # Count active GSPs
    gsp_gen_max = np.nanmax(ds.generation_mw.values, axis=0)
    active_gsps = np.sum(gsp_gen_max > 0)
    logging.info("")
    logging.info(f"Active GSPs (max generation > 0): {active_gsps} / {len(ds.gsp_id)}")
    
    logging.info("")
    logging.info("--- Temporal Analysis ---")
    datetimes = pd.to_datetime(ds.datetime_gmt.values)
    hours = np.array([dt.hour for dt in datetimes])
    months = np.array([dt.month for dt in datetimes])
    years = np.array([dt.year for dt in datetimes])
    
    total_gen_by_time = np.nansum(ds.generation_mw.values, axis=1)
    
    hourly_gen = np.zeros(24)
    hourly_counts = np.zeros(24)
    
    for hour in range(24):
        hour_indices = np.where(hours == hour)[0]
        if len(hour_indices) > 0:
            hourly_gen[hour] = np.mean(total_gen_by_time[hour_indices])
            hourly_counts[hour] = len(hour_indices)
    
    peak_hour = np.argmax(hourly_gen)
    logging.info(f"Peak Generation Hour (UTC): {peak_hour:02d}:00")
    
    monthly_gen = np.zeros(12)
    monthly_counts = np.zeros(12)
    
    for month in range(1, 13):
        month_indices = np.where(months == month)[0]
        if len(month_indices) > 0:
            monthly_gen[month-1] = np.mean(total_gen_by_time[month_indices])
            monthly_counts[month-1] = len(month_indices)
    
    month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    best_month = np.argmax(monthly_gen) + 1
    worst_month = np.argmin(monthly_gen) + 1
    logging.info(f"Best Month (Avg Generation): {month_names[best_month-1]} ({monthly_gen[best_month-1]:.2f} MW)")
    logging.info(f"Worst Month (Avg Generation): {month_names[worst_month-1]} ({monthly_gen[worst_month-1]:.2f} MW)")
    
    all_years = sorted(np.unique(years))
    yearly_gen = np.zeros(len(all_years))
    yearly_counts = np.zeros(len(all_years))
    
    for i, year in enumerate(all_years):
        year_indices = np.where(years == year)[0]
        if len(year_indices) > 0:
            yearly_gen[i] = np.mean(total_gen_by_time[year_indices])
            yearly_counts[i] = len(year_indices)
    
    logging.info("")
    logging.info("--- Yearly Analysis ---")
    logging.info("Average Total Generation by Year (MW):")
    yearly_summary = pd.DataFrame({
        'Year': all_years,
        'Avg_Gen_MW': yearly_gen,
        'Total_Timesteps': yearly_counts
    })
    
    if len(all_years) > 1:
        yearly_summary['Gen_Change_MW'] = yearly_summary['Avg_Gen_MW'].diff()
        yearly_summary['Gen_Pct_Change'] = yearly_summary['Avg_Gen_MW'].pct_change() * 100
        logging.info(yearly_summary)
    else:
        logging.info(yearly_summary[['Year', 'Avg_Gen_MW', 'Total_Timesteps']])
        logging.info("Year-on-Year Growth: Cannot be calculated (single year data).")


def main() -> None:
    """Main function to parse arguments and run the analysis."""
    parser = argparse.ArgumentParser(description='Analyses PV GSP data from .zarr dataset')
    parser.add_argument('zarr_path', help='Path to the zarr dataset (local or GCS)')
    args = parser.parse_args()
    
    ds = load_data(args.zarr_path)
    if ds is not None:
        analyse_data(ds)
    
    # Clean up temporary files if using GCS
    if args.zarr_path.startswith("gs://"):
        local_zarr_path = os.path.join(tempfile.gettempdir(), "temp.zarr")
        if os.path.exists(local_zarr_path):
            shutil.rmtree(local_zarr_path)


if __name__ == "__main__":
    main()
