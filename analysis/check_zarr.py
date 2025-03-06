"""Analyses / checks .zarr dataset"""

import argparse
import logging
import os
import shutil
import sys
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from google.cloud import storage


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def download_zarr_from_gcs(bucket_name, gcs_path, local_path):
    """Downloads a Zarr dataset (directory) from GCS."""
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


def load_data(zarr_path):
    """Load the GSP data from a zarr file or GCS path."""
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
                sys.exit(1)
            ds = xr.open_zarr(local_zarr_path)
            
        except Exception as e:
            logging.error(f"Error loading zarr dataset: {e}")
            shutil.rmtree(temp_dir)
            sys.exit(1)
            
    else:
        try:
            ds = xr.open_zarr(zarr_path)
        except Exception as e:
            logging.error(f"Error loading zarr dataset: {e}")
            sys.exit(1)
    
    logging.info(f"Loaded zarr dataset with dimensions: {dict(ds.dims)}")
    logging.info(f"Variables: {list(ds.data_vars)}")
    logging.info(f"Time range: {ds.datetime_gmt.values.min()} to {ds.datetime_gmt.values.max()}")
    logging.info(f"GSP IDs: {len(ds.gsp_id)} GSPs (0 to {ds.gsp_id.values.max()})")
    
    return ds


def print_separator():
    """Print a separator line."""
    logging.info("\n" + "="*80 + "\n")


def basic_stats(ds):
    """Calculate and print basic statistics."""
    print_separator()
    logging.info("BASIC STATISTICS")
    print_separator()
    
    logging.info(f"Dataset Summary:")
    logging.info(f"Time period: {ds.datetime_gmt.values.min()} to {ds.datetime_gmt.values.max()}")
    logging.info(f"Total timesteps: {len(ds.datetime_gmt)}")
    logging.info(f"Total GSPs: {len(ds.gsp_id)}")
    logging.info(f"Total data points: {len(ds.datetime_gmt) * len(ds.gsp_id):,}")
    
    years = np.unique([pd.Timestamp(dt).year for dt in ds.datetime_gmt.values])
    logging.info(f"Years covered: {', '.join(map(str, sorted(years)))}")
    
    missing_gen = np.isnan(ds.generation_mw.values).sum()
    missing_cap = np.isnan(ds.capacity_mwp.values).sum()
    missing_installed = np.isnan(ds.installedcapacity_mwp.values).sum()
    
    total_points = len(ds.datetime_gmt) * len(ds.gsp_id)
    
    logging.info("\nMissing Values:")
    logging.info(f"  generation_mw: {missing_gen:,} ({missing_gen/total_points*100:.2f}%)")
    logging.info(f"  capacity_mwp: {missing_cap:,} ({missing_cap/total_points*100:.2f}%)")
    logging.info(f"  installedcapacity_mwp: {missing_installed:,} ({missing_installed/total_points*100:.2f}%)")
    

    gen_mean = np.nanmean(ds.generation_mw.values)
    gen_max = np.nanmax(ds.generation_mw.values)
    cap_mean = np.nanmean(ds.capacity_mwp.values)
    cap_max = np.nanmax(ds.capacity_mwp.values)
    installed_mean = np.nanmean(ds.installedcapacity_mwp.values)
    installed_max = np.nanmax(ds.installedcapacity_mwp.values)
    
    logging.info("\nGeneration (MW):")
    logging.info(f"  Mean across all GSPs: {gen_mean:.2f}")
    logging.info(f"  Max across all GSPs: {gen_max:.2f}")
    
    logging.info("\nCapacity (MWp):")
    logging.info(f"  Mean across all GSPs: {cap_mean:.2f}")
    logging.info(f"  Max across all GSPs: {cap_max:.2f}")
    
    logging.info("\nInstalled Capacity (MWp):")
    logging.info(f"  Mean across all GSPs: {installed_mean:.2f}")
    logging.info(f"  Max across all GSPs: {installed_max:.2f}")

    max_gen_idx = np.nanargmax(ds.generation_mw.values)
    max_gen_flat_indices = np.unravel_index(max_gen_idx, ds.generation_mw.shape)
    max_gen_time_idx = max_gen_flat_indices[0]
    max_gen_gsp_idx = max_gen_flat_indices[1]
    
    max_gen_time = ds.datetime_gmt.values[max_gen_time_idx]
    max_gen_gsp = ds.gsp_id.values[max_gen_gsp_idx]
    
    logging.info(f"\nMaximum generation of {gen_max:.2f} MW occurred at:")
    logging.info(f"  Time: {max_gen_time}")
    logging.info(f"  GSP ID: {max_gen_gsp}")


def gsp_analysis(ds):
    """Analyse generation and capacity"""
    print_separator()
    logging.info("GSP ANALYSIS")
    print_separator()
    
    gsp_gen_mean = np.nanmean(ds.generation_mw.values, axis=0)
    gsp_gen_max = np.nanmax(ds.generation_mw.values, axis=0)
    gsp_cap_mean = np.nanmean(ds.capacity_mwp.values, axis=0)
    gsp_cap_max = np.nanmax(ds.capacity_mwp.values, axis=0)
    gsp_installed_mean = np.nanmean(ds.installedcapacity_mwp.values, axis=0)

    gsp_cf_mean = gsp_gen_mean / gsp_cap_mean * 100
    
    gsp_df = pd.DataFrame({
        'gsp_id': ds.gsp_id.values,
        'mean_generation_mw': gsp_gen_mean,
        'max_generation_mw': gsp_gen_max,
        'mean_capacity_mwp': gsp_cap_mean,
        'max_capacity_mwp': gsp_cap_max,
        'mean_installedcapacity_mwp': gsp_installed_mean,
        'mean_capacity_factor': gsp_cf_mean
    })
    
    gsp_df_sorted = gsp_df.sort_values('mean_generation_mw', ascending=False)
    
    logging.info("Top 10 GSPs by mean generation (MW):")
    for idx, row in gsp_df_sorted.head(10).iterrows():
        logging.info(f"  GSP {int(row['gsp_id'])}: {row['mean_generation_mw']:.2f} MW (Capacity Factor: {row['mean_capacity_factor']:.2f}%)")
    
    gsp_df_sorted = gsp_df.sort_values('mean_capacity_factor', ascending=False)
    
    logging.info("\nTop 10 GSPs by mean capacity factor (%):")
    for idx, row in gsp_df_sorted.head(10).iterrows():
        logging.info(f"  GSP {int(row['gsp_id'])}: {row['mean_capacity_factor']:.2f}% (Mean Generation: {row['mean_generation_mw']:.2f} MW)")
    
    logging.info("\nDistribution of mean generation capacity:")
    gen_bins = [0, 5, 10, 20, 50, 100, 500, float('inf')]
    gen_counts = pd.cut(gsp_df['mean_generation_mw'], bins=gen_bins).value_counts().sort_index()
    
    for interval, count in gen_counts.items():
        logging.info(f"  {interval}: {count} GSPs")

    active_gsps = np.sum(gsp_gen_max > 0)
    logging.info(f"\nNumber of active GSPs (max generation > 0): {active_gsps} / {len(ds.gsp_id)}")

    zero_gen_gsps = np.where(gsp_gen_max == 0)[0]
    if len(zero_gen_gsps) > 0:
        logging.info(f"\nGSPs with zero generation: {len(zero_gen_gsps)}")
        logging.info(f"  GSP IDs: {', '.join(map(str, ds.gsp_id.values[zero_gen_gsps][:10]))}{'...' if len(zero_gen_gsps) > 10 else ''}")


def temporal_analysis(ds):
    """Analyse temporal patterns."""
    print_separator()
    logging.info("TEMPORAL ANALYSIS")
    print_separator()
    
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
    
    logging.info("Average Total Generation by Hour of Day (MW):")
    for hour in range(24):
        if hourly_counts[hour] > 0:
            logging.info(f"  {hour:02d}:00-{hour+1:02d}:00: {hourly_gen[hour]:.2f} MW")
    
    monthly_gen = np.zeros(12)
    monthly_counts = np.zeros(12)
    
    for month in range(1, 13):
        month_indices = np.where(months == month)[0]
        if len(month_indices) > 0:
            monthly_gen[month-1] = np.mean(total_gen_by_time[month_indices])
            monthly_counts[month-1] = len(month_indices)
    
    month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    logging.info("\nAverage Total Generation by Month (MW):")
    for month in range(12):
        if monthly_counts[month] > 0:
            logging.info(f"  {month_names[month]}: {monthly_gen[month]:.2f} MW")

    all_years = sorted(np.unique(years))
    yearly_gen = np.zeros(len(all_years))
    yearly_counts = np.zeros(len(all_years))
    
    for i, year in enumerate(all_years):
        year_indices = np.where(years == year)[0]
        if len(year_indices) > 0:
            yearly_gen[i] = np.mean(total_gen_by_time[year_indices])
            yearly_counts[i] = len(year_indices)
    
    logging.info("\nAverage Total Generation by Year (MW):")
    for i, year in enumerate(all_years):
        if yearly_counts[i] > 0:
            logging.info(f"  {year}: {yearly_gen[i]:.2f} MW (from {int(yearly_counts[i])} timesteps)")


def spatial_analysis(ds):
    """Analyse spatial patterns (GSP distribution)."""
    print_separator()
    logging.info("SPATIAL ANALYSIS (GSP DISTRIBUTION)")
    print_separator()

    gsp_gen_mean = np.nanmean(ds.generation_mw.values, axis=0)

    total_gen_capacity = np.sum(gsp_gen_mean)
    logging.info(f"Total mean generation capacity across all GSPs: {total_gen_capacity:.2f} MW")

    gsp_contribution = (gsp_gen_mean / total_gen_capacity) * 100

    top_contributors = np.argsort(gsp_contribution)[::-1]
    cumulative_contribution = np.cumsum(gsp_contribution[top_contributors])

    logging.info("\nContribution to total generation:")
    logging.info(f"  Top 10 GSPs contribute: {np.sum(gsp_contribution[top_contributors[:10]]):.2f}%")
    logging.info(f"  Top 50 GSPs contribute: {np.sum(gsp_contribution[top_contributors[:50]]):.2f}%")
    logging.info(f"  Top 100 GSPs contribute: {np.sum(gsp_contribution[top_contributors[:100]]):.2f}%")

    gsps_for_50pct = np.where(cumulative_contribution >= 50)[0][0] + 1
    gsps_for_80pct = np.where(cumulative_contribution >= 80)[0][0] + 1

    logging.info(f"\nGSP concentration:")
    logging.info(f"  Number of GSPs needed for 50% of total generation: {gsps_for_50pct}")
    logging.info(f"  Number of GSPs needed for 80% of total generation: {gsps_for_80pct}")

    logging.info("\nDistribution of GSPs by mean generation:")
    gen_thresholds = [0, 1, 5, 10, 20, 50, 100]

    for i in range(len(gen_thresholds)):
        if i < len(gen_thresholds) - 1:
            count = np.sum((gsp_gen_mean >= gen_thresholds[i]) & (gsp_gen_mean < gen_thresholds[i+1]))
            logging.info(f"  {gen_thresholds[i]}-{gen_thresholds[i+1]} MW: {count} GSPs")
        else:
            count = np.sum(gsp_gen_mean >= gen_thresholds[i])
            logging.info(f"  >{gen_thresholds[i]} MW: {count} GSPs")


def correlation_analysis(ds):
    """Analyse correlations between GSPs."""
    print_separator()
    logging.info("CORRELATION ANALYSIS")
    print_separator()

    timestamps = pd.to_datetime(ds.datetime_gmt.values)
    random_date = timestamps[len(timestamps)//2].date()

    day_indices = [i for i, ts in enumerate(timestamps) if ts.date() == random_date]

    if len(day_indices) == 0:
        logging.info(f"No data found for date {random_date}")
        return

    logging.info(f"Analyzing correlations using data from {random_date} ({len(day_indices)} timesteps)")

    day_data = ds.generation_mw.values[day_indices, :]
    active_gsp_indices = np.where(np.nansum(day_data, axis=0) > 0)[0]
    active_gsp_ids = ds.gsp_id.values[active_gsp_indices]

    if len(active_gsp_ids) < 5:
        logging.info(f"Too few active GSPs ({len(active_gsp_ids)}) found for correlation analysis")
        return

    logging.info(f"Found {len(active_gsp_ids)} active GSPs for correlation analysis")

    day_data_active = day_data[:, active_gsp_indices]
    day_data_active = np.nan_to_num(day_data_active, nan=0)
    correlations = np.corrcoef(day_data_active.T)

    correlation_values = correlations[np.triu_indices(len(correlations), k=1)]

    logging.info("\nCorrelation statistics:")
    logging.info(f"  Mean correlation: {np.mean(correlation_values):.4f}")
    logging.info(f"  Median correlation: {np.median(correlation_values):.4f}")
    logging.info(f"  Min correlation: {np.min(correlation_values):.4f}")
    logging.info(f"  Max correlation: {np.max(correlation_values):.4f}")

    strong_positive = np.sum(correlation_values > 0.8) / len(correlation_values) * 100
    moderate_positive = np.sum((correlation_values > 0.5) & (correlation_values <= 0.8)) / len(correlation_values) * 100
    weak_positive = np.sum((correlation_values > 0.2) & (correlation_values <= 0.5)) / len(correlation_values) * 100

    logging.info("\nCorrelation distribution:")
    logging.info(f"  Strong positive (>0.8): {strong_positive:.2f}%")
    logging.info(f"  Moderate positive (0.5-0.8): {moderate_positive:.2f}%")
    logging.info(f"  Weak positive (0.2-0.5): {weak_positive:.2f}%")

    logging.info(f"\nInterpretation: {strong_positive:.1f}% of GSP pairs show strong correlation,")
    logging.info(f"suggesting that solar generation is {'highly' if strong_positive > 50 else 'moderately' if strong_positive > 25 else 'somewhat'} synchronized across the UK.")


def capacity_factor_analysis(ds):
    """Analyse capacity factors across GSPs"""
    print_separator()
    logging.info("CAPACITY FACTOR ANALYSIS")
    print_separator()

    capacity = ds.capacity_mwp.values
    generation = ds.generation_mw.values
    valid_mask = (capacity > 0) & (~np.isnan(capacity)) & (~np.isnan(generation))
    capacity_factor = np.full_like(generation, np.nan)
    capacity_factor[valid_mask] = (generation[valid_mask] / capacity[valid_mask]) * 100

    mean_cf_by_gsp = np.nanmean(capacity_factor, axis=0)
    mean_cf_by_time = np.nanmean(capacity_factor, axis=1)

    timestamps = pd.to_datetime(ds.datetime_gmt.values)
    cf_time_series = pd.Series(mean_cf_by_time, index=timestamps)

    logging.info(f"Capacity Factor Statistics (%):")
    logging.info(f"  Mean across all GSPs and times: {np.nanmean(capacity_factor):.2f}%")
    logging.info(f"  Median across all GSPs and times: {np.nanmedian(capacity_factor):.2f}%")
    logging.info(f"  Max across all GSPs and times: {np.nanmax(capacity_factor):.2f}%")

    monthly_cf = cf_time_series.groupby(cf_time_series.index.month).mean()

    month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']

    logging.info("\nMean Capacity Factor by Month (%):")
    for month in range(1, 13):
        if month in monthly_cf.index:
            logging.info(f"  {month_names[month-1]}: {monthly_cf[month]:.2f}%")

    hourly_cf = cf_time_series.groupby(cf_time_series.index.hour).mean()

    logging.info("\nMean Capacity Factor by Hour (%):")
    for hour in range(24):
        if hour in hourly_cf.index:
            logging.info(f"  {hour:02d}:00-{hour+1:02d}:00: {hourly_cf[hour]:.2f}%")


    top_cf_indices = np.argsort(mean_cf_by_gsp)[::-1]
    bottom_cf_indices = np.argsort(mean_cf_by_gsp)

    top_cf_indices = top_cf_indices[~np.isnan(mean_cf_by_gsp[top_cf_indices])]
    bottom_cf_indices = bottom_cf_indices[~np.isnan(mean_cf_by_gsp[bottom_cf_indices])]

    logging.info("\nGSPs with Highest Mean Capacity Factor:")
    for i in range(min(10, len(top_cf_indices))):
        gsp_idx = top_cf_indices[i]
        gsp_id = ds.gsp_id.values[gsp_idx]
        logging.info(f"  GSP {int(gsp_id)}: {mean_cf_by_gsp[gsp_idx]:.2f}%")

    logging.info("\nGSPs with Lowest Mean Capacity Factor (excluding zero/NaN):")
    for i in range(min(10, len(bottom_cf_indices))):
        gsp_idx = bottom_cf_indices[i]
        if mean_cf_by_gsp[gsp_idx] <= 0 or np.isnan(mean_cf_by_gsp[gsp_idx]):
            continue
        gsp_id = ds.gsp_id.values[gsp_idx]
        logging.info(f"  GSP {int(gsp_id)}: {mean_cf_by_gsp[gsp_idx]:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Analyse PV GSP data from .zarr dataset')
    parser.add_argument('zarr_path', help='Path to the zarr dataset ( GCS)')

    args = parser.parse_args()

    ds = load_data(args.zarr_path)

    basic_stats(ds)
    gsp_analysis(ds)
    temporal_analysis(ds)
    spatial_analysis(ds)
    correlation_analysis(ds)
    capacity_factor_analysis(ds)

    logging.info("\nAnalysis complete.")

    if args.zarr_path.startswith("gs://"):
        local_zarr_path = os.path.join(tempfile.gettempdir(), "temp.zarr")
        if os.path.exists(local_zarr_path):
            shutil.rmtree(local_zarr_path)

if __name__ == "__main__":
    main()
