"""Analyse data from .csv file"""

import argparse
import io
import logging
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from google.cloud import storage


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(csv_path):
    """Load the PV data CSV and prepare it for analysis."""
    logging.info(f"Loading data from {csv_path}...")

    if csv_path.startswith("https://storage.cloud.google.com/"):
        import requests
        try:
            response = requests.get(csv_path)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text))
        except Exception as e:
            logging.exception(f"Error downloading from URL: {e}")
            return None

    elif csv_path.startswith("gs://"):
        path_parts = csv_path.replace("gs://", "").split("/")
        bucket_name = path_parts[0]
        blob_path = "/".join(path_parts[1:])

        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            content = blob.download_as_bytes()
            df = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            logging.exception(f"Error accessing GCS: {e}")
            return None
    else:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logging.exception(f"Error reading file: {e}")
            return None

    if df is None or len(df) == 0:
        logging.error("Error: Loaded DataFrame is empty or None")
        return None

    df['start_datetime_utc'] = pd.to_datetime(df['start_datetime_utc'])
    df['end_datetime_utc'] = pd.to_datetime(df['end_datetime_utc'])

    df['date'] = df['end_datetime_utc'].dt.date
    df['hour'] = df['end_datetime_utc'].dt.hour
    df['month'] = df['end_datetime_utc'].dt.month
    df['year'] = df['end_datetime_utc'].dt.year
    df['day_of_year'] = df['end_datetime_utc'].dt.dayofyear
    df['day_name'] = df['end_datetime_utc'].dt.day_name()

    df['capacity_factor'] = df['generation_mw'] / df['capacity_mwp'] * 100

    logging.info(f"Loaded {len(df):,} records from {df['start_datetime_utc'].min()} to {df['end_datetime_utc'].max()}")
    return df


def print_separator():
    """Print a separator line."""
    logging.info("\n" + "="*80 + "\n")


def basic_stats(df):
    """Calculate and print basic statistics."""
    print_separator()
    logging.info("BASIC STATISTICS")
    print_separator()

    logging.info(f"Dataset Summary:")
    logging.info(f"Time period: {df['start_datetime_utc'].min()} to {df['end_datetime_utc'].max()}")
    logging.info(f"Total records: {len(df):,}")
    logging.info(f"Data points per day: {len(df) / (df['end_datetime_utc'].max() - df['start_datetime_utc'].min()).days:.1f}")
    logging.info(f"Years covered: {', '.join(map(str, sorted(df['year'].unique())))}")

    missing_values = df.isna().sum()
    if missing_values.sum() > 0:
        logging.info("\nMissing Values:")
        for col, count in missing_values.items():
            if count > 0:
                logging.info(f"  {col}: {count:,} ({count/len(df)*100:.2f}%)")
    else:
        logging.info("\nNo missing values detected.")

    logging.info("\nGeneration (MW):")
    logging.info(f"  Mean: {df['generation_mw'].mean():.2f}")
    logging.info(f"  Median: {df['generation_mw'].median():.2f}")
    logging.info(f"  Std Dev: {df['generation_mw'].std():.2f}")
    logging.info(f"  Min: {df['generation_mw'].min():.2f}")
    logging.info(f"  Max: {df['generation_mw'].max():.2f} (on {df.loc[df['generation_mw'].idxmax(), 'end_datetime_utc']})")

    logging.info("\nCapacity (MWp):")
    logging.info(f"  Initial: {df.loc[df['start_datetime_utc'].idxmin(), 'capacity_mwp']:.2f}")
    logging.info(f"  Final: {df.loc[df['end_datetime_utc'].idxmax(), 'capacity_mwp']:.2f}")
    logging.info(f"  Min: {df['capacity_mwp'].min():.2f}")
    logging.info(f"  Max: {df['capacity_mwp'].max():.2f}")
    logging.info(f"  Growth: {df['capacity_mwp'].iloc[-1] - df['capacity_mwp'].iloc[0]:.2f} MWp")
    logging.info(f"  Annual growth rate: {(df['capacity_mwp'].iloc[-1]/df['capacity_mwp'].iloc[0])**(1/(df['year'].max()-df['year'].min())) - 1:.2%}")

    logging.info("\nInstalled Capacity (MWp):")
    logging.info(f"  Initial: {df.loc[df['start_datetime_utc'].idxmin(), 'installedcapacity_mwp']:.2f}")
    logging.info(f"  Final: {df.loc[df['end_datetime_utc'].idxmax(), 'installedcapacity_mwp']:.2f}")
    logging.info(f"  Min: {df['installedcapacity_mwp'].min():.2f}")
    logging.info(f"  Max: {df['installedcapacity_mwp'].max():.2f}")
    logging.info(f"  Growth: {df['installedcapacity_mwp'].iloc[-1] - df['installedcapacity_mwp'].iloc[0]:.2f} MWp")
    logging.info(f"  Annual growth rate: {(df['installedcapacity_mwp'].iloc[-1]/df['installedcapacity_mwp'].iloc[0])**(1/(df['year'].max()-df['year'].min())) - 1:.2%}")

    logging.info("\nCapacity Factor (%):")
    logging.info(f"  Mean: {df['capacity_factor'].mean():.2f}%")
    logging.info(f"  Median: {df['capacity_factor'].median():.2f}%")
    logging.info(f"  Std Dev: {df['capacity_factor'].std():.2f}%")
    logging.info(f"  Min: {df['capacity_factor'].min():.2f}%")
    logging.info(f"  Max: {df['capacity_factor'].max():.2f}% (on {df.loc[df['capacity_factor'].idxmax(), 'end_datetime_utc']})")

    daylight_df = df[df['generation_mw'] > 0]
    logging.info("\nDaylight Hours (Generation > 0):")
    logging.info(f"  Average daylight hours: {len(daylight_df) * 0.5 / (df['date'].nunique()):.2f} hours/day")
    logging.info(f"  Percentage of time with generation: {len(daylight_df) / len(df) * 100:.2f}%")


def yearly_analysis(df):
    """Analyse and print yearly statistics."""
    print_separator()
    logging.info("YEARLY ANALYSIS")
    print_separator()

    yearly = df.groupby('year').agg({
        'generation_mw': ['mean', 'std', 'max', 'min'],
        'capacity_factor': ['mean', 'std', 'max', 'min'],
        'capacity_mwp': ['mean', 'max'],
        'installedcapacity_mwp': ['mean', 'max']
    })

    gen_by_year = df.groupby('year')['generation_mw'].sum() * 0.5 / 1000  # Convert to GWh
    yearly[('generation_mw', 'total_gwh')] = gen_by_year
    yearly[('generation_mw', 'total_twh')] = gen_by_year / 1000

    logging.info("Yearly Statistics:\n")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 180)
    logging.info(yearly)

    logging.info("\nYear-over-Year Changes:")
    for year in range(df['year'].min() + 1, df['year'].max() + 1):
        prev_year = year - 1
        if prev_year in yearly.index and year in yearly.index:
            gen_change = yearly.loc[year, ('generation_mw', 'total_gwh')] - yearly.loc[prev_year, ('generation_mw', 'total_gwh')]
            gen_pct_change = gen_change / yearly.loc[prev_year, ('generation_mw', 'total_gwh')] * 100
            cap_change = yearly.loc[year, ('installedcapacity_mwp', 'max')] - yearly.loc[prev_year, ('installedcapacity_mwp', 'max')]
            cap_pct_change = cap_change / yearly.loc[prev_year, ('installedcapacity_mwp', 'max')] * 100

            logging.info(f"  {prev_year} → {year}:")
            logging.info(f"    Generation: {gen_change:.2f} GWh ({gen_pct_change:+.2f}%)")
            logging.info(f"    Capacity: {cap_change:.2f} MWp ({cap_pct_change:+.2f}%)")

    seasons = {
        1: 'Winter', 2: 'Winter', 3: 'Spring',
        4: 'Spring', 5: 'Spring', 6: 'Summer',
        7: 'Summer', 8: 'Summer', 9: 'Autumn',
        10: 'Autumn', 11: 'Autumn', 12: 'Winter'
    }
    df['season'] = df['month'].map(seasons)

    seasonal = df.groupby(['year', 'season']).agg({
        'generation_mw': 'mean',
        'capacity_factor': 'mean'
    }).unstack()

    logging.info("\nSeasonal Generation by Year (Average MW):")
    logging.info(seasonal['generation_mw'])

    logging.info("\nSeasonal Capacity Factor by Year (%):")
    logging.info(seasonal['capacity_factor'])


def monthly_analysis(df):
    """Analyse and print monthly statistics."""
    print_separator()
    logging.info("MONTHLY ANALYSIS")
    print_separator()

    monthly_stats = df.groupby('month').agg({
        'generation_mw': ['mean', 'std', 'max'],
        'capacity_factor': ['mean', 'std', 'max']
    })

    month_names = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April',
        5: 'May', 6: 'June', 7: 'July', 8: 'August',
        9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }

    monthly_stats.index = [month_names[m] for m in monthly_stats.index]

    logging.info("Monthly Statistics (Averaged Across All Years):\n")
    logging.info(monthly_stats)

    best_month = monthly_stats[('capacity_factor', 'mean')].idxmax()
    worst_month = monthly_stats[('capacity_factor', 'mean')].idxmin()

    logging.info(f"\nBest month for solar generation: {best_month} (Avg. capacity factor: {monthly_stats.loc[best_month, ('capacity_factor', 'mean')]:.2f}%)")
    logging.info(f"Worst month for solar generation: {worst_month} (Avg. capacity factor: {monthly_stats.loc[worst_month, ('capacity_factor', 'mean')]:.2f}%)")

    ratio = monthly_stats.loc[best_month, ('generation_mw', 'mean')] / monthly_stats.loc[worst_month, ('generation_mw', 'mean')]
    logging.info(f"Ratio of generation between best and worst months: {ratio:.2f}")

    logging.info("\nAverage Generation by Hour for Each Month (MW):")
    hourly_monthly = df.groupby(['month', 'hour'])['generation_mw'].mean().unstack()
    hourly_monthly.index = [month_names[m] for m in hourly_monthly.index]
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 180)
    logging.info(hourly_monthly)


def daily_patterns(df):
    """Analyse and print daily generation patterns."""
    print_separator()
    logging.info("DAILY PATTERNS")
    print_separator()

    day_of_week = df.groupby('day_name').agg({
        'generation_mw': ['mean', 'std'],
        'capacity_factor': ['mean', 'std']
    })

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_of_week = day_of_week.reindex(day_order)

    logging.info("Generation by Day of Week:")
    logging.info(day_of_week)

    logging.info("\nAverage Daily Generation Profile (MW):")
    daily_profile = df.groupby('hour')['generation_mw'].mean()
    for hour, gen in daily_profile.items():
        logging.info(f"  {hour:02d}:00-{hour+1:02d}:00: {gen:.2f} MW")

    peak_hour = daily_profile.idxmax()
    logging.info(f"\nPeak generation occurs around {peak_hour:02d}:00-{peak_hour+1:02d}:00 UTC")

    daily_first_gen = df[df['generation_mw'] > 0].groupby('date')['hour'].min()
    daily_last_gen = df[df['generation_mw'] > 0].groupby('date')['hour'].max()

    logging.info(f"\nApproximate sunrise time (first generation > 0):")
    logging.info(f"  Average: {daily_first_gen.mean():.2f}:00 UTC")
    logging.info(f"  Earliest: {daily_first_gen.min()}:00 UTC")
    logging.info(f"  Latest: {daily_first_gen.max()}:00 UTC")

    logging.info(f"\nApproximate sunset time (last generation > 0):")
    logging.info(f"  Average: {daily_last_gen.mean():.2f}:00 UTC")
    logging.info(f"  Earliest: {daily_last_gen.min()}:00 UTC")
    logging.info(f"  Latest: {daily_last_gen.max()}:00 UTC")

    daylight_hours = df.groupby(['year', 'month']).apply(
        lambda x: len(x[x['generation_mw'] > 0]) * 0.5
    ).unstack()

    logging.info("\nAverage Daylight Hours by Month and Year:")
    logging.info(daylight_hours)


def main():
    parser = argparse.ArgumentParser(description='Analyse PV GSP data from a CSV file')
    parser.add_argument('csv_path', help='Path to the CSV file (local, https, or GCS)')
    args = parser.parse_args()
    df = load_data(args.csv_path)

    if df is not None:
        basic_stats(df)
        yearly_analysis(df)
        monthly_analysis(df)
        daily_patterns(df)
        logging.info("\nAnalysis complete.")

if __name__ == "__main__":
    main()
