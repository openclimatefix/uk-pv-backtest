"""Analyses PV generation data from a CSV file.

Loads data, performs analyses, and prints a summary.
"""

import argparse
import logging
import numpy as np
import pandas as pd
from google.cloud import storage
import io
from typing import Optional


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(csv_path: str) -> Optional[pd.DataFrame]:
    """Loads data, handling local, URL, and GCS paths.

    Args:
        csv_path: Path to the CSV file (local, https, or GCS).

    Returns:
        A pandas DataFrame, or None if loading fails.
    """
    logging.info(f"Loading data from {csv_path}...")
    try:
        if csv_path.startswith("https://storage.cloud.google.com/"):
            import requests
            response = requests.get(csv_path)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text))
        elif csv_path.startswith("gs://"):
            path_parts = csv_path.replace("gs://", "").split("/")
            bucket_name, blob_path = path_parts[0], "/".join(path_parts[1:])
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            content = blob.download_as_bytes()
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_csv(csv_path)

        df['start_datetime_utc'] = pd.to_datetime(df['start_datetime_utc'])
        df['end_datetime_utc'] = pd.to_datetime(df['end_datetime_utc'])
        df['date'] = df['end_datetime_utc'].dt.date
        df['hour'] = df['end_datetime_utc'].dt.hour
        df['month'] = df['end_datetime_utc'].dt.month
        df['year'] = df['end_datetime_utc'].dt.year
        df['day_of_year'] = df['end_datetime_utc'].dt.dayofyear
        df['day_name'] = df['end_datetime_utc'].dt.day_name()
        df['capacity_utilisation'] = df['generation_mw'] / df['capacity_mwp'] * 100
        return df
    except Exception as e:
        logging.exception(f"Error loading data: {e}")
        return None


def analyse_data(df: pd.DataFrame) -> None:
    """Analyses data and prints key findings.

    Args:
        df: The pandas DataFrame containing the PV data.

    Returns:
        None (prints the analysis results).
    """
    if df is None or df.empty:
        logging.error("DataFrame is empty. Exiting.")
        return

    print("=== PV Data Analysis ===")
    print(f"Data from {df['start_datetime_utc'].min()} to {df['end_datetime_utc'].max()}")
    print(f"Total Data Points: {len(df):,}")

    print("\n--- Basic Stats ---")
    print(df[['generation_mw', 'capacity_utilisation']].describe().T)
    year_diff = df['year'].max() - df['year'].min()
    if year_diff > 0:
      growth = (df['installedcapacity_mwp'].iloc[-1] / df['installedcapacity_mwp'].iloc[0]) ** (1 / year_diff) - 1
      print(f"Installed Capacity Growth (Annual): {growth:.2%}")
    else:
      print("Installed Capacity Growth (Annual): Cannot be calculated (single year data).")

    print("\n--- Yearly Analysis ---")
    yearly = df.groupby('year').agg({
        'generation_mw': ['sum', 'mean'],
        'capacity_utilisation': 'mean',
        'installedcapacity_mwp': 'max'
    })
    yearly.columns = ['Gen_Sum_MW', 'Gen_Mean_MW', 'Avg_Cap_Utilisation', 'Max_Installed_Cap']
    yearly['Gen_Sum_GWh'] = yearly['Gen_Sum_MW'] * 0.5 / 1000
    print(yearly[['Gen_Sum_GWh', 'Avg_Cap_Utilisation', 'Max_Installed_Cap']])

    print("\n--- Year on Year ---")
    yoy = pd.DataFrame()
    yoy["gen_change_gwh"] = yearly['Gen_Sum_GWh'].diff()
    yoy["gen_pct_change"] = yearly['Gen_Sum_GWh'].pct_change() * 100
    yoy["cap_change_mwp"] = yearly['Max_Installed_Cap'].diff()
    yoy["cap_pct_change"] = yearly['Max_Installed_Cap'].pct_change() * 100
    print(yoy)

    print("\n--- Monthly Analysis ---")
    monthly = df.groupby('month').agg({'generation_mw': 'mean', 'capacity_utilisation': 'mean'})
    best_month = monthly['capacity_utilisation'].idxmax()
    worst_month = monthly['capacity_utilisation'].idxmin()
    print(f"Best Month (Avg Capacity Utilisation): {best_month} ({monthly['capacity_utilisation'].max():.2f}%)")
    print(f"Worst Month (Avg Capacity Utilisation): {worst_month} ({monthly['capacity_utilisation'].min():.2f}%)")

    print("\n--- Daily Patterns ---")
    daily_profile = df.groupby('hour')['generation_mw'].mean()
    peak_hour = daily_profile.idxmax()
    print(f"Peak Generation Hour (UTC): {peak_hour:02d}:00")
    daylight_df = df[df['generation_mw'] > 0]
    if not daylight_df.empty:
      print(f"Avg Daylight Hours: {len(daylight_df) * 0.5 / (df['date'].nunique()):.2f}")
    else:
      print("Avg Daylight Hours: No daylight hours detected.")


def main() -> None:
    """Main function to parse arguments and run the analysis."""
    parser = argparse.ArgumentParser(description='Analyses PV GSP data.')
    parser.add_argument('csv_path', help='Path to the CSV file (local, https, or GCS)')
    args = parser.parse_args()
    df: Optional[pd.DataFrame] = load_data(args.csv_path)
    if df is not None:
        analyse_data(df)


if __name__ == "__main__":
    main()
