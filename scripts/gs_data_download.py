"""A alternative method to download data from GCS instead of doing everything in the command line using gsutil"""

import subprocess


def download_from_gs(gs_path, local_path):
    """
    Download a file or directory from Google Cloud Storage (gs) to a local path.

    This function is capable of downloading directories, which is necessary for zarr formats.

    Args:
    gs_path (str): The gs path to the file or directory to download.
    local_path (str): The local path where to save the downloaded file or directory.
    """
    import os

    destination_dir = local_path
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    try:
        # For zarr formats, ensure directories are handled correctly
        if gs_path.endswith(".zarr/"):
            subprocess.check_call(["gsutil", "-m", "cp", "-r", gs_path, local_path])
        else:
            subprocess.check_call(["gsutil", "-m", "cp", gs_path, local_path])
        print(f"Successfully downloaded {gs_path} to {local_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {gs_path}. Error: {e}")


if __name__ == "__main__":
    # Define the Google Storage path and the local path
    # gs_path = "gs://solar-pv-nowcasting-data/backtest/national_xg_2018_2022/full_predictions_cross_validation_v2.csv"
    # gs_path = "gs://solar-pv-nowcasting-data/backtest/drs_backtest/model_ensemble.nc" - previous backtest
    gs_path_w_sat = "gs://solar-pv-nowcasting-data/backtest/drs_backtest/with_sat_ensemble_mean.zarr/"  # backtest valid as of 270324
    gs_path_wo_sat = "gs://solar-pv-nowcasting-data/backtest/drs_backtest/without_sat_ensemble_mean.zarr/"

    local_path_w_sat = "../../data/drs_backtest_270324_with_sat.zarr/"
    local_path_wo_sat = "../../data/drs_backtest_270324_without_sat.zarr/"

    # Download the data
    download_from_gs(gs_path_w_sat, local_path_w_sat)
    download_from_gs(gs_path_wo_sat, local_path_wo_sat)
