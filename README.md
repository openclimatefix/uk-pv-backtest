# Backtest Formatting

This repo contains scripts and notebooks for formatting and verifying the backtest data produced by national solar forecasting models from Open Climate Fix.

## How to run backtests in the first place!

National backtests can be run using our latest PVNet model and NationalXG model. In the PVNet repo, under scripts, there is a python file called `gsp_run_backtest.py`. This script can be used to run the backtests by setting the models and the dates you to use. For PVNet there is a model to run the GSP level forecast and another model, called the summation model, which is used to aggregate the GSP level forecasts to a national level. Each of these model checkpoints should be downloaded locally before running the backtest.

Additionally, specific configuration will also be required dependent on what inputs the trained model expects. The data_config.yml file is used to store the location of the data to use as well as the settings for data input parameters such as the lag for each of the inputs and the amount of history provided.

As backtests can take a long time to run, it is best to used a environment like `tmux` to run the backtests. This allows you to keep the job running even if the SSH connection is lost.

After installing tmux you can create a new session with `tmux new -s [SESSION_NAME]`.

Then activate, the appropriate environment to run the backtest. Once you have created a tmux session you can run the backtest with `python run_backtest.py`.

The progress of the backtest can be checked again used `tmux attach -n [SESSION_NAME]`

It can be useful to inspect how much of the machines resources are being used via `top` or `htop`. This shows the memory usage of the CPU and RAM, which is useful when optimising the number of workers and batches.


### Additional notes for running backtests.

- For PVNet, two sets of models are belended together to produce the final forecast. The first is for the Intraday forecasts (0-8 hours), and the second is for the Day Ahead forecast (0-36 hours). The number of workers, batches and overall configuration will be different for each of these. The day ahead forecasts will take the most time to run, due to the length of the forecasts.
- If an error appears such as `terminate called without an active exception`, this is likely due to memory issues. This can fix this by reducing the number of workers and batches or by increasing the machines resources.
- It is useful to run small test backtests and compare the forecast accuracy to previous forecasts using the `compare_forecast_error.ipynb` notebook. Previous backtest data can be found on the google storage bucket under `solar-pv-nowcasting-data/backtest/`. This helps to validate things are as expected before kicking off a larger backtest.


## Formatting of Forecasts produced by PVNet

For PVNet, the processing and formatting scripts are found in `/scripts/pvnet_prob/` and consists of the 4 steps below:

1. Compile the raw PVNet files to a zarr file (`compile_raw_files.py`)
2. Filter the data for GSP 0 (National) and the quantiles to output as a single csv (`filter_zarr_to_csv.py`)
3. Merge and blend the Intraday and Day Ahead forecasts to produce a single csv (`merge_and_blend.py`)
4. Add PVLive installed capacity and format the final forecast file (`format_forecast.py`)


### Compiling Raw PVNet Files
PVNet produces a single netcdf file (.nc) per initialisation time. These files need to be combined together. The script to do this is called `compile_raw_files.py`. This will produce a zarr file containing all the data.

The `filter_zarr_to_csv.py` script turns the data from a zarr into a csv, keeping just the national forecast rather than the GSP level forecasts. This needs to be performed for the Intraday and Dayahead forecasts separately.

Once the files are in the correct format, the `merge_and_blend_prob.py` script can be used. This merges the two datasets together and blends the forecasts together based on defined weightings in the script.

The file then runs through a last formatting script called, `format_forecast.py`. This script adds the PVLive installed capacity and outputs the final forecast file.


#### Additional notes on compiling forecasts

- Legacy archive scripts from previous projects have been included in this repo and can be found in `/scripts/archived_scripts/` folder.
- Previously the combination of the day ahead forecasts from NationalXG and intraday forecasts from PVNet were blended together. Now only PVNet is used however, previous use of NationalXG has determined some of the formatting choices made in the code.
- The formatting functions have been kept modular to allow for ease if new models with different formats are to be used and data to be checked along the way.


### National XG formatting

Previously scripts have been written for interpolating hourly forecasts to half hourly `interpolate_30min.py`. And also for unnormalising forecasts using the installed capacity for PVLive `unnorm_forecast.py`.

## Post Formatting

Notebooks for checking the data and comparing the forecasts are found in `/notebooks/`
- `check_blending.ipynb` can be used to verify the blending of the forecasts.
- `check_forecast_consistency.ipynb` can be used to validate various aspects of the forecast data.
- `compare_forecast_error.ipynb` can be used to compare the error of the forecasts to previous forecasts/different models. Previous forecasts can be moved to the `/data/compare_forecasts` folder to use this notebook.

### Additional scripts

Check for missing data using the `missing_data.py` file. This script checks the data for gaps in the forecasts and outputs a csv detailing the size and start of the gaps.

To name the file in the standardised format, use the `rename_forecast_file.py` script. For model version numbers, the pvnet_app version number is used.


## Uploading Data to Google Storage

It is best to upload the raw backtest data to a Google storage as you will likely be able to process data much faster due to high bandwidth. You can do this using `gsutil`

`gsutil -m cp -r [LOCAL_FILE_PATH] gs://[BUCKET_NAME]/[OBJECT_NAME]`

Don't forget to use `-m` to significantly speed up the transfer.

Data can then be downloaded onto another machine for processing.
