
[![contributors badge](https://img.shields.io/github/contributors/openclimatefix/uk-pv-backtest?color=FFFFFF)](    https://github.com/openclimatefix/uk-pv-backtest/graphs/contributors) 
[![ease of contribution: medium](https://img.shields.io/badge/ease%20of%20contribution:%20medium-f4900c)](https://github.com/openclimatefix/ocf-meta-repo?tab=readme-ov-file#overview-of-ocfs-nowcasting-repositories)


# Backtest Formatting For UK National Solar Forecasting

This repo is used for formatting and verification of backtests from national solar forecasting models at Open Climate Fix.

## Running PVNet backtests

###  Setting up the model and data configuration

National solar forecasting backtests can be run using OCFs PVNet and NationalXG models. In the [PVNet repo](https://github.com/openclimatefix/PVNet), under scripts, there is the file `gsp_run_backtest.py`. This script can be used to run the backtests by setting the models and the dates ranges to use. 

For PVNet there is one model to run the GSP level forecasts and another model, called the summation model, which is used to aggregate the GSP level forecasts to a national level. Each of these models checkpoints can be downloaded locally before running the backtest or can be streamed in from Hugging Face.

The model requires a specific configuration file called `data_config.yml`. This file defines:
- The location of input data
- Input parameters like time lags 
- How much historical data to include
- Other model-specific data settings

**The configuration file must exactly match the settings used during model training for the backtest to run correctly.**

### Executing the Backtest Process

As backtests can take a long time to run, it is best to used a environment like `tmux` to run the backtests. This allows you to keep the job running even if the SSH connection is lost.

After installing `tmux` you can create a new session with:
`tmux new -s [SESSION_NAME]`

Then activate the appropriate conda environment to run the backtest. Once you have created and are inside a `tmux` session you can run the backtest with:
`python run_backtest.py`

The progress of the backtest can be viewed by reconnecting to the `tmux` session with:
`tmux attach -n [SESSION_NAME]`

It can be useful to inspect how much of the machines resources are being used via `top` or `htop`. This shows the CPU and RAM usage, which is useful when optimising the number of workers and batches.


### Additional notes for running backtests.

- For PVNet backtests, two sets of models are blended together to produce the final forecast. The first is for **Intraday forecasts (0-8 hours)**, and the second is for **Day Ahead forecasts (0-36 hours)**. The number of workers, batches and overall configuration will be different for each model.
- If an error appears such as `terminate called without an active exception`, this is likely due to memory issues. This can be fixed by reducing the number of workers or batches or by increasing the machines resources.
- It is useful to run small experimental backtests and compare the forecast accuracy to previous forecasts using the `compare_forecast_mae.ipynb` notebook. Previous backtest data can be found in the google storage bucket under `solar-pv-nowcasting-data/backtest/`. This helps to validate things are as expected before kicking off a larger backtest.


## Formatting of Forecasts produced by PVNet

For PVNet, processing and formatting scripts are found in `/scripts/pvnet_prob/` and consists of the 4 steps below:

1. Compile the raw PVNet files to a zarr file (`compile_raw_files.py`)
2. Filter the data for GSP 0 (National) and the quantiles to output as a single csv (`filter_zarr_to_csv.py`)
3. Merge and blend the Intraday and Day Ahead forecasts to produce a single csv (`merge_and_blend.py`)
4. Add PVLive installed capacity and format the final forecast file (`format_forecast.py`)


### Compiling Raw PVNet Files
PVNet produces a single netcdf file (.nc) per initialisation time. These files need to be combined together. The script to do this is called `compile_raw_files.py`. This will produce a zarr file containing the data.

The `filter_zarr_to_csv.py` script turns the data from a zarr into a csv, keeping just the national forecast rather than the GSP level forecasts. This needs to be performed for the Intraday and Dayahead forecasts separately.

Once the files are in the correct format, the `merge_and_blend_prob.py` script can be used. This merges the two datasets together and blends the forecasts together based on defined weightings at different forecast horizons in the script.

The data then needs to run through a last formatting script called, `format_forecast.py`. This script adds the PVLive installed capacity and outputs the final forecast file.


#### Additional notes on compiling forecasts

- Legacy archive scripts from previous projects have been included in this repo and can be found in `/scripts/archived_scripts/` folder.
- Previously the combination of the day ahead forecasts from NationalXG and Intraday forecasts from PVNet were blended together. Now only PVNet is used however, previous use of NationalXG has determined some of the formatting choices made in the code.
- The formatting functions have been kept modular.


#### National XG formatting

Scripts have been written for interpolating hourly forecasts to half hourly `interpolate_30min.py` and for unnormalising forecasts using the installed capacity for PVLive `unnorm_forecast.py`.

## Post Formatting

Notebooks for verifying the data and comparing forecasts is found in `/notebooks/`
- `check_blending.ipynb` can be used to verify the blending of the forecasts.
- `check_forecast_consistency.ipynb` can be used to check data quality.
- `compare_forecast_mae.ipynb` can be used to compare the error of the forecasts to previous forecasts and models. Previous forecasts can be moved to the `/data/compare_forecasts` folder to use this notebook.

### Additional scripts

Check for missing data in the backtest using the `missing_data.py` file. This script checks the data for gaps in the forecasts and outputs a csv detailing the size and start of the gaps.

To name the file in the standardised format, use the `rename_forecast_file.py` script. For model version numbers, the [pvnet_app](https://github.com/openclimatefix/uk-pvnet-app) version number is used.


## Uploading Data to Google Storage

After running a backtest, the raw data can be uploaded to Google Storage. The `gsutil` command line tool can be used:

`gsutil -m cp -r [LOCAL_FILE_PATH] gs://[BUCKET_NAME]/[OBJECT_NAME]`

The `-m` flag enables parallel multi-threading, allowing multiple files to be transferred simultaneously which significantly speeds up the transfer.

Data can then be downloaded onto another machine for distribution.

## Contributing and community

- PR's are welcome! See the [Organisation Profile](https://github.com/openclimatefix) for details on contributing
- Find out about our other projects in the [OCF Meta Repo](https://github.com/openclimatefix/ocf-meta-repo)
- Check out the [OCF blog](https://openclimatefix.org/blog) for updates
- Follow OCF on [LinkedIn](https://uk.linkedin.com/company/open-climate-fix)

---

*Part of the [Open Climate Fix](https://github.com/orgs/openclimatefix/people) community.*

[![OCF Logo](https://cdn.prod.website-files.com/62d92550f6774db58d441cca/6324a2038936ecda71599a8b_OCF_Logo_black_trans.png)](https://openclimatefix.org)
