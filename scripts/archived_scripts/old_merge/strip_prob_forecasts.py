"""Simple script to strip the probability forecasts from the XGBoost predictions."""

import pandas as pd

xgb_df = pd.read_csv("../../data/full_predictions_cross_validation_v4_prob_full.csv")

xgb_df = xgb_df.loc[:, ~xgb_df.columns.str.startswith("p10")]
xgb_df = xgb_df.loc[:, ~xgb_df.columns.str.startswith("p90")]

xgb_df.to_csv("../../data/full_predictions_cross_validation_v4_without_prob_full.csv", index=False)
