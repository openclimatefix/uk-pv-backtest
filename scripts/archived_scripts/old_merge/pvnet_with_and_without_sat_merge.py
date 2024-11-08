import pandas as pd

df_pvnet_sat = pd.read_csv("../../data/pvnet_sum_model_w_sat_prob.csv")
df_pvnet_no_sat = pd.read_csv("../../data/pvnet_sum_model_without_sat_prob.csv")


# First, let's check for any similar 'Init Time' between the two dataframes
common_init_times = pd.merge(df_pvnet_sat[["Init Time"]], df_pvnet_no_sat[["Init Time"]], on="Init Time", how="inner")

if not common_init_times.empty:
    print(f"Found {len(common_init_times)} common 'Init Time' entries.")
    print(common_init_times)
else:
    print("No common 'Init Time' entries found.")

# Assuming we want to merge regardless of common 'Init Time'
# Concatenate the two dataframes
merged_df = (
    pd.concat([df_pvnet_sat, df_pvnet_no_sat]).drop_duplicates().sort_values(by="Init Time").reset_index(drop=True)
)

print("Dataframes merged successfully.")

merged_df.to_csv("../../data/pvnet_sum_model_combined_270324.csv", index=False)
