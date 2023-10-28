from price_impact_comp import *

bin_sample_path = u"/content/drive/My Drive/Data/binSamples/"
fill_sample_path = u"/content/drive/My Drive/Data/fillSamples/"
result_path = u"/content/drive/My Drive/Data/inter_results/"
year = "2019"

time_unit = 10 # 10 seconds is the time interval between any two adjacent samples in the binned data
half_life = 3600
beta = np.log(2) / half_life
explanation_horizon = 60
explanation_horizon_periods = explanation_horizon // time_unit
model_types = ["reduced_form", "linear", "sqrt"]
cutoff_time = "10:00:00"

### Store net traded volume, price and intraday volume (exponential weighted average of the abs traded volume)

for month_int in range(1, 13):
    month = '%02d' % month_int
    print(month)
    filename = "bin%s%s.csv" % (year, month)
    bin_sample_df = pd.read_csv(bin_sample_path+filename)
    bin_sample_df["datetime"] = pd.to_datetime(bin_sample_df["date"] + " " + bin_sample_df["time"], format="%Y%m%d %H:%M:%S")

    traded_volume_df = bin_sample_df[["stock", "date", "trade", "time"]].pivot(index=["stock", "date"], columns=["time"])["trade"].fillna(0).astype(int)
    filename = f"inter_results_volume_{year}{month}.csv"
    traded_volume_df.to_csv(result_path+filename)
    px_df = bin_sample_df[["stock", "date", "midEnd", "time"]].pivot(index=["stock", "date"], columns=["time"])["midEnd"].fillna(method="ffill", axis="columns").fillna(method="bfill", axis="columns")
    filename = f"inter_results_px_{year}{month}.csv"
    px_df.to_csv(result_path+filename)
    intraday_volume_df = compute_intraday_volume(traded_volume_df, half_life, time_unit)
    filename = f"inter_results_intraday_volume_{year}{month}.csv"
    intraday_volume_df.to_csv(result_path+filename)

### Pre-compute price vol & ADV

monthly_stock_info_dfs = []
for month_int in range(1, 13):
    month = '%02d' % month_int
    print(month)
    filename = "bin%s%s.csv" % (year, month)
    bin_sample_df = pd.read_csv(bin_sample_path+filename)
    monthly_stock_info_dfs.append(compute_vol_volume(bin_sample_df))

stock_info_df = pd.concat(monthly_stock_info_dfs).reset_index(drop=True)
filename = f"inter_results_pre_{year}all.csv"
stock_info_df.to_csv(result_path+filename, index=False)

scaling_df = compute_rolling_stats(stock_info_df, num_rolling_days=20)
filename = f"SP500_{year}_scaling_factors.csv"
scaling_df.to_csv(result_path+filename, index=False)

### Pre-compute impact states

filename = f"SP500_{year}_scaling_factors.csv"
scaling_df = pd.read_csv(result_path+filename)

for month_int in range(1, 13):
    month = '%02d' % month_int
    print(month)
    filename = f"inter_results_volume_{year}{month}.csv"
    traded_volume_df = pd.read_csv(result_path+filename, index_col=[0,1])
    filename = f"inter_results_px_{year}{month}.csv"
    px_df = pd.read_csv(result_path+filename, index_col=[0,1])
    filename = f"inter_results_intraday_volume_{year}{month}.csv"
    intraday_volume_df = pd.read_csv(result_path+filename, index_col=[0,1])

    monthly_scaling_factor = scaling_df.set_index(["stock", "date"]).loc[traded_volume_df.index]

    for model_type in model_types:
        cumulative_impact = impact_state(traded_volume_df, monthly_scaling_factor, half_life, model_type, time_unit, intraday_volume_df)
        reg_stat_df = impact_regression_statistics(cumulative_impact, explanation_horizon_periods, px_df, cutoff_time)
        summary_df = reg_stat_df.groupby(["stock", "date"]).sum().reset_index()
        filename = f"inter_results_{model_type}_{year}{month}.csv"
        summary_df.to_csv(result_path+filename, index=False)

### Estimate price impacts

for model_type in model_types:
    reg_stat_dfs = []
    for month_int in range(1, 13):
        month = '%02d' % month_int
        filename = f"inter_results_{model_type}_{year}{month}.csv"
        reg_stat_dfs.append(pd.read_csv(result_path+filename))
    daily_stock_reg_info_df = pd.concat(reg_stat_dfs).reset_index(drop=True)
    daily_stock_reg_info_df["date"] = pd.to_datetime(daily_stock_reg_info_df["date"])
    regression_res_dfs = []
    for in_sample_month in range(1, 12):
        regression_res_dfs.append(ridge_regression_result(daily_stock_reg_info_df, in_sample_month))
    regression_res_df = pd.concat(regression_res_dfs).reset_index(drop=True)
    filename = f"{model_type}_impact_coefficients.csv"
    regression_res_df.to_csv(result_path+filename)
