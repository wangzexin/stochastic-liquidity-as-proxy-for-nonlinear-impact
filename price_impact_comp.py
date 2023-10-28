import numpy as np
import pandas as pd

def compute_vol_volume(bin_sample_df):
    traded_volume_df = bin_sample_df[["stock", "date", "trade", "time"]].pivot(index=["stock", "date"], columns=["time"])["trade"].fillna(0).astype(int)
    px_df = bin_sample_df[["stock", "date", "midEnd", "time"]].pivot(index=["stock", "date"], columns=["time"])["midEnd"].fillna(method="ffill", axis="columns").fillna(method="bfill", axis="columns")
    return pd.DataFrame({
        "px_vol" : px_df.pct_change(1, axis="columns").std(axis="columns") * np.sqrt(2340),
        "volume" : traded_volume_df.abs().sum(axis="columns"),
    }).reset_index()

def compute_intraday_volume(traded_volume_df, half_life, time_unit):
    beta = np.log(2) / half_life
    decay_factor = np.exp(-beta * time_unit)
    pre_ewm = traded_volume_df.abs().copy()
    pre_ewm.iloc[:, 1:] /= (1 - decay_factor)
    return pre_ewm.ewm(alpha=1-decay_factor, adjust=False, axis="columns").mean() * traded_volume_df.shape[1]

def compute_rolling_stats(stock_info_df, num_rolling_days):
    stacked_info = stock_info_df.pivot(index="date", columns="stock", values=["px_vol", "volume"]).rolling(num_rolling_days).mean().shift(1)
    return pd.DataFrame({
        "px_vol" : stacked_info["px_vol"].unstack(),
        "volume" : stacked_info["volume"].unstack(),
    }).reset_index()

def reduced_form_impact_state(traded_volume_df, monthly_scaling_factor, half_life, time_unit, intraday_volume_df):
    # stochastic liquidity here presents itself as a modification of the volume (denominator)
    beta = np.log(2) / half_life
    decay_factor = np.exp(-beta * time_unit)
    pre_ewm = traded_volume_df.copy()
    pre_ewm = pre_ewm.divide(np.sqrt(monthly_scaling_factor["volume"]), axis="rows")
    pre_ewm = pre_ewm / np.sqrt(intraday_volume_df.clip(lower=1.))
    pre_ewm = pre_ewm.multiply(monthly_scaling_factor["px_vol"], axis="rows")
    pre_ewm.iloc[:, 1:] /= (1 - decay_factor)
    cumulative_impact = pre_ewm.ewm(alpha=1-decay_factor, adjust=False, axis="columns").mean()
    return cumulative_impact

def impact_state(traded_volume_df, monthly_scaling_factor, half_life, model_type, time_unit, intraday_volume_df):
    if model_type == "reduced_form":
        return reduced_form_impact_state(traded_volume_df, monthly_scaling_factor, half_life, time_unit, intraday_volume_df)

    space_kernels = {
        "linear" : lambda x : x,
        "sqrt" : lambda x : np.sign(x) * np.sqrt(np.abs(x)),
    }

    beta = np.log(2) / half_life
    decay_factor = np.exp(-beta * time_unit)
    pre_ewm = traded_volume_df.copy()
    pre_ewm = pre_ewm.divide(monthly_scaling_factor["volume"], axis="rows")
    pre_ewm = space_kernels[model_type](pre_ewm)
    pre_ewm = pre_ewm.multiply(monthly_scaling_factor["px_vol"], axis="rows")
    pre_ewm.iloc[:, 1:] /= (1 - decay_factor)
    cumulative_impact = pre_ewm.ewm(alpha=1-decay_factor, adjust=False, axis="columns").mean()
    return cumulative_impact

def impact_regression_statistics(cumulative_impact, explanation_horizon_periods, px_df, cutoff_time):
    impact_changes = cumulative_impact.diff(explanation_horizon_periods, axis="columns").T.unstack()
    reg_stat_df = impact_changes.reset_index().rename({"level_2":"time", 0:"x"}, axis="columns")
    returns = px_df.pct_change(explanation_horizon_periods, axis="columns").T.unstack().reset_index().rename({"level_2":"time", 0:"y"}, axis="columns")
    reg_stat_df = reg_stat_df.loc[reg_stat_df["time"] >= cutoff_time].dropna(axis=0).copy()
    reg_stat_df["y"] = returns["y"]
    reg_stat_df["xy"] = reg_stat_df["x"] * reg_stat_df["y"]
    reg_stat_df["xx"] = reg_stat_df["x"] * reg_stat_df["x"]
    reg_stat_df["yy"] = reg_stat_df["y"] * reg_stat_df["y"]
    reg_stat_df["count"] = 1
    return reg_stat_df

def ridge_regression_result(daily_stock_reg_info_df, in_sample_month, lambd=0):
    in_sample_df = daily_stock_reg_info_df.loc[daily_stock_reg_info_df["date"].dt.month == in_sample_month].copy()
    out_sample_df = daily_stock_reg_info_df.loc[daily_stock_reg_info_df["date"].dt.month == in_sample_month + 1].copy()
    in_sample_summary_df = in_sample_df.groupby("stock")[["xy", "xx", "yy", "x", "y", "count"]].sum()
    out_sample_summary_df = out_sample_df.groupby("stock")[["xy", "xx", "yy", "x", "y", "count"]].sum()
    in_sample_summary_df.columns = "is_" + in_sample_summary_df.columns
    out_sample_summary_df.columns = "oos_" + out_sample_summary_df.columns

    summary_df = pd.merge(in_sample_summary_df, out_sample_summary_df, left_index=True, right_index=True, how="inner")
    denominator = (lambd+summary_df["is_xx"])*(lambd+summary_df["is_count"]) - summary_df["is_x"] ** 2
    summary_df["beta_estimate"] = ((lambd+summary_df["is_count"])*summary_df["is_xy"] - summary_df["is_x"]*summary_df["is_y"]) / denominator
    summary_df["alpha_estimate"] = ((lambd+summary_df["is_xx"])*summary_df["is_y"] - summary_df["is_x"]*summary_df["is_xy"]) / denominator
    summary_df["is_sse"] = summary_df["is_yy"] - summary_df["is_y"] ** 2 / summary_df["is_count"]
    summary_df["is_mse"] = summary_df["is_yy"] - 2 * summary_df["beta_estimate"] * summary_df["is_xy"] - 2 * summary_df["alpha_estimate"] * summary_df["is_y"] + 2 * summary_df["alpha_estimate"] * summary_df["beta_estimate"] * summary_df["is_x"] + summary_df["beta_estimate"] ** 2 * summary_df["is_xx"] + summary_df["alpha_estimate"] ** 2 * summary_df["is_count"]
    summary_df["is_rsq"] = 1 - summary_df["is_mse"] / summary_df["is_sse"]
    summary_df["oos_sse"] = summary_df["oos_yy"] - summary_df["oos_y"] ** 2 / summary_df["oos_count"]
    summary_df["oos_mse"] = summary_df["oos_yy"] - 2 * summary_df["beta_estimate"] * summary_df["oos_xy"] - 2 * summary_df["alpha_estimate"] * summary_df["oos_y"] + 2 * summary_df["alpha_estimate"] * summary_df["beta_estimate"] * summary_df["oos_x"] + summary_df["beta_estimate"] ** 2 * summary_df["oos_xx"] + summary_df["alpha_estimate"] ** 2 * summary_df["oos_count"]
    summary_df["oos_rsq"] = 1 - summary_df["oos_mse"] / summary_df["oos_sse"]
    return summary_df
