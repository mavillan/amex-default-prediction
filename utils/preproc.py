from collections import OrderedDict
from decimal import ROUND_HALF_UP, Decimal
from joblib import Parallel, delayed
import numpy as np 
import numba
import pandas as pd
from tqdm import tqdm


def round_values(x):
    return float(Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))  


def compute_mcw_idx(df):
    return (df.close_change*df.MarketCapitalization).sum()/df.MarketCapitalization.sum()


def compute_normalized_index(df, col, init_value=1):
    idx_values = [init_value,]
    for _,row in df.iterrows():
        idx_values.append(row[col] * idx_values[-1] + idx_values[-1])
    return np.array(idx_values[1:])


def compute_indexes_stats(indexes_df, indexes_cols, verbose=True):
    indexes_df = indexes_df.copy()
    for idx_col in indexes_cols:
        if verbose: print(idx_col)
        indexes_df[f"return_{idx_col}_lag1"] = indexes_df[idx_col].pct_change(1)
        indexes_df[f"return_{idx_col}_lag5"] = indexes_df[idx_col].pct_change(5)
        indexes_df[f"return_{idx_col}_lag10"] = indexes_df[idx_col].pct_change(10)
        indexes_df[f"return_{idx_col}_lag20"] = indexes_df[idx_col].pct_change(20)
        indexes_df[f"return_{idx_col}_lag40"] = indexes_df[idx_col].pct_change(40)
        indexes_df[f"return_{idx_col}_lag60"] = indexes_df[idx_col].pct_change(60)
        indexes_df[f"return_{idx_col}_lag100"] = indexes_df[idx_col].pct_change(100)

        indexes_df[f"return_{idx_col}_win5"] = np.log(indexes_df[idx_col] / indexes_df[idx_col].rolling(5).mean())
        indexes_df[f"return_{idx_col}_win10"] = np.log(indexes_df[idx_col] / indexes_df[idx_col].rolling(10).mean())
        indexes_df[f"return_{idx_col}_win20"] = np.log(indexes_df[idx_col] / indexes_df[idx_col].rolling(20).mean())
        indexes_df[f"return_{idx_col}_win40"] = np.log(indexes_df[idx_col] / indexes_df[idx_col].rolling(40).mean())
        indexes_df[f"return_{idx_col}_win60"] = np.log(indexes_df[idx_col] / indexes_df[idx_col].rolling(60).mean())
        indexes_df[f"return_{idx_col}_win100"] = np.log(indexes_df[idx_col] / indexes_df[idx_col].rolling(100).mean())
    return indexes_df


@numba.njit()
def compute_slope(x, y):
    x_mean = np.nanmean(x)
    y_mean = np.nanmean(y)
    return np.nansum((x-x_mean)*(y-y_mean)) / (np.nansum((x-x_mean)**2) + 1e-16)


@numba.njit()
def compute_beta(y, x):
    """
    y: target series
    x: reference series
    """
    if len(x) < 10:
        return np.nan
    else:
        return compute_slope(y, x)

    
def _build_securities_level_feats(df):
    """
    sources for some of the features: 
        - https://github.com/UKI000/JQuants-Forum/blob/main/jquants01_fund_uki_predictor.py
    """
    df = df.copy()
    original_cols = df.columns.tolist()
    df["prevClose"] = df["Close"].shift(1).fillna(df["Open"].values[0])

    ####################################################################################################
    # 1st order features
    ####################################################################################################

    built_feats = OrderedDict()
    
    # return features (over lag)
    built_feats["return_lag1"] = df["Close"].pct_change(1)
    built_feats["return_lag5"] = df["Close"].pct_change(5)
    built_feats["return_lag10"] = df["Close"].pct_change(10)
    built_feats["return_lag20"] = df["Close"].pct_change(20)
    built_feats["return_lag40"] = df["Close"].pct_change(40)
    built_feats["return_lag60"] = df["Close"].pct_change(60)
    built_feats["return_lag100"] = df["Close"].pct_change(100)

    # return features (over rolling window)
    built_feats["return_win5"] = np.log(df.Close / df.Close.rolling(5).mean())
    built_feats["return_win10"] = np.log(df.Close / df.Close.rolling(10).mean())
    built_feats["return_win20"] = np.log(df.Close / df.Close.rolling(20).mean())
    built_feats["return_win40"] = np.log(df.Close / df.Close.rolling(40).mean())
    built_feats["return_win60"] = np.log(df.Close / df.Close.rolling(60).mean())
    built_feats["return_win100"] = np.log(df.Close / df.Close.rolling(100).mean())
    
    # Trading price
    df["volume"] = df["Close"] * df["Volume"]

    built_feats["vol_1"] = df["volume"]
    built_feats["vol_5"] = df["volume"].rolling(5).mean()
    built_feats["vol_10"] = df["volume"].rolling(10).mean()
    built_feats["vol_20"] = df["volume"].rolling(20).mean()
    built_feats["vol_40"] = df["volume"].rolling(40).mean()
    built_feats["vol_60"] = df["volume"].rolling(60).mean()
    built_feats["vol_100"] = df["volume"].rolling(100).mean()
    
    # Range
    df["range"] = (df[["prevClose", "High"]].max(axis=1) - df[["prevClose", "Low"]].min(axis=1)) / df["prevClose"]
    
    built_feats["atr_1"] = df["range"]
    built_feats["atr_5"] = df["range"].rolling(5).mean()
    built_feats["atr_10"] = df["range"].rolling(10).mean()
    built_feats["atr_20"] = df["range"].rolling(20).mean()
    built_feats["atr_40"] = df["range"].rolling(40).mean()
    built_feats["atr_60"] = df["range"].rolling(60).mean()
    built_feats["atr_100"] = df["range"].rolling(100).mean()
    
    # Gap range
    df["gap_range"] = (np.abs(df["Open"] - df["prevClose"])) / df["prevClose"]
    
    built_feats["g_atr_1"] = df["gap_range"]
    built_feats["g_atr_5"] = df["gap_range"].rolling(5).mean()
    built_feats["g_atr_10"] = df["gap_range"].rolling(10).mean()
    built_feats["g_atr_20"] = df["gap_range"].rolling(20).mean()
    built_feats["g_atr_40"] = df["gap_range"].rolling(40).mean()
    built_feats["g_atr_60"] = df["gap_range"].rolling(60).mean()
    built_feats["g_atr_100"] = df["gap_range"].rolling(100).mean()
    
    # Day range
    df["day_range"] = (df["High"] - df["Low"]) / df["prevClose"]
    
    built_feats["d_atr_1"] = df["day_range"]
    built_feats["d_atr_5"] = df["day_range"].rolling(5).mean()
    built_feats["d_atr_10"] = df["day_range"].rolling(10).mean()
    built_feats["d_atr_20"] = df["day_range"].rolling(20).mean()
    built_feats["d_atr_40"] = df["day_range"].rolling(40).mean()
    built_feats["d_atr_60"] = df["day_range"].rolling(60).mean()
    built_feats["d_atr_100"] = df["day_range"].rolling(100).mean()

    # ヒゲレンジ
    df["hig_range"] = ((df["High"] - df["Low"]) - np.abs(df["Open"] - df["Close"])) / df["prevClose"]
    
    built_feats["h_atr_1"] = df["hig_range"]
    built_feats["h_atr_5"] = df["hig_range"].rolling(5).mean()
    built_feats["h_atr_10"] = df["hig_range"].rolling(10).mean()
    built_feats["h_atr_20"] = df["hig_range"].rolling(20).mean()
    built_feats["h_atr_40"] = df["hig_range"].rolling(40).mean()
    built_feats["h_atr_60"] = df["hig_range"].rolling(60).mean()
    built_feats["h_atr_100"] = df["hig_range"].rolling(100).mean()
    
    # Volatility v1
    built_feats["vola1_5"] = ((np.log(df["Close"])).diff()).rolling(5).std()
    built_feats["vola1_10"] = ((np.log(df["Close"])).diff()).rolling(10).std()
    built_feats["vola1_20"] = ((np.log(df["Close"])).diff()).rolling(20).std()
    built_feats["vola1_40"] = ((np.log(df["Close"])).diff()).rolling(40).std()
    built_feats["vola1_60"] = ((np.log(df["Close"])).diff()).rolling(60).std()
    built_feats["vola1_100"] = ((np.log(df["Close"])).diff()).rolling(100).std()
    
    # HL band
    built_feats["hl_5"] = df["High"].rolling(5).max() - df["Low"].rolling(5).min()
    built_feats["hl_10"] = df["High"].rolling(10).max() - df["Low"].rolling(10).min()
    built_feats["hl_20"] = df["High"].rolling(20).max() - df["Low"].rolling(20).min()
    built_feats["hl_40"] = df["High"].rolling(40).max() - df["Low"].rolling(40).min()
    built_feats["hl_60"] = df["High"].rolling(60).max() - df["Low"].rolling(60).min()
    built_feats["hl_100"] = df["High"].rolling(100).max() - df["Low"].rolling(100).min()
    
    # Market impact
    df["mi"] = df["range"] / (df["Volume"] * df["Close"])

    built_feats["mi_5"] = df["mi"].rolling(5).mean()
    built_feats["mi_10"] = df["mi"].rolling(10).mean()
    built_feats["mi_20"] = df["mi"].rolling(20).mean()
    built_feats["mi_40"] = df["mi"].rolling(40).mean()
    built_feats["mi_60"] = df["mi"].rolling(60).mean()
    built_feats["mi_100"] = df["mi"].rolling(100).mean()

    df = pd.concat([df[original_cols], pd.DataFrame(built_feats)], axis=1)

    ####################################################################################################
    # 2nd order features
    ####################################################################################################

    built_feats = OrderedDict()

    # Trading price - relative
    built_feats["d_vol"] = df["vol_1"]/df["vol_20"]

    # Range - relative
    built_feats["d_atr"] = df["atr_1"]/df["atr_20"]

    # Volatility v2
    built_feats["vola2_5"] = df["return_lag1"].rolling(5).std()
    built_feats["vola2_10"] = df["return_lag1"].rolling(10).std()
    built_feats["vola2_20"] = df["return_lag1"].rolling(20).std()
    built_feats["vola2_40"] = df["return_lag1"].rolling(40).std()
    built_feats["vola2_60"] = df["return_lag1"].rolling(60).std()
    built_feats["vola2_100"] = df["return_lag1"].rolling(100).std()
        
    # beta features on EW index
    for window in [10,20,40,60,100]:
        roll1 = df["return_lag1"].rolling(window)
        roll2 = df["return_idx_MKT_ew_lag1"].rolling(window)
        results = [compute_beta(y.values, x.values) for y,x in zip(roll1,roll2)]
        built_feats[f"beta_ewi_w{window}"] = np.array(results)
    
    # beta features on MC index
    for window in [10,20,40,60,100]:
        roll1 = df["return_lag1"].rolling(window)
        roll2 = df["return_idx_MKT_mcw_lag1"].rolling(window)
        results = [compute_beta(y.values, x.values) for y,x in zip(roll1,roll2)]
        built_feats[f"beta_mci_w{window}"] = np.array(results)  
    
    # difference of return
    for lag in [1,5,10,20,40,60,100]:
        built_feats[f"return_diff_idx_MKT_ew_lag{lag}"] = df.eval(f"return_lag{lag} - return_idx_MKT_ew_lag{lag}")
        built_feats[f"return_diff_idx_MKT_mcw_lag{lag}"] = df.eval(f"return_lag{lag} - return_idx_MKT_mcw_lag{lag}")

    for window in [5,10,20,40,60,100]:
        built_feats[f"return_diff_idx_MKT_ew_win{window}"] = df.eval(f"return_win{window} - return_idx_MKT_ew_win{window}")
        built_feats[f"return_diff_idx_MKT_mcw_win{window}"] = df.eval(f"return_win{window} - return_idx_MKT_mcw_win{window}")
    
    return pd.concat([df, pd.DataFrame(built_feats)], axis=1)


def build_securities_level_feats(stock_prices:pd.DataFrame, n_jobs:int = 8, verbose:bool = False):
    with Parallel(n_jobs=n_jobs) as parallel:
            delayed_func = delayed(_build_securities_level_feats)
            all_dfs = parallel(
                delayed_func(df) 
                for _,df in tqdm(stock_prices.groupby("SecuritiesCode"), disable=not verbose)
            )
    return pd.concat(all_dfs, ignore_index=True)


def _build_securities_level_feats_inference(df):
    """
    sources for some of the features: 
        - https://github.com/UKI000/JQuants-Forum/blob/main/jquants01_fund_uki_predictor.py
    """
    df = df.copy()
    original_cols = df.columns.tolist()
    df["prevClose"] = df["Close"].shift(1).fillna(df["Open"].values[0])
    df["return_lag1"] = df["Close"].pct_change(1)

    ####################################################################################################
    # 1st order features
    ####################################################################################################

    # return features (over lag)
    built_feats = OrderedDict()
    
    # return features (over lag)
    built_feats["return_lag1"] = (df["Close"].values[-1] - df["Close"].values[-2]) / df["Close"].values[-2]
    built_feats["return_lag5"] = (df["Close"].values[-1] - df["Close"].values[-6]) / df["Close"].values[-6]
    built_feats["return_lag10"] = (df["Close"].values[-1] - df["Close"].values[-11]) / df["Close"].values[-11]
    built_feats["return_lag20"] = (df["Close"].values[-1] - df["Close"].values[-21]) / df["Close"].values[-21]
    built_feats["return_lag40"] = (df["Close"].values[-1] - df["Close"].values[-41]) / df["Close"].values[-41]
    built_feats["return_lag60"] = (df["Close"].values[-1] - df["Close"].values[-61]) / df["Close"].values[-61]
    built_feats["return_lag100"] = (df["Close"].values[-1] - df["Close"].values[-101]) / df["Close"].values[-101]

    # return features (over rolling window)
    built_feats["return_win5"] = np.log(df["Close"].values[-1] / df["Close"].values[-5:].mean())
    built_feats["return_win10"] = np.log(df["Close"].values[-1] / df["Close"].values[-10:].mean())
    built_feats["return_win20"] = np.log(df["Close"].values[-1] / df["Close"].values[-20:].mean())
    built_feats["return_win40"] = np.log(df["Close"].values[-1] / df["Close"].values[-40:].mean())
    built_feats["return_win60"] = np.log(df["Close"].values[-1] / df["Close"].values[-60:].mean())
    built_feats["return_win100"] = np.log(df["Close"].values[-1] / df["Close"].values[-100:].mean())
    
    # Trading price
    df["volume"] = df["Close"] * df["Volume"]

    built_feats["vol_1"] = df["volume"].values[-1]
    built_feats["vol_5"] = df["volume"].values[-5:].mean()
    built_feats["vol_10"] = df["volume"].values[-10:].mean()
    built_feats["vol_20"] = df["volume"].values[-20:].mean()
    built_feats["vol_40"] = df["volume"].values[-40:].mean()
    built_feats["vol_60"] = df["volume"].values[-60:].mean()
    built_feats["vol_100"] = df["volume"].values[-100:].mean()
    
    # Range
    df["range"] = (df[["prevClose", "High"]].max(axis=1) - df[["prevClose", "Low"]].min(axis=1)) / df["prevClose"]
    
    built_feats["atr_1"] = df["range"].values[-1]
    built_feats["atr_5"] = df["range"].values[-5:].mean()
    built_feats["atr_10"] = df["range"].values[-10:].mean()
    built_feats["atr_20"] = df["range"].values[-20:].mean()
    built_feats["atr_40"] = df["range"].values[-40:].mean()
    built_feats["atr_60"] = df["range"].values[-60:].mean()
    built_feats["atr_100"] = df["range"].values[-100:].mean()
    
    # Gap range
    df["gap_range"] = (np.abs(df["Open"] - df["prevClose"])) / df["prevClose"]
    
    built_feats["g_atr_1"] = df["gap_range"].values[-1]
    built_feats["g_atr_5"] = df["gap_range"].values[-5:].mean()
    built_feats["g_atr_10"] = df["gap_range"].values[-10:].mean()
    built_feats["g_atr_20"] = df["gap_range"].values[-20:].mean()
    built_feats["g_atr_40"] = df["gap_range"].values[-40:].mean()
    built_feats["g_atr_60"] = df["gap_range"].values[-60:].mean()
    built_feats["g_atr_100"] = df["gap_range"].values[-100:].mean()
    
    # Day range
    df["day_range"] = (df["High"] - df["Low"]) / df["prevClose"]
    
    built_feats["d_atr_1"] = df["day_range"].values[-1]
    built_feats["d_atr_5"] = df["day_range"].values[-5:].mean()
    built_feats["d_atr_10"] = df["day_range"].values[-10:].mean()
    built_feats["d_atr_20"] = df["day_range"].values[-20:].mean()
    built_feats["d_atr_40"] = df["day_range"].values[-40:].mean()
    built_feats["d_atr_60"] = df["day_range"].values[-60:].mean()
    built_feats["d_atr_100"] = df["day_range"].values[-100:].mean()

    # ヒゲレンジ
    df["hig_range"] = ((df["High"] - df["Low"]) - np.abs(df["Open"] - df["Close"])) / df["prevClose"]
    
    built_feats["h_atr_1"] = df["hig_range"].values[-1]
    built_feats["h_atr_5"] = df["hig_range"].values[-5:].mean()
    built_feats["h_atr_10"] = df["hig_range"].values[-10:].mean()
    built_feats["h_atr_20"] = df["hig_range"].values[-20:].mean()
    built_feats["h_atr_40"] = df["hig_range"].values[-40:].mean()
    built_feats["h_atr_60"] = df["hig_range"].values[-60:].mean()
    built_feats["h_atr_100"] = df["hig_range"].values[-100:].mean()
    
    # Volatility v1
    built_feats["vola1_5"] = np.diff(np.log(df["Close"].values))[-5:].std(ddof=1)
    built_feats["vola1_10"] = np.diff(np.log(df["Close"].values))[-10:].std(ddof=1)
    built_feats["vola1_20"] = np.diff(np.log(df["Close"].values))[-20:].std(ddof=1)
    built_feats["vola1_40"] = np.diff(np.log(df["Close"].values))[-40:].std(ddof=1)
    built_feats["vola1_60"] = np.diff(np.log(df["Close"].values))[-60:].std(ddof=1)
    built_feats["vola1_100"] = np.diff(np.log(df["Close"].values))[-100:].std(ddof=1)
    
    # HL band
    built_feats["hl_5"] = df["High"].values[-5:].max() - df["Low"].values[-5:].min()
    built_feats["hl_10"] = df["High"].values[-10:].max() - df["Low"].values[-10:].min()
    built_feats["hl_20"] = df["High"].values[-20:].max() - df["Low"].values[-20:].min()
    built_feats["hl_40"] = df["High"].values[-40:].max() - df["Low"].values[-40:].min()
    built_feats["hl_60"] = df["High"].values[-60:].max() - df["Low"].values[-60:].min()
    built_feats["hl_100"] = df["High"].values[-100:].max() - df["Low"].values[-100:].min()
    
    # Market impact
    df["mi"] = df["range"] / (df["Volume"] * df["Close"])

    built_feats["mi_5"] = df["mi"].values[-5:].mean()
    built_feats["mi_10"] = df["mi"].values[-10:].mean()
    built_feats["mi_20"] = df["mi"].values[-20:].mean()
    built_feats["mi_40"] = df["mi"].values[-40:].mean()
    built_feats["mi_60"] = df["mi"].values[-60:].mean()
    built_feats["mi_100"] = df["mi"].values[-100:].mean()
        
    ####################################################################################################
    # 2nd order features
    ####################################################################################################

    # Trading price - relative
    built_feats["d_vol"] = built_feats["vol_1"]/built_feats["vol_20"]

    # Range - relative
    built_feats["d_atr"] = built_feats["atr_1"]/built_feats["atr_20"]

    # Volatility v2
    built_feats["vola2_5"] = df["return_lag1"].values[-5:].std(ddof=1)
    built_feats["vola2_10"] = df["return_lag1"].values[-10:].std(ddof=1)
    built_feats["vola2_20"] = df["return_lag1"].values[-20:].std(ddof=1)
    built_feats["vola2_40"] = df["return_lag1"].values[-40:].std(ddof=1)
    built_feats["vola2_60"] = df["return_lag1"].values[-60:].std(ddof=1)
    built_feats["vola2_100"] = df["return_lag1"].values[-100:].std(ddof=1)
        
    #beta features on EW index
    for window in [10,20,40,60,100]:
        returns_y = df["return_lag1"].values[-window:]
        returns_x = df["return_idx_MKT_ew_lag1"].values[-window:]
        built_feats[f"beta_ewi_w{window}"] = compute_beta(returns_y, returns_x)
    
    #beta features on MC index
    for window in [10,20,40,60,100]:
        returns_y = df["return_lag1"].values[-window:]
        returns_x = df["return_idx_MKT_mcw_lag1"].values[-window:]
        built_feats[f"beta_mci_w{window}"] = compute_beta(returns_y, returns_x)
    
    # difference of return
    for lag in [1,5,10,20,40,60,100]:
        built_feats[f"return_diff_idx_MKT_ew_lag{lag}"] = built_feats[f"return_lag{lag}"] - df[f"return_idx_MKT_ew_lag{lag}"].values[-1]
        built_feats[f"return_diff_idx_MKT_mcw_lag{lag}"] = built_feats[f"return_lag{lag}"] - df[f"return_idx_MKT_mcw_lag{lag}"].values[-1]

    for window in [5,10,20,40,60,100]:
        built_feats[f"return_diff_idx_MKT_ew_win{window}"] = built_feats[f"return_win{window}"] - df[f"return_idx_MKT_ew_win{window}"].values[-1]
        built_feats[f"return_diff_idx_MKT_mcw_win{window}"] = built_feats[f"return_win{window}"] - df[f"return_idx_MKT_mcw_win{window}"].values[-1]
        
    out = df.tail(1)[original_cols].to_dict(orient="list", into=OrderedDict)
    out.update(built_feats)
    
    return pd.DataFrame(out)


def build_securities_level_feats_inference(stock_prices:pd.DataFrame, n_jobs:int = 8, verbose:bool = False,):
    with Parallel(n_jobs=n_jobs) as parallel:
            delayed_func = delayed(_build_securities_level_feats_inference)
            all_dfs = parallel(
                delayed_func(df) 
                for _,df in tqdm(stock_prices.groupby("SecuritiesCode"), disable=not verbose)
            )
    return pd.concat(all_dfs, ignore_index=True)


def build_day_level_feats(stock_prices:pd.DataFrame) -> pd.DataFrame:
    
    # rank by MarketCapitalization
    stock_prices["MC_rank"] = (
        stock_prices
        .groupby("Date")
        ["MarketCapitalization"]
        .rank(ascending=True, method="first", pct=True)
    )

    stock_prices["MC_rank_by_33Sector"] = (
        stock_prices
        .groupby(["Date","33SectorName"])
        ["MarketCapitalization"]
        .rank(ascending=True, method="first", pct=True)
    )

    stock_prices["MC_rank_by_17Sector"] = (
        stock_prices
        .groupby(["Date","17SectorName"])
        ["MarketCapitalization"]
        .rank(ascending=True, method="first", pct=True)
    )
    
    # rank by relative performance (relative to markets)
    for lag in [1,5,10,20,40,60,100]:
        stock_prices[f"return_diff_rnk_idx_MKT_ew_lag{lag}"] = (
            stock_prices
            .groupby("Date")
            [f"return_diff_idx_MKT_ew_lag{lag}"]
            .rank(ascending=True, method="first", pct=True)
        )
        stock_prices[f"return_diff_rnk_idx_MKT_mcw_lag{lag}"] = (
            stock_prices
            .groupby("Date")
            [f"return_diff_idx_MKT_mcw_lag{lag}"]
            .rank(ascending=True, method="first", pct=True)
        )

    for window in [5,10,20,40,60,100]:
        stock_prices[f"return_diff_rnk_idx_MKT_ew_win{window}"] = (
            stock_prices
            .groupby("Date")
            [f"return_diff_idx_MKT_ew_win{window}"]
            .rank(ascending=True, method="first", pct=True)
        )
        stock_prices[f"return_diff_rnk_idx_MKT_mcw_win{window}"] = (
            stock_prices
            .groupby("Date")
            [f"return_diff_idx_MKT_mcw_win{window}"]
            .rank(ascending=True, method="first", pct=True)
        )
        
    return stock_prices


def compute_time_features(dataframe:pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe["day_of_week"] = dataframe.Date.dt.day_of_week.astype(int)
    dataframe["day_of_month"] = dataframe.Date.dt.day / dataframe.Date.dt.days_in_month
    dataframe["week_of_year"] = dataframe.Date.dt.isocalendar().week.astype(int)
    dataframe["month"] = dataframe.Date.dt.month.astype(int)
    return dataframe