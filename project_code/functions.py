# AUTO-GENERATED FROM notebooks/functions.ipynb
# DO NOT EDIT DIRECTLY

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re
import optuna

# modeling
from xgboost import XGBRegressor
from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from scipy.special import logit, expit
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline

# PyTorch for conversion model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# nfl pbp loader
import nfl_data_py as nfl

from datetime import datetime
import joblib

def parse_weather(weather_str):
    """
    Parses a weather string into structured features:
        - temp_F: float
        - humidity: float (percentage)
        - wind_mph: float
        - wind_dir: str
        - conditions: str (general description, e.g., 'sunny', 'cloudy', etc.)
    """
    result = {
        "temp_F": None,
        "humidity": None,
        "wind_mph": None,
        "wind_dir": None,
        "conditions": None
    }
    
    if not isinstance(weather_str, str):
        return result
    
    lower_str = weather_str.lower()
    
    # Extract temperature
    temp_match = re.search(r'(\d+)\s*°?\s*f', lower_str)
    if temp_match:
        result['temp_F'] = float(temp_match.group(1))
    
    # Extract humidity
    hum_match = re.search(r'humidity[:\s]*(\d+)%', lower_str)
    if hum_match:
        result['humidity'] = float(hum_match.group(1))
    
    # Extract wind speed and direction
    wind_match = re.search(r'wind[:\s]*([nesw]+)\s*(\d+)\s*mph', lower_str)
    if wind_match:
        result['wind_dir'] = wind_match.group(1).upper()
        result['wind_mph'] = float(wind_match.group(2))
    
    # Extract general conditions
    conditions = []
    for cond in ['sunny', 'cloudy', 'clear', 'rain', 'snow', 'fog', 'drizzle', 'storm', 'windy']:
        if cond in lower_str:
            conditions.append(cond)
    if conditions:
        result['conditions'] = ','.join(conditions)
    
    return result

def deconstruct_weather(df, weather_col='weather'):
    """
    Adds structured weather columns to a DataFrame based on a weather string column.
    
    New columns added:
      - temp_F
      - humidity
      - wind_mph
      - wind_dir
      - conditions
    """
    weather_data = df[weather_col].apply(parse_weather)
    weather_df = pd.DataFrame(weather_data.tolist())
    df = pd.concat([df.reset_index(drop=True), weather_df], axis=1)
    
    # Fill missing wind speeds with 0
    df['wind_mph'] = df['wind_mph'].fillna(0)

    # Fill missing temperatures with 60°F
    df['temp_F'] = df['temp_F'].fillna(60)

    return df

def create_features(df):
    """
    Safely adds derived football features.
    Only creates features when required base columns exist.
    Missing dependencies -> feature is created as NaN or 0.
    """

    df = df.copy()

    if "yardline_100" in df.columns:
        df["is_redzone"] = (df["yardline_100"] <= 20).astype(int)
    else:
        df["is_redzone"] = 0

    if {"ydstogo", "yardline_100"}.issubset(df.columns):
        df["is_goal_to_go"] = (df["ydstogo"] >= df["yardline_100"]).astype(int)
    else:
        df["is_goal_to_go"] = 0

    if "ydstogo" in df.columns:
        df["log_ydstogo"] = np.log1p(df["ydstogo"].clip(lower=0))
    else:
        df["log_ydstogo"] = np.nan

    if "game_seconds_remaining" in df.columns:
        df["log_game_seconds_remaining"] = np.log1p(df["game_seconds_remaining"].clip(lower=0))
    else:
        df["log_game_seconds_remaining"] = np.nan

    if "score_differential" in df.columns:
        df["abs_score_differential"] = df["score_differential"].abs()
    else:
        df["abs_score_differential"] = np.nan

    if {"score_differential", "game_seconds_remaining"}.issubset(df.columns):
        df["score_time_ratio"] = (df["score_differential"].abs() / (df["game_seconds_remaining"] + 1))
    else:
        df["score_time_ratio"] = np.nan

    return df

def make_temporal_folds(df, season_col="season", min_train_seasons=3):
    """
    Expanding-window CV folds by season.
    Returns list of (train_idx, val_idx).
    """
    seasons = np.sort(df[season_col].unique())
    folds = []

    for i in range(min_train_seasons, len(seasons)):
        train_seasons = seasons[:i]
        val_season = seasons[i]

        train_idx = df[df[season_col].isin(train_seasons)].index
        val_idx = df[df[season_col] == val_season].index

        folds.append((train_idx, val_idx))

    return folds

def wp_objective(trial, wp_fixed_params, X_wp, y_wp_clipped, wp_folds, mono_tuple):

    tuned_params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
        "max_depth": trial.suggest_int("max_depth", 5, 6),
        "subsample": trial.suggest_float("subsample", 0.7, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.9),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 100, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 50.0, log=True),
    }

    params = {
        **wp_fixed_params,
        **tuned_params,
        "monotone_constraints": mono_tuple,
    }   
    
    rmses = []
    for train_idx, val_idx in wp_folds:
        X_train = X_wp.iloc[train_idx].to_numpy(dtype=np.float32, copy=False)
        X_val   = X_wp.iloc[val_idx].to_numpy(dtype=np.float32, copy=False)
        y_train = y_wp_clipped.iloc[train_idx].to_numpy(dtype=np.float32, copy=False)
        y_val   = y_wp_clipped.iloc[val_idx].to_numpy(dtype=np.float32, copy=False)

        model = XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        preds = model.predict(X_val)
        rmses.append(mean_squared_error(y_val, preds, squared=False))

    return float(np.mean(rmses))

def predict_wp(state_df, wp_model, wp_features):
    """
    Returns win probability for the team with possession in state_df.
    """
    
    preds = wp_model.predict(state_df[wp_features])
    
    return np.clip(preds, 0.0, 1.0)

def wp_symmetric_adjust(state_df, predict_wp, wp_model, wp_features):

    # Ensure engineered features exist for the "original" prediction too
    state_df_feat = create_features(state_df.copy())
    wp = predict_wp(state_df_feat, wp_model, wp_features)

    state_flipped = state_df.copy()

    if "score_differential" in state_flipped.columns:
        state_flipped["score_differential"] *= -1
    if "possession_spread_line" in state_flipped.columns:
        state_flipped["possession_spread_line"] *= -1

    if {"posteam_timeouts_remaining", "defteam_timeouts_remaining"}.issubset(state_flipped.columns):
        state_flipped[["posteam_timeouts_remaining", "defteam_timeouts_remaining"]] = (
            state_flipped[["defteam_timeouts_remaining", "posteam_timeouts_remaining"]].values
        )

    if "yardline_100" in state_flipped.columns:
        state_flipped["yardline_100"] = 100 - state_flipped["yardline_100"]

    state_flipped_feat = create_features(state_flipped)
    wp_flipped = predict_wp(state_flipped_feat, wp_model, wp_features)

    wp_sym = 0.5 * (wp + (1 - wp_flipped))
    sym_weighting = 0.2
    return (1 - sym_weighting) * wp + sym_weighting * wp_sym

def punt_objective(trial, X_punt, y_punt, punt_folds):

    tuned_params = {
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 5.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }

    fixed_params = {
        "n_estimators": 5000,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "max_bin": 256,
        "early_stopping_rounds": 100,
        "verbosity": 0,
        "n_jobs": 14,
    }

    params = {**fixed_params, **tuned_params}

    rmses = []
    for train_idx, val_idx in punt_folds:
        X_train = X_punt[train_idx].astype(np.float32, copy=False)
        X_val   = X_punt[val_idx].astype(np.float32, copy=False)
        y_train = y_punt[train_idx].astype(np.float32, copy=False)
        y_val   = y_punt[val_idx].astype(np.float32, copy=False)

        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        best_it = getattr(model, "best_iteration", None)
        if best_it is None:
            preds = model.predict(X_val)
        else:
            preds = model.predict(X_val, iteration_range=(0, best_it + 1))

        rmses.append(mean_squared_error(y_val, preds, squared=False))

    return float(np.mean(rmses))

def go_objective(trial, go_fixed_params, X_go, y_go, go_folds, mono_tuple_go):
    
    # Suggest hyperparameters
    tuned_params = {
        "max_depth": trial.suggest_int("max_depth", 2, 4),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.9),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 50.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
    }
    
    params = {
        **go_fixed_params,
        **tuned_params,
        "monotone_constraints": mono_tuple_go,
    }   

    log_losses = []

    for train_idx, val_idx in go_folds:
        X_train, X_val = X_go[train_idx], X_go[val_idx]
        y_train, y_val = y_go[train_idx], y_go[val_idx]

        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        best_it = getattr(model, "best_iteration", None)
        if best_it is None:
            preds = model.predict_proba(X_val)[:, 1]
        else:
            preds = model.predict_proba(X_val, iteration_range=(0, best_it + 1))[:, 1]

        preds = np.clip(preds, 1e-15, 1 - 1e-15)
        log_losses.append(log_loss(y_val, preds))

    return float(np.mean(log_losses))

def create_plays_df(df):
    
    # Compute final scores and win from offensive team perspective
    final_scores = (
        df.groupby("game_id")
           .tail(1)[["game_id","home_team","away_team","home_score","away_score"]]
           .copy()
    )
    final_scores["home_win"] = (final_scores["home_score"] > final_scores["away_score"]).astype(int)

    df = df.merge(final_scores[["game_id","home_win"]], on="game_id", how="left")
    df["win_actual"] = np.where(
        df["posteam"] == df["home_team"],
        df["home_win"],
        1 - df["home_win"]
    )
    
    return df

def create_next_fg_conv_states(df):
    
    # Next state if successful field goal attempt
    df['fg_success_yardline_100'] = 75
    df['fg_success_down'] = 1
    df['fg_success_ydstogo'] = 10
    df['fg_success_game_seconds_remaining'] = np.maximum(0, df['game_seconds_remaining'] - 5)
    df['fg_success_half_seconds_remaining'] = np.maximum(0, df['half_seconds_remaining'] - 5)
    df['fg_success_score_differential'] = -(df['score_differential'] + 3)
    df['fg_success_posteam_timeouts_remaining'] = df['defteam_timeouts_remaining']
    df['fg_success_defteam_timeouts_remaining'] = df['posteam_timeouts_remaining']
    df['fg_success_temp_F'] = df['temp_F']
    df['fg_success_wind_mph'] = df['wind_mph']
    df['fg_success_possession_spread_line'] = -df['possession_spread_line']
    df['fg_success_total_line'] = df['total_line']
    
    # Next state if failed field goal attempt
    df['fg_fail_yardline_100'] = np.minimum(80, 100 - (df['yardline_100'] + 7)) # Account for inside 20-yardline edge case
    df['fg_fail_down'] = 1
    df['fg_fail_ydstogo'] = 10
    df['fg_fail_game_seconds_remaining'] = np.maximum(0, df['game_seconds_remaining'] - 5)
    df['fg_fail_half_seconds_remaining'] = np.maximum(0, df['half_seconds_remaining'] - 5)
    df['fg_fail_score_differential'] = -df['score_differential']
    df['fg_fail_posteam_timeouts_remaining'] = df['defteam_timeouts_remaining']
    df['fg_fail_defteam_timeouts_remaining'] = df['posteam_timeouts_remaining']
    df['fg_fail_temp_F'] = df['temp_F']
    df['fg_fail_wind_mph'] = df['wind_mph']
    df['fg_fail_possession_spread_line'] = -df['possession_spread_line']
    df['fg_fail_total_line'] = df['total_line']
    
    # Next state if successful conversion attempt
    # Note that possession flips only if successful conversio attempt results in a touchdown
    # If successful, assume advancement to 1 yd beyond line to gain
    go_new_yline = df["yardline_100"] - df["ydstogo"] - 1
    go_td = go_new_yline <= 0
    df["go_success_flip"] = go_td.astype(int)  # 1 if TD (possession flips), else 0
    df["go_success_yardline_100"] = np.where(go_td, 75, np.maximum(1, go_new_yline))
    df["go_success_down"] = 1
    df["go_success_ydstogo"] = np.where(go_td, 10, np.minimum(10, df["go_success_yardline_100"]))
    df["go_success_game_seconds_remaining"] = np.maximum(0, df["game_seconds_remaining"] - 5)
    df["go_success_half_seconds_remaining"] = np.maximum(0, df["half_seconds_remaining"] - 5)
    df["go_success_score_differential"] = np.where(go_td, -(df["score_differential"] + 7), df["score_differential"])
    df["go_success_posteam_timeouts_remaining"] = np.where(go_td, df["defteam_timeouts_remaining"], df["posteam_timeouts_remaining"])
    df["go_success_defteam_timeouts_remaining"] = np.where(go_td, df["posteam_timeouts_remaining"], df["defteam_timeouts_remaining"])
    df["go_success_temp_F"] = df["temp_F"]
    df["go_success_wind_mph"] = df["wind_mph"]
    df["go_success_possession_spread_line"] = np.where(go_td, -df["possession_spread_line"], df["possession_spread_line"])
    df["go_success_total_line"] = df["total_line"]
    
    # Next state if failed conversion attempt
    df['go_fail_yardline_100'] = 100 - df['yardline_100']
    df['go_fail_down'] = 1
    df['go_fail_ydstogo'] = 10
    df['go_fail_game_seconds_remaining'] = np.maximum(0, df['game_seconds_remaining'] - 5)
    df['go_fail_half_seconds_remaining'] = np.maximum(0, df['half_seconds_remaining'] - 5)
    df['go_fail_score_differential'] = -df['score_differential']
    df['go_fail_posteam_timeouts_remaining'] = df['defteam_timeouts_remaining']
    df['go_fail_defteam_timeouts_remaining'] = df['posteam_timeouts_remaining']
    df['go_fail_temp_F'] = df['temp_F']
    df['go_fail_wind_mph'] = df['wind_mph']
    df['go_fail_possession_spread_line'] = -df['possession_spread_line']
    df['go_fail_total_line'] = df['total_line']
    
    return df

def calculate_ewp_fg(df, wp_model, fg_model, wp_features, wp_base_features, fg_features):
    max_fg = 65
    fg_decay_threshold = 60

    if "down" in df.columns:
        fourth_down_mask = df["down"] == 4
    else:
        fourth_down_mask = pd.Series(True, index=df.index)

    # --- build success/fail WP frames from prefixed base columns
    success_base_cols = [f"fg_success_{f}" for f in wp_base_features]
    fail_base_cols = [f"fg_fail_{f}"    for f in wp_base_features]

    X_fg_success = df.loc[fourth_down_mask, success_base_cols].copy()
    X_fg_success.columns = wp_base_features
    X_fg_success = create_features(X_fg_success)

    X_fg_fail = df.loc[fourth_down_mask, fail_base_cols].copy()
    X_fg_fail.columns = wp_base_features
    X_fg_fail = create_features(X_fg_fail)

    wp_fg_success = 1 - wp_symmetric_adjust(X_fg_success, predict_wp, wp_model, wp_features)
    wp_fg_fail    = 1 - wp_symmetric_adjust(X_fg_fail,    predict_wp, wp_model, wp_features)

    # --- FG make probability uses current state
    X_fg_current = df.loc[fourth_down_mask, fg_features].copy()
    p_make = fg_model.predict_proba(X_fg_current)[:, 1]
    yardlines = X_fg_current["yardline_100"].to_numpy()

    p_make_decayed = np.where(
        yardlines >= (fg_decay_threshold - 17),
        p_make * np.maximum(0.0, (max_fg - 17 - yardlines) / (max_fg - fg_decay_threshold)),
        p_make,
    )

    # --- EWP (only defined on 4th downs)
    ewp_fg_4th = np.clip(
        p_make_decayed * wp_fg_success + (1.0 - p_make_decayed) * wp_fg_fail,
        0.0, 1.0
    )

    # --- write back safely (no broadcasting issues)
    df["ewp_fg"] = np.nan
    df["wp_fg_success"] = np.nan
    df["wp_fg_fail"] = np.nan
    df["p_make_fg"] = np.nan

    df.loc[fourth_down_mask, "ewp_fg"] = ewp_fg_4th
    df.loc[fourth_down_mask, "wp_fg_success"] = wp_fg_success
    df.loc[fourth_down_mask, "wp_fg_fail"] = wp_fg_fail
    df.loc[fourth_down_mask, "p_make_fg"] = p_make_decayed

    return df

def calculate_ewp_go(df, wp_model, go_model, wp_features, wp_base_features, go_features):
    
    if "down" in df.columns:
        fourth_down_mask = df["down"] == 4
    else:
        fourth_down_mask = pd.Series(True, index=df.index)

    # Build success/fail WP feature frames
    success_cols = [f"go_success_{f}" for f in wp_base_features]
    fail_cols = [f"go_fail_{f}"    for f in wp_base_features]

    X_go_success = df.loc[fourth_down_mask, success_cols].copy()
    X_go_success.columns = wp_base_features
    X_go_success = create_features(X_go_success)

    X_go_fail = df.loc[fourth_down_mask, fail_cols].copy()
    X_go_fail.columns = wp_base_features
    X_go_fail = create_features(X_go_fail)    
    
    # If TD happened on success, that next-state row is from new offense perspective; invert it
    wp_go_success_raw = wp_symmetric_adjust(X_go_success, predict_wp, wp_model, wp_features)
    flip_success = df.loc[fourth_down_mask, "go_success_flip"].to_numpy(dtype=int)
    wp_go_success = np.where(flip_success == 1, 1.0 - wp_go_success_raw, wp_go_success_raw)
    wp_go_fail = 1 - wp_symmetric_adjust(X_go_fail, predict_wp, wp_model, wp_features)

    # Conversion probabilities
    X_go_current = df.loc[fourth_down_mask, go_features].copy()
    p_convert = go_model.predict_proba(X_go_current)[:, 1]

    # Raw EWP
    ewp_go = np.clip(p_convert * wp_go_success + (1.0 - p_convert) * wp_go_fail, 0.0, 1.0)

    # Write back
    df.loc[fourth_down_mask, "p_convert"] = p_convert
    df.loc[fourth_down_mask, "ewp_go"] = ewp_go
    df.loc[fourth_down_mask, "wp_go_success"] = wp_go_success
    df.loc[fourth_down_mask, "wp_go_fail"] = wp_go_fail

    return df

def create_punt_next_state(df, punt_model, punt_features):
    
    if "down" in df.columns:
        fourth_down_mask = df["down"] == 4
    else:
        fourth_down_mask = pd.Series(True, index=df.index)

    # Initialize outputs as NaN so non-4th rows don't get junk
    df["punt_pred_yards"] = np.nan
    df["post_punt_yardline_100"] = np.nan
    df["post_punt_down"] = np.nan
    df["post_punt_ydstogo"] = np.nan
    df["post_punt_game_seconds_remaining"] = np.nan
    df["post_punt_half_seconds_remaining"] = np.nan
    df["post_punt_score_differential"] = np.nan
    df["post_punt_posteam_timeouts_remaining"] = np.nan
    df["post_punt_defteam_timeouts_remaining"] = np.nan
    df["post_punt_temp_F"] = np.nan
    df["post_punt_wind_mph"] = np.nan
    df["post_punt_possession_spread_line"] = np.nan
    df["post_punt_total_line"] = np.nan

    # --- Predict punt yards (4th-down rows only)
    X_punt_current = df.loc[fourth_down_mask, punt_features].to_numpy(dtype=np.float32, copy=False)
    punt_pred_yards = punt_model.predict(X_punt_current)
    df.loc[fourth_down_mask, "punt_pred_yards"] = punt_pred_yards

    # --- Next state (base features only)
    yardline = df.loc[fourth_down_mask, "yardline_100"].to_numpy(dtype=np.float32, copy=False)
    landing_kicking = yardline - punt_pred_yards
    landing_kicking = np.where(landing_kicking <= 0, 20, landing_kicking)  # touchback if beyond endzone

    df.loc[fourth_down_mask, "post_punt_yardline_100"] = 100 - landing_kicking  # flip field
    df.loc[fourth_down_mask, "post_punt_down"] = 1
    df.loc[fourth_down_mask, "post_punt_ydstogo"] = 10
    df.loc[fourth_down_mask, "post_punt_game_seconds_remaining"] = np.maximum(
        0, df.loc[fourth_down_mask, "game_seconds_remaining"] - 8
    )
    df.loc[fourth_down_mask, "post_punt_half_seconds_remaining"] = np.maximum(
        0, df.loc[fourth_down_mask, "half_seconds_remaining"] - 8
    )
    df.loc[fourth_down_mask, "post_punt_score_differential"] = -df.loc[fourth_down_mask, "score_differential"]
    df.loc[fourth_down_mask, "post_punt_posteam_timeouts_remaining"] = df.loc[fourth_down_mask, "defteam_timeouts_remaining"]
    df.loc[fourth_down_mask, "post_punt_defteam_timeouts_remaining"] = df.loc[fourth_down_mask, "posteam_timeouts_remaining"]
    df.loc[fourth_down_mask, "post_punt_temp_F"] = df.loc[fourth_down_mask, "temp_F"]
    df.loc[fourth_down_mask, "post_punt_wind_mph"] = df.loc[fourth_down_mask, "wind_mph"]
    df.loc[fourth_down_mask, "post_punt_possession_spread_line"] = -df.loc[fourth_down_mask, "possession_spread_line"]
    df.loc[fourth_down_mask, "post_punt_total_line"] = df.loc[fourth_down_mask, "total_line"]

    return df

def calculate_ewp_punt(df, wp_model, wp_features, wp_base_features):
    
    if "down" in df.columns:
        fourth_down_mask = df["down"] == 4
    else:
        fourth_down_mask = pd.Series(True, index=df.index)

    # Build from base post-punt columns, then derive engineered features
    post_base_cols = [f"post_punt_{f}" for f in wp_base_features]

    X_post_punt = df.loc[fourth_down_mask, post_base_cols].copy()
    X_post_punt.columns = wp_base_features
    X_post_punt = create_features(X_post_punt)

    wp_post_punt = 1 - wp_symmetric_adjust(X_post_punt, predict_wp, wp_model, wp_features)

    df["ewp_punt"] = np.nan
    df.loc[fourth_down_mask, "ewp_punt"] = wp_post_punt

    return df

def make_recommendations(df, test=False):
    
    ewp_cols = ["ewp_punt", "ewp_fg", "ewp_go"]
    
    if not test:

        # Compute actual EWP for each row (using actual_ewp_col)
        bad = set(df["actual_ewp_col"].dropna().unique()) - set(df.columns)
        if bad:
            raise KeyError(f"actual_ewp_col points to missing columns: {bad}")

        col_idx = df[["actual_ewp_col"]].apply(
            lambda x: df.columns.get_loc(x[0]),
            axis=1
        ).to_numpy()

        row_idx = np.arange(len(df))
        df["ewp_actual"] = df.to_numpy()[row_idx, col_idx]

    # Compute best EWP
    df["ewp_best"] = df[ewp_cols].max(axis=1)

    # Mask
    valid = (df["down"] == 4) & df[ewp_cols].notna().all(axis=1)

    # decision margin only for valid rows (avoids weird NaNs)
    df["decision_margin"] = np.nan
    ewp_sorted = np.sort(df.loc[valid, ewp_cols].values, axis=1)
    df.loc[valid, "decision_margin"] = ewp_sorted[:, -1] - ewp_sorted[:, -2]
    
    # go_margin: go vs best alternative
    df["go_margin"] = np.nan
    df.loc[valid, "go_margin"] = (
        df.loc[valid, "ewp_go"]
        - df.loc[valid, ["ewp_punt", "ewp_fg"]].max(axis=1)
    )
    
    col_to_action = {
        "ewp_punt": "punt",
        "ewp_fg": "field_goal",
        "ewp_go": "go"
    }

    # determine best_col and recommended_play only for valid rows
    df["best_col"] = np.nan
    df.loc[valid, "best_col"] = df.loc[valid, ewp_cols].idxmax(axis=1)
    df["recommended_play"] = np.nan
    if valid.any():
        df.loc[valid, "recommended_play"] = (
            df.loc[valid, ewp_cols].idxmax(axis=1).map(col_to_action)
        )
    
    # For cases where we know play_type_actual
    if not test:
        df["regret_actual"] = pd.to_numeric(df["ewp_best"] - df["ewp_actual"], errors="coerce")
        
        # Identify disagreement (only meaningful when recommendation exists)
        df["disagreed"] = np.nan
        df.loc[valid, "disagreed"] = ~(
            ((df.play_type_actual == "punt") & (df.recommended_play == "punt")) |
            ((df.play_type_actual == "field_goal") & (df.recommended_play == "field_goal")) |
            ((df.play_type_actual == "go") & (df.recommended_play == "go"))
        )

        df["follow_model"] = np.nan
        df.loc[valid, "follow_model"] = (df.loc[valid, "actual_ewp_col"] == df.loc[valid, "best_col"]).astype(int)
       
    return df

def report_state(pbp_fourth, nd=4):
    r = pbp_fourth.iloc[0]

    # --- Core quantities
    wp_current = float(r.wp_current)
    p_convert  = float(r.p_convert)

    wp_succ = float(r.wp_go_success)
    wp_fail = float(r.wp_go_fail)

    # --- Deltas vs current
    go_delta = float(r.ewp_go - wp_current)
    fg_delta     = float(r.ewp_fg     - wp_current)
    punt_delta   = float(r.ewp_punt   - wp_current)

    # =========================
    print("\nTOPLINE")
    print("-" * 40)
    print(f"wp_current                    : {wp_current:.{nd}f}")
    print(f"recommended_play              : {r.recommended_play}")
    print(f"decision_margin               : {float(r.decision_margin):.{nd}f}")

    print("\nEXPECTED WIN PROBABILITIES")
    print("-" * 40)
    print(f"go                            : {float(r.ewp_go):.{nd}f}   (Δ: {go_delta:+.{nd}f})")
    print(f"field goal                    : {float(r.ewp_fg):.{nd}f}   (Δ: {fg_delta:+.{nd}f})")
    print(f"punt                          : {float(r.ewp_punt):.{nd}f}   (Δ: {punt_delta:+.{nd}f})")

    print("\nGO DETAILS")
    print("-" * 40)
    print(f"p_convert                     : {p_convert:.{nd}f}")
    print(f"wp_success                    : {wp_succ:.{nd}f}")
    print(f"wp_fail                       : {wp_fail:.{nd}f}")

    print("\nFG DETAILS")
    print("-" * 40)
    print(f"p_make_fg                     : {float(r.p_make_fg):.{nd}f}")
    print(f"wp_success                    : {float(r.wp_fg_success):.{nd}f}")
    print(f"wp_fail                       : {float(r.wp_fg_fail):.{nd}f}")

    print("\nPUNT CONTEXT")
    print("-" * 40)
    print(f"predicted net punt yds        : {float(r.punt_pred_yards):.{nd}f}")

    print()

def create_df_with_ewp(df,
                       wp_model=None,
                       go_model=None,
                       fg_model=None,
                       punt_model=None,
                       wp_features=None,
                       wp_base_features=None,
                       go_features=None,
                       fg_features=None,
                       punt_features=None,
                       test=False
                      ):

    pbp_pre_computed = df.copy()
    pbp_pre_computed = create_features(pbp_pre_computed)

    # Predict WP on the full current-state df
    pbp_pre_computed["wp_pred"] = wp_symmetric_adjust(
        pbp_pre_computed, predict_wp, wp_model, wp_features
    )

    # Outcomes only exist for real data
    if not test:
        pbp_pre_computed = create_plays_df(pbp_pre_computed)

    # current-state WP
    pbp_pre_computed["wp_current"] = wp_symmetric_adjust(
        pbp_pre_computed, predict_wp, wp_model, wp_features
    )

    # EWP components (each handles its own 4th-down mask)
    pbp_pre_computed = create_next_fg_conv_states(pbp_pre_computed)
    pbp_pre_computed = calculate_ewp_fg(pbp_pre_computed, wp_model, fg_model, wp_features, wp_base_features, fg_features)
    pbp_pre_computed = calculate_ewp_go(pbp_pre_computed, wp_model, go_model, wp_features, wp_base_features, go_features)
    pbp_pre_computed = create_punt_next_state(pbp_pre_computed, punt_model, punt_features)
    pbp_pre_computed = calculate_ewp_punt(pbp_pre_computed, wp_model, wp_features, wp_base_features)

    pbp_pre_computed = make_recommendations(pbp_pre_computed, test=test)
    pbp_fourth = pbp_pre_computed[pbp_pre_computed.down == 4].copy()

    if test:
        report_state(pbp_fourth)

    return pbp_pre_computed, pbp_fourth

from pathlib import Path
import nbformat

# Path to this notebook file
NB_PATH = Path("functions.ipynb")  # because both inside notebooks/
OUT_PATH = Path("..") / "project_code" / "functions.py"

nb = nbformat.read(NB_PATH, as_version=4)

lines = []
lines.append("# AUTO-GENERATED FROM notebooks/functions.ipynb")
lines.append("# DO NOT EDIT DIRECTLY\n")

for cell in nb.cells:
    if cell.cell_type != "code":
        continue
    src = cell.source.strip()
    if not src:
        continue

    # Heuristic: export cells that look like they define functions/classes/imports
    if ("def " in src) or ("class " in src) or src.startswith("import ") or src.startswith("from "):
        lines.append(src)
        lines.append("")  # blank line between cells

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text("\n".join(lines), encoding="utf-8")

#print(f"Wrote: {OUT_PATH.resolve()}")
