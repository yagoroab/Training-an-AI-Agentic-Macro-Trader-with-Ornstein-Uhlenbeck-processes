from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf


def set_global_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_lstm_feature_frame(vix: pd.Series, traded: pd.Series) -> pd.DataFrame:
    df = pd.concat(
        [
            vix.rename("vix"),
            traded.rename("traded"),
        ],
        axis=1,
    ).dropna().sort_index()

    log_vix = np.log(df["vix"])
    log_traded = np.log(df["traded"])
    traded_ret = df["traded"].pct_change()
    vix_ret = df["vix"].pct_change()

    feat = pd.DataFrame(index=df.index)
    feat["log_vix"] = log_vix
    feat["log_traded"] = log_traded
    feat["log_vix_over_traded"] = log_vix - log_traded
    feat["vix_ret_1"] = vix_ret
    feat["traded_ret_1"] = traded_ret
    feat["vix_ret_5"] = df["vix"].pct_change(5)
    feat["traded_ret_5"] = df["traded"].pct_change(5)
    feat["vix_ret_20"] = df["vix"].pct_change(20)
    feat["traded_ret_20"] = df["traded"].pct_change(20)
    feat["vix_vol_20"] = vix_ret.rolling(20).std()
    feat["traded_vol_20"] = traded_ret.rolling(20).std()
    # Predict a 5-day forward return direction to reduce one-day noise.
    feat["target_ret_5d"] = df["traded"].pct_change(5).shift(-5)
    feat["traded"] = df["traded"]
    feat["vix"] = df["vix"]
    return feat.dropna().copy()


def _fit_standardizer(train_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train_values, axis=0)
    std = np.std(train_values, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def _apply_standardizer(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (values - mean) / std


def build_lstm_sequences(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray]:
    values = feature_df[feature_cols].to_numpy(dtype=np.float32)
    target = feature_df[target_col].to_numpy(dtype=np.float32)
    traded = feature_df["traded"].to_numpy(dtype=np.float32)
    dates = feature_df.index

    x_seq: list[np.ndarray] = []
    y_seq: list[float] = []
    seq_dates: list[pd.Timestamp] = []
    traded_px: list[float] = []

    for end_idx in range(sequence_length - 1, len(feature_df) - 1):
        start_idx = end_idx - sequence_length + 1
        x_seq.append(values[start_idx : end_idx + 1])
        y_seq.append(target[end_idx])
        seq_dates.append(dates[end_idx])
        traded_px.append(traded[end_idx])

    return (
        np.asarray(x_seq, dtype=np.float32),
        np.asarray(y_seq, dtype=np.float32),
        pd.DatetimeIndex(seq_dates),
        np.asarray(traded_px, dtype=np.float32),
    )


def train_lstm_regressor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int = 42,
    epochs: int = 40,
    batch_size: int = 32,
) -> tf.keras.Model:
    set_global_seed(seed)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
            tf.keras.layers.LSTM(32, dropout=0.10, recurrent_dropout=0.00),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.Huber(),
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
        )
    ]

    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0,
        callbacks=callbacks,
        shuffle=False,
    )
    return model


def train_lstm_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int = 42,
    epochs: int = 30,
    batch_size: int = 32,
) -> tf.keras.Model:
    set_global_seed(seed)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
            tf.keras.layers.LSTM(16, dropout=0.20, recurrent_dropout=0.00),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        )
    ]

    model.fit(
        x_train,
        y_train,
        epochs=min(epochs, 20),
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0,
        callbacks=callbacks,
        shuffle=False,
    )
    return model


def train_gru_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int = 42,
    epochs: int = 30,
    batch_size: int = 32,
) -> tf.keras.Model:
    set_global_seed(seed)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
            tf.keras.layers.GRU(16, dropout=0.20, recurrent_dropout=0.00),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        )
    ]

    model.fit(
        x_train,
        y_train,
        epochs=min(epochs, 20),
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0,
        callbacks=callbacks,
        shuffle=False,
    )
    return model


def backtest_lstm_predictions(
    dates: pd.DatetimeIndex,
    traded_price: np.ndarray,
    predicted_ret: np.ndarray,
    *,
    cash_yield_annual: float = 0.03,
    cost_bps: float = 1.0,
    max_leverage: float = 1.00,
    signal_threshold: float = 0.10,
    pos_ema_alpha: float = 0.50,
    rebalance_thresh: float = 0.10,
) -> dict:
    traded_price = np.asarray(traded_price, dtype=float)
    predicted_ret = np.asarray(predicted_ret, dtype=float)
    trade_cost_rate = float(cost_bps) * 1e-4
    daily_cash_yield = (1.0 + float(cash_yield_annual)) ** (1.0 / 252.0) - 1.0

    n = len(predicted_ret)
    traded_ret = np.zeros(n, dtype=float)
    for t in range(1, n):
        if traded_price[t - 1] > 0:
            traded_ret[t] = traded_price[t] / traded_price[t - 1] - 1.0

    pred_scale = float(np.std(predicted_ret[: max(20, n // 4)]))
    pred_scale = max(pred_scale, 1e-4)

    pos = 0.0
    risky_weight = 0.0
    cash_weight = 1.0

    pnl = np.zeros(n, dtype=float)
    pos_arr = np.zeros(n, dtype=float)
    risky_weight_arr = np.zeros(n, dtype=float)
    cash_weight_arr = np.ones(n, dtype=float)

    for t in range(1, n):
        ret = traded_ret[t]
        pnl[t] = risky_weight * np.sign(pos) * ret + cash_weight * daily_cash_yield

        strength = predicted_ret[t] / pred_scale
        if abs(strength) < signal_threshold:
            target_pos = 0.0
        else:
            target_pos = float(max_leverage * np.tanh(strength))

        if abs(target_pos - pos) < rebalance_thresh:
            new_pos = pos
        else:
            new_pos = (1.0 - pos_ema_alpha) * pos + pos_ema_alpha * target_pos

        pnl[t] -= trade_cost_rate * abs(new_pos - pos)

        risky_weight = min(abs(new_pos), max_leverage)
        cash_weight = max(0.0, 1.0 - risky_weight)
        pos = new_pos

        pos_arr[t] = pos
        risky_weight_arr[t] = risky_weight
        cash_weight_arr[t] = cash_weight

    wealth = np.cumprod(1.0 + np.clip(pnl, -0.99, None))

    return {
        "dates": dates,
        "pnl": pnl,
        "wealth": wealth,
        "pos": pos_arr,
        "risky_weight": risky_weight_arr,
        "cash_weight": cash_weight_arr,
        "predicted_ret": predicted_ret,
        "traded_ret": traded_ret,
    }


def backtest_lstm_probabilities(
    dates: pd.DatetimeIndex,
    traded_price: np.ndarray,
    probability_up: np.ndarray,
    *,
    cash_yield_annual: float = 0.03,
    cost_bps: float = 1.0,
    max_leverage: float = 1.00,
    probability_scale: float = 6.0,
    deadband: float = 0.02,
    pos_ema_alpha: float = 0.50,
    rebalance_thresh: float = 0.10,
) -> dict:
    traded_price = np.asarray(traded_price, dtype=float)
    probability_up = np.asarray(probability_up, dtype=float)
    trade_cost_rate = float(cost_bps) * 1e-4
    daily_cash_yield = (1.0 + float(cash_yield_annual)) ** (1.0 / 252.0) - 1.0

    n = len(probability_up)
    traded_ret = np.zeros(n, dtype=float)
    for t in range(1, n):
        if traded_price[t - 1] > 0:
            traded_ret[t] = traded_price[t] / traded_price[t - 1] - 1.0

    pos = 0.0
    risky_weight = 0.0
    cash_weight = 1.0

    pnl = np.zeros(n, dtype=float)
    pos_arr = np.zeros(n, dtype=float)
    risky_weight_arr = np.zeros(n, dtype=float)
    cash_weight_arr = np.ones(n, dtype=float)

    for t in range(1, n):
        ret = traded_ret[t]
        pnl[t] = risky_weight * np.sign(pos) * ret + cash_weight * daily_cash_yield

        centered_prob = float(probability_up[t] - 0.5)
        if abs(centered_prob) <= deadband:
            target_pos = 0.0
        else:
            signal = np.tanh(probability_scale * centered_prob)
            target_pos = float(max_leverage * signal)

        if abs(target_pos - pos) < rebalance_thresh:
            new_pos = pos
        else:
            new_pos = (1.0 - pos_ema_alpha) * pos + pos_ema_alpha * target_pos

        pnl[t] -= trade_cost_rate * abs(new_pos - pos)

        risky_weight = min(abs(new_pos), max_leverage)
        cash_weight = max(0.0, 1.0 - risky_weight)
        pos = new_pos

        pos_arr[t] = pos
        risky_weight_arr[t] = risky_weight
        cash_weight_arr[t] = cash_weight

    wealth = np.cumprod(1.0 + np.clip(pnl, -0.99, None))

    return {
        "dates": dates,
        "pnl": pnl,
        "wealth": wealth,
        "pos": pos_arr,
        "risky_weight": risky_weight_arr,
        "cash_weight": cash_weight_arr,
        "probability_up": probability_up,
        "traded_ret": traded_ret,
    }


def prepare_lstm_dataset(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    sequence_length: int = 30,
    split_date: str = "2020-01-01",
    target_col: str = "target_ret_5d",
) -> dict:
    x_seq, y_seq, dates, traded_px = build_lstm_sequences(
        feature_df=feature_df,
        feature_cols=feature_cols,
        target_col=target_col,
        sequence_length=sequence_length,
    )

    train_mask = dates < pd.to_datetime(split_date)
    if train_mask.sum() < 50:
        raise ValueError("Training sample is too small for LSTM training.")

    mean, std = _fit_standardizer(x_seq[train_mask].reshape(-1, len(feature_cols)))
    x_scaled = _apply_standardizer(x_seq, mean, std)

    return {
        "x": x_scaled.astype(np.float32),
        "y": y_seq.astype(np.float32),
        "dates": dates,
        "traded_price": traded_px.astype(np.float32),
        "train_mask": train_mask,
        "feature_mean": mean,
        "feature_std": std,
    }
