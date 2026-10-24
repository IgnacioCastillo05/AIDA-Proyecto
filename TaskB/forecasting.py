"""Entrenamiento, evaluación (MAE/RMSE), baseline y predicción de la próxima semana."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

import config
from model import SalesLSTM


@dataclass
class ForecastResult:
    series_name: str
    weeks: pd.DatetimeIndex        # semanas usadas (después de recortar historia)
    actual: np.ndarray             # valores reales de todas esas semanas
    test_weeks: pd.DatetimeIndex
    test_actual: np.ndarray
    test_pred: np.ndarray          # predicciones LSTM en el periodo de prueba
    test_baseline: np.ndarray      # predicciones de media móvil en el periodo de prueba
    next_week: pd.Timestamp
    next_pred: float
    next_baseline: float
    mae: float
    rmse: float
    baseline_mae: float
    baseline_rmse: float
    loss_history: list


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def set_seed(seed=config.SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_windows(scaled, lookback, target_indices):
    """Cada muestra usa las `lookback` semanas anteriores para predecir la semana t."""
    X = np.stack([scaled[t - lookback:t] for t in target_indices])
    y = scaled[target_indices]
    return (
        torch.tensor(X, dtype=torch.float32).unsqueeze(-1),
        torch.tensor(y, dtype=torch.float32).unsqueeze(-1),
    )


def moving_average_forecast(values, target_indices, window):
    return np.array([values[max(0, t - window):t].mean() for t in target_indices])


def train_and_forecast(
    series: pd.Series,
    lookback=config.LOOKBACK,
    test_weeks=config.TEST_WEEKS,
    epochs=config.EPOCHS,
    history_weeks=None,
    hidden_size=config.HIDDEN_SIZE,
    num_layers=config.NUM_LAYERS,
    lr=config.LEARNING_RATE,
    ma_window=config.MOVING_AVG_WINDOW,
    seed=config.SEED,
) -> ForecastResult:
    """Entrena un LSTM sobre una serie semanal y lo evalúa en las últimas `test_weeks`.

    history_weeks: si se indica, solo se usan las últimas N semanas de la serie
    (permite ver el efecto de entrenar con más o menos historia).
    """
    set_seed(seed)
    if history_weeks:
        series = series.iloc[-history_weeks:]

    values = series.to_numpy(dtype=np.float64)
    n = len(values)
    split = n - test_weeks
    if split - lookback < 4:
        raise ValueError(
            f"Muy pocas semanas ({n}) para lookback={lookback} y test={test_weeks}. "
            "Reduce lookback/test o usa más historia."
        )

    # Escalado min-max ajustado SOLO con el periodo de entrenamiento (evita fuga de datos)
    lo, hi = values[:split].min(), values[:split].max()
    scale = hi - lo if hi > lo else 1.0
    scaled = (values - lo) / scale

    train_idx = np.arange(lookback, split)
    test_idx = np.arange(split, n)
    X_train, y_train = make_windows(scaled, lookback, train_idx)
    X_test, _ = make_windows(scaled, lookback, test_idx)

    model = SalesLSTM(hidden_size=hidden_size, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loss_history = []

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

    model.eval()
    with torch.no_grad():
        test_pred_scaled = model(X_test).squeeze(-1).numpy()
        last_window = torch.tensor(scaled[n - lookback:], dtype=torch.float32).view(1, lookback, 1)
        next_scaled = model(last_window).item()

    # Volver a la escala original; las ventas no pueden ser negativas
    test_pred = np.clip(test_pred_scaled * scale + lo, 0, None)
    next_pred = max(0.0, next_scaled * scale + lo)

    test_actual = values[test_idx]
    test_baseline = moving_average_forecast(values, test_idx, ma_window)
    next_baseline = float(values[-ma_window:].mean())

    weeks = series.index
    freq = weeks.freq or pd.infer_freq(weeks) or config.WEEK_FREQ
    next_week = weeks[-1] + pd.tseries.frequencies.to_offset(freq)

    return ForecastResult(
        series_name=str(series.name),
        weeks=weeks,
        actual=values,
        test_weeks=weeks[test_idx],
        test_actual=test_actual,
        test_pred=test_pred,
        test_baseline=test_baseline,
        next_week=next_week,
        next_pred=float(next_pred),
        next_baseline=next_baseline,
        mae=mae(test_actual, test_pred),
        rmse=rmse(test_actual, test_pred),
        baseline_mae=mae(test_actual, test_baseline),
        baseline_rmse=rmse(test_actual, test_baseline),
        loss_history=loss_history,
    )


def result_to_frame(result: ForecastResult) -> pd.DataFrame:
    """Tabla con las predicciones de prueba + la de la próxima semana (para exportar a CSV)."""
    rows = pd.DataFrame({
        "series": result.series_name,
        "week": result.test_weeks,
        "type": "test",
        "actual": result.test_actual,
        "lstm_pred": result.test_pred,
        "moving_avg_pred": result.test_baseline,
    })
    nxt = pd.DataFrame([{
        "series": result.series_name,
        "week": result.next_week,
        "type": "next_week_forecast",
        "actual": np.nan,
        "lstm_pred": result.next_pred,
        "moving_avg_pred": result.next_baseline,
    }])
    return pd.concat([rows, nxt], ignore_index=True)


def export_predictions(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.concat([result_to_frame(r) for r in results], ignore_index=True)
    frame.to_csv(path, index=False, float_format="%.2f")
    return path
