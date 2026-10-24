"""Entrena y evalúa todas las series desde la terminal (útil para el informe).

Uso:
    python train.py
    python train.py --epochs 500 --lookback 6 --test-weeks 8
    python train.py --data data/sample_retail_sales.csv
"""
import argparse
from pathlib import Path

import pandas as pd

import config
from data_prep import load_weekly_series
from forecasting import export_predictions, train_and_forecast


def main():
    parser = argparse.ArgumentParser(description="Tarea B - Pronóstico semanal con LSTM")
    parser.add_argument("--data", type=Path, default=config.DATA_PATH,
                        help="Ruta al CSV de ventas")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--lookback", type=int, default=config.LOOKBACK)
    parser.add_argument("--test-weeks", type=int, default=config.TEST_WEEKS)
    parser.add_argument("--history-weeks", type=int, default=None,
                        help="Usar solo las últimas N semanas (por defecto, todas)")
    args = parser.parse_args()

    weekly = load_weekly_series(args.data)
    print(f"Datos: {len(weekly)} semanas, {weekly.shape[1]} series "
          f"({weekly.index.min():%Y-%m-%d} a {weekly.index.max():%Y-%m-%d})\n")

    results, rows = [], []
    for name in weekly.columns:
        r = train_and_forecast(
            weekly[name],
            lookback=args.lookback,
            test_weeks=args.test_weeks,
            epochs=args.epochs,
            history_weeks=args.history_weeks,
        )
        results.append(r)
        rows.append({
            "Serie": name,
            "MAE LSTM": r.mae,
            "RMSE LSTM": r.rmse,
            "MAE Media móvil": r.baseline_mae,
            "RMSE Media móvil": r.baseline_rmse,
            f"Pronóstico {r.next_week:%Y-%m-%d}": r.next_pred,
        })

    summary = pd.DataFrame(rows).set_index("Serie")
    pd.set_option("display.width", 200)
    print(summary.round(1).to_string())
    print("\nPromedio:")
    print(summary.iloc[:, :4].mean().round(1).to_string())

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.OUTPUT_DIR / "metrics_summary.csv", float_format="%.2f")
    path = export_predictions(results, config.OUTPUT_DIR / "predictions_all.csv")
    print(f"\nGuardado: {config.OUTPUT_DIR / 'metrics_summary.csv'}")
    print(f"Guardado: {path}")


if __name__ == "__main__":
    main()
