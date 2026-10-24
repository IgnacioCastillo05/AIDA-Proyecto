# Tarea B — Pronóstico de demanda (LSTM)

Parte del proyecto UrbanStyle. Predice las ventas de la próxima semana por categoría
con un LSTM pequeño, lo evalúa con MAE y RMSE y lo compara con un baseline de media
móvil. Incluye una interfaz en Pygame.

## Estructura

```
task_b_forecasting/
├── config.py            # Rutas e hiperparámetros
├── data_prep.py         # CSV -> series semanales por categoría
├── model.py             # Definición del LSTM (PyTorch)
├── forecasting.py       # Entrenamiento, MAE/RMSE, media móvil, exportación CSV
├── train.py             # Corre todas las series desde la terminal (tabla para el informe)
├── app.py               # Interfaz Pygame
├── make_sample_data.py  # Datos sintéticos para probar sin el dataset real
├── requirements.txt
├── data/                # Aquí va el CSV de Kaggle
└── outputs/             # Predicciones y métricas exportadas
```

## Instalación

```bash
cd task_b_forecasting
python -m venv .venv
# Windows: .venv\Scripts\activate     |  Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Datos

1. Descargar el dataset: https://www.kaggle.com/datasets/mohammadtalib786/retail-sales-dataset
2. Copiar `retail_sales_dataset.csv` en `task_b_forecasting/data/`.

Columnas usadas: `Date`, `Gender`, `Age`, `Product Category`, `Quantity`, `Total Amount`.

Para probar antes de tener el dataset: `python make_sample_data.py` y luego agregar
`--data data/sample_retail_sales.csv` a los comandos. Esos datos son aleatorios y no
sirven para el informe.

## Uso

```bash
python train.py          # entrena las 6 series, imprime MAE/RMSE y exporta a outputs/
python app.py            # interfaz Pygame
```

Opciones de `train.py`: `--epochs`, `--lookback`, `--test-weeks`, `--history-weeks`, `--data`.

### Controles de la app

| Control | Función |
|---|---|
| Botones de serie | Elegir la categoría (reentrena automáticamente) |
| Slider Épocas | Duración del entrenamiento (50–1000) |
| Slider Lookback | Cuántas semanas anteriores ve el modelo (2–8) |
| Slider Semanas de historia | Cuánta historia se usa para entrenar |
| Entrenar | Aplica los valores de los sliders (aparece `*` si hay cambios pendientes) |
| Media móvil | Mostrar u ocultar el baseline |
| Exportar CSV | Guarda las predicciones de la serie actual en `outputs/` |
| Esc | Salir |

## Metodología

1. **Series:** el dataset trae solo 3 categorías (Beauty, Clothing, Electronics). Para
   cumplir con las 5–8 categorías del enunciado se cruza **categoría × género**, lo que
   da 6 series. Se puede cambiar en `config.SEGMENTATION`: `category` (3),
   `category_gender` (6) o `category_age` (9).
2. **Preprocesamiento:** se suma `Total Amount` por semana (semanas que terminan en
   lunes). Las semanas sin ventas quedan en 0.
3. **División temporal:** las últimas 8 semanas son de prueba y el resto de
   entrenamiento. El escalado min-max se ajusta solo con el periodo de entrenamiento,
   para no filtrar información del futuro.
4. **Modelo:** LSTM de 1 capa con 32 neuronas y una capa lineal de salida. Recibe las
   últimas `lookback` semanas y predice la siguiente. Se entrena con Adam y pérdida MSE,
   con semilla fija para que los resultados sean reproducibles.
5. **Evaluación:** predicción de un paso (una semana) en el periodo de prueba, con MAE y
   RMSE.
6. **Baseline (opcional):** media móvil de 4 semanas.
7. **Próxima semana:** se predice a partir de las últimas `lookback` semanas observadas.
8. **Exportación CSV (opcional):** `outputs/predictions_all.csv` y
   `outputs/metrics_summary.csv`.

## Limitaciones

- Hay unas 53 semanas por serie y pocas transacciones por semana, así que las series
  son ruidosas. Es normal que la media móvil compita con el LSTM o le gane, y eso
  también es un hallazgo para el informe.
- La primera semana es parcial, porque los datos empiezan a mitad de esa semana.
