# Informe — Tarea B: Pronóstico de Demanda con LSTM

**Proyecto:** UrbanStyle — Retail de moda
**Tarea:** B — Demand Forecasting (Time-Series con LSTM pequeño)
**Autor(es):** Ignacio Castillo, Sofía Ariza, Daniel Palacios, Juan Daid Rodríguez
**Fecha:** 31 de Octubre del 2026

---

## 1. Contexto de negocio

UrbanStyle es un retailer de moda que necesita anticipar la demanda de sus productos para tomar mejores decisiones de inventario. Si compra de más, el capital queda inmovilizado y aumentan los descuentos de liquidación. Si compra de menos, se pierden ventas por falta de stock.

El objetivo de esta tarea es **predecir las ventas de la próxima semana por categoría de producto** con una red neuronal recurrente (LSTM), y medir qué tan confiables son esas predicciones con métricas de error estándar.

## 2. Objetivos

1. Seleccionar entre 5 y 8 series (categorías) de ventas.
2. Transformar las transacciones en **series de tiempo semanales**.
3. Entrenar un **LSTM pequeño** que prediga la venta de la semana siguiente.
4. Pronosticar las ventas de la próxima semana.
5. Evaluar el modelo con **MAE** y **RMSE**.
6. *(Opcional)* Compararlo con un **baseline de media móvil** y **exportar las predicciones a CSV**.
7. Construir una interfaz interactiva en **Pygame** para explorar los pronósticos.

## 3. Datos

**Fuente:** [Retail Sales Dataset — Kaggle](https://www.kaggle.com/datasets/mohammadtalib786/retail-sales-dataset)

| Característica | Valor |
|---|---|
| Registros (transacciones) | [COMPLETAR — p. ej. 1.000] |
| Periodo | [COMPLETAR — p. ej. ene-2023 a ene-2024] |
| Categorías de producto | Beauty, Clothing, Electronics |
| Columnas usadas | `Date`, `Gender`, `Age`, `Product Category`, `Quantity`, `Total Amount` |

**Variable objetivo:** `Total Amount`, el ingreso total por transacción, sumado por semana.

### 3.1 Definición de las series

El enunciado pide entre 5 y 8 categorías, pero el dataset trae solo **3 categorías de producto**. Para cumplir el requisito sin inventar datos, cada categoría se dividió por **género del cliente**, lo que da **6 series**:

| # | Serie |
|---|---|
| 1 | Beauty - Female |
| 2 | Beauty - Male |
| 3 | Clothing - Female |
| 4 | Clothing - Male |
| 5 | Electronics - Female |
| 6 | Electronics - Male |

Esta segmentación también tiene sentido para el negocio, porque la oferta y la demanda en moda suelen gestionarse por línea (dama y caballero). El código permite cambiarla en `config.py` (`SEGMENTATION`): `category` da 3 series y `category_age`, por grupos de edad, da 9.

## 4. Preprocesamiento

1. **Conversión de fechas:** la columna `Date` se pasa a formato fecha.
2. **Agregación semanal:** se suma `Total Amount` por serie y por semana. Las semanas terminan en lunes (`W-MON`).
3. **Semanas sin ventas:** se rellenan con 0 para que la serie sea continua.
4. **División temporal:**
   - **Entrenamiento:** todas las semanas excepto las últimas 8.
   - **Prueba:** las **últimas 8 semanas**, que el modelo no ve durante el entrenamiento.
   - La división respeta el orden cronológico; no se mezclan las semanas al azar, porque eso usaría el futuro para predecir el pasado.
5. **Escalado min-max a [0, 1]:** se ajusta **solo con los datos de entrenamiento**, para que no se filtre información del periodo de prueba.
6. **Ventanas deslizantes:** cada muestra usa las últimas `lookback = 4` semanas como entrada y la semana siguiente como objetivo.

```
[ s1, s2, s3, s4 ] → s5
[ s2, s3, s4, s5 ] → s6
...
```

**Resultado del preprocesamiento:** [COMPLETAR — p. ej. "53 semanas por serie, 6 series"; se obtiene con `python data_prep.py`]

## 5. Modelo

Se usó un **LSTM pequeño** implementado en PyTorch:

| Componente | Configuración |
|---|---|
| Entrada | Secuencia de 4 semanas × 1 variable |
| Capa LSTM | 1 capa, 32 neuronas ocultas |
| Capa de salida | Lineal (32 → 1) |
| Salida | Venta de la semana siguiente (escalada) |
| Función de pérdida | Error cuadrático medio (MSE) |
| Optimizador | Adam, tasa de aprendizaje 0.01 |
| Épocas | 100 |
| Semilla aleatoria | 42, para que los resultados sean reproducibles |

**Por qué un LSTM:** es una red recurrente diseñada para secuencias. Retiene información de semanas anteriores para capturar tendencias y patrones. Se eligió un modelo **pequeño** a propósito, porque con pocas semanas de historia uno grande se aprendería el ruido de memoria (sobreajuste).

Se entrena **un modelo independiente por cada serie**.

## 6. Métricas de evaluación

Ambas métricas están en la misma unidad que las ventas (dólares), así que se interpretan directamente. Siendo $y_t$ la venta real, $\hat{y}_t$ la predicción y $n$ el número de semanas de prueba:

$$\text{MAE} = \frac{1}{n}\sum_{t=1}^{n} |y_t - \hat{y}_t|$$

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{t=1}^{n} (y_t - \hat{y}_t)^2}$$

- **MAE:** cuánto se equivoca el modelo, en promedio, cada semana.
- **RMSE:** castiga más los errores grandes. Si el RMSE es mucho mayor que el MAE, hay semanas con errores muy fuertes.

### 6.1 Baseline: media móvil (opcional)

Para saber si el LSTM aporta valor, se comparó con un método simple: **predecir que la próxima semana venderá el promedio de las últimas 4 semanas**. Un modelo de machine learning solo se justifica si supera, o al menos iguala, a una regla tan sencilla.

## 7. Resultados

> Se obtienen ejecutando `python train.py`. Los valores quedan en `outputs/metrics_summary.csv`.

### 7.1 Métricas por serie (periodo de prueba: últimas 8 semanas)

| Serie | MAE LSTM | RMSE LSTM | MAE Media móvil | RMSE Media móvil | Mejor modelo |
|---|---|---|---|---|---|
| Beauty - Female | [ ] | [ ] | [ ] | [ ] | [ ] |
| Beauty - Male | [ ] | [ ] | [ ] | [ ] | [ ] |
| Clothing - Female | [ ] | [ ] | [ ] | [ ] | [ ] |
| Clothing - Male | [ ] | [ ] | [ ] | [ ] | [ ] |
| Electronics - Female | [ ] | [ ] | [ ] | [ ] | [ ] |
| Electronics - Male | [ ] | [ ] | [ ] | [ ] | [ ] |
| **Promedio** | **[ ]** | **[ ]** | **[ ]** | **[ ]** | |

### 7.2 Pronóstico de la próxima semana

| Serie | Semana | Pronóstico LSTM | Pronóstico media móvil |
|---|---|---|---|
| Beauty - Female | [ ] | [ ] | [ ] |
| Beauty - Male | [ ] | [ ] | [ ] |
| Clothing - Female | [ ] | [ ] | [ ] |
| Clothing - Male | [ ] | [ ] | [ ] |
| Electronics - Female | [ ] | [ ] | [ ] |
| Electronics - Male | [ ] | [ ] | [ ] |

*Los valores vienen de las filas `next_week_forecast` de `outputs/predictions_all.csv`.*

### 7.3 Visualización

![Pronóstico vs real — app Pygame](report/img/app_forecast.png)

*Figura 1. La línea gris son las ventas reales, la naranja el pronóstico del LSTM y la verde la media móvil. La zona sombreada es el periodo de prueba y el círculo final es el pronóstico de la próxima semana.*

### 7.4 Experimentos con hiperparámetros (opcional)

| Configuración | Comando | MAE promedio LSTM | RMSE promedio LSTM |
|---|---|---|---|
| Base (100 épocas, lookback 4) | `python train.py` | [ ] | [ ] |
| Más épocas | `python train.py --epochs 300` | [ ] | [ ] |
| Menos épocas | `python train.py --epochs 50` | [ ] | [ ] |
| Ventana más larga | `python train.py --lookback 6` | [ ] | [ ] |
| Menos historia | `python train.py --history-weeks 30` | [ ] | [ ] |

## 8. Interfaz Pygame

La aplicación (`python app.py`) permite explorar los pronósticos de forma interactiva:

| Control | Función |
|---|---|
| Botones de serie | Elegir la categoría que se pronostica (reentrena automáticamente) |
| Slider **Épocas** | Ajustar la duración del entrenamiento (50–1000) |
| Slider **Lookback** | Cuántas semanas anteriores usa el modelo como entrada (2–8) |
| Slider **Semanas de historia** | Cuánta historia se usa para entrenar |
| Botón **Entrenar** | Reentrenar con la configuración elegida |
| Botón **Media móvil** | Mostrar u ocultar el baseline |
| Botón **Exportar CSV** | Guardar las predicciones de la serie actual |

La pantalla muestra la gráfica de ventas reales contra el pronóstico, una tabla con el MAE y el RMSE del LSTM y de la media móvil, el pronóstico de la próxima semana y la curva de pérdida del entrenamiento.

![Interfaz Pygame](report/img/app_ui.png)

*Figura 2. Interfaz completa de la aplicación.*

## 9. Discusión

> Ajustar según los resultados reales. Abajo van guías para la redacción.

- **¿Superó el LSTM a la media móvil?** [COMPLETAR — p. ej. "El LSTM obtuvo menor MAE en X de 6 series..."]
- **Series más predecibles y menos predecibles:** [COMPLETAR — cuáles tienen menor y mayor error, y por qué podría ser: más o menos volumen, más o menos volatilidad]
- **MAE frente a RMSE:** [COMPLETAR — si el RMSE es bastante mayor que el MAE, hay semanas con picos que el modelo no anticipa, por ejemplo compras grandes de Electronics]
- **Efecto de los hiperparámetros:** [COMPLETAR — p. ej. "más épocas no siempre mejoraron el error en prueba, lo que indica sobreajuste"]

**Interpretación esperada:** con cerca de un año de datos (unas 53 semanas) y pocas transacciones por semana, las series son **cortas y ruidosas**. En ese escenario es común que un método simple como la media móvil compita con el LSTM o le gane. Es un hallazgo válido: muestra que **la complejidad del modelo debe ir de la mano con la cantidad de datos disponibles**.

## 10. Limitaciones

1. **Poca historia:** cerca de 53 semanas por serie no alcanzan para capturar estacionalidad anual, que requeriría al menos 2 o 3 años.
2. **Bajo volumen por serie:** al dividir por género, cada serie tiene pocas transacciones por semana, lo que aumenta el ruido.
3. **Una sola variable de entrada:** el modelo solo ve las ventas pasadas, sin promociones, precios, festivos ni clima.
4. **Primera semana parcial:** los datos empiezan a mitad de semana, así que la primera semana tiene menos ventas de las reales.
5. **Periodo de prueba corto:** 8 semanas dan una estimación del error con alta variabilidad.

## 11. Posibles modificaciones a futuro

- Agregar **variables externas**: festivos, temporadas como Black Friday o Navidad, precio promedio.
- Entrenar un **modelo global** con todas las series a la vez, para aprovechar más datos.
- Probar **validación cruzada temporal** (walk-forward) para medir el error de forma más robusta.
- Comparar con modelos clásicos como **ARIMA** o **suavizamiento exponencial**.
- Pronosticar **unidades** (`Quantity`) además de ingresos, porque es más directo para el inventario. Basta con cambiar `TARGET_COLUMN` en `config.py`.

## 12. Reproducibilidad

```bash
cd task_b_forecasting
python -m venv .venv
.venv\Scripts\activate            # Windows  (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
# Colocar el CSV de Kaggle en data/retail_sales_dataset.csv
python train.py                   # métricas y CSV en outputs/
python app.py                     # interfaz Pygame
```

Todas las ejecuciones usan semilla fija (`SEED = 42`), así que los resultados se repiten. La configuración completa está en `config.py`.

---

## Anexo A — Cumplimiento de la rúbrica (Tarea B, 20%)

| Criterio | Estado | Evidencia |
|---|---|---|
| Seleccionar 5–8 categorías | ✅ | 6 series (categoría × género), sección 3.1 |
| Preprocesar como serie semanal | ✅ | Sección 4, `data_prep.py` |
| Entrenar un LSTM pequeño | ✅ | Sección 5, `model.py`, `forecasting.py` |
| Predecir la venta de la próxima semana | ✅ | Sección 7.2 |
| Evaluar con MAE y RMSE | ✅ | Secciones 6 y 7.1 |
| Pygame: gráfica y sliders (categoría y entrenamiento) | ✅ | Sección 8, `app.py` |
| *Opcional:* baseline de media móvil | ✅ | Secciones 6.1 y 7.1 |
| *Opcional:* exportar predicciones a CSV | ✅ | `outputs/predictions_all.csv` |

## Anexo B — Estructura del código

| Archivo | Descripción |
|---|---|
| `config.py` | Rutas e hiperparámetros |
| `data_prep.py` | Carga del CSV y construcción de las series semanales |
| `model.py` | Definición del LSTM |
| `forecasting.py` | Entrenamiento, métricas, baseline y exportación |
| `train.py` | Ejecución por terminal de todas las series |
| `app.py` | Interfaz Pygame |
| `make_sample_data.py` | Datos sintéticos, solo para pruebas |
