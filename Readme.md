# Modelatón - Equipo 7
## Predicción del consumo energético de electrodomésticos

Este proyecto responde a la pregunta: **¿Es posible predecir el consumo energético
de los electrodomésticos de una vivienda a partir de las condiciones ambientales
interiores y exteriores?**

Para ello se entrenan y comparan tres modelos de regresión sobre el dataset
*Appliances Energy Prediction* (UCI Machine Learning Repository).

---

## Dataset

- **Fuente:** UCI Machine Learning Repository - Appliances Energy Prediction
- **DOI:** 10.24432/C5VC8G
- **Registros:** 19,735 mediciones cada 10 minutos (~4.5 meses)
- **Variables:** 28 columnas (temperatura y humedad interior/exterior, presión,
  viento, visibilidad, punto de rocío, consumo eléctrico)
- **Variable objetivo:** `Appliances` (consumo energético en Wh)

El script carga los datos automáticamente desde la URL oficial del repositorio UCI,
por lo que no es necesario descargarlos manualmente.

---

## Requisitos

- **Python:** 3.9 o superior (probado en Python 3.11)
- **Librerías:** ver `requirements.txt`

---

## Instalación

Desde la carpeta del proyecto, abrir una terminal y ejecutar:

```bash
pip install -r requirements.txt
```

Esto instala las siguientes librerías:

- pandas
- numpy (versión < 2, por compatibilidad)
- matplotlib
- seaborn
- scikit-learn

---

## Ejecución

Una vez instaladas las dependencias, ejecutar el script principal:

```bash
python modelaton_equipo_7.py
```

El script realiza el análisis completo de forma secuencial e imprime los
resultados en consola. Además, abre varias ventanas con gráficas de análisis
que deben cerrarse manualmente para que el script continúe.

---

## Estructura del análisis

El script está organizado en las siguientes secciones:

1. **Carga y exploración inicial del dataset.**
2. **Transformación de la columna de fecha** en variables temporales
   (hora, día de la semana, fin de semana).
3. **Separación cronológica 80/20** del dataset (entrenamiento vs. prueba
   simulando datos futuros).
4. **Selección de variables** y análisis de correlación con heatmaps.
5. **Entrenamiento de tres modelos:**
   - Regresión Lineal
   - Random Forest
   - Gradient Boosting
6. **Evaluación con métricas MAE, RMSE y R².**
7. **Análisis comparativo con split aleatorio** para contrastar el efecto
   de la estacionalidad.
8. **Gráficas de resultados** e importancia de variables.
9. **Recomendaciones de ahorro energético** basadas en las variables más
   influyentes según los modelos.

---

---

## Hallazgo principal

### El split cronológico expone una limitación fundamental de los modelos de árboles

El dataset cubre el periodo enero-mayo de 2016. Al aplicar un split cronológico
80/20, el conjunto de entrenamiento queda restringido a invierno e inicio de
primavera, mientras que el conjunto de prueba corresponde a un periodo de
primavera avanzada. Esto genera un desfase estacional significativo en las
variables predictoras:

| Variable            | Media en train | Media en test | Cambio    |
|---------------------|---------------:|--------------:|----------:|
| Temperatura exterior (T_out) |    5.9 °C |       13.6 °C |   +7.8 °C |
| Humedad exterior (RH_out)    |    81.8 % |       71.4 % |   -10.4 % |
| Temperatura interior (T2)    |   19.6 °C |       23.2 °C |   +3.6 °C |
| Temperatura habitación 6     |    6.1 °C |       15.0 °C |   +8.8 °C |

Los valores del target (`Appliances`) se mantienen prácticamente iguales entre
train (media ≈ 98 Wh) y test (media ≈ 96 Wh), por lo que el cambio de régimen
ocurre en las **features**, no en la variable objetivo.

### Consecuencia: los modelos de árboles fallan por incapacidad de extrapolar

Random Forest y Gradient Boosting presentan una limitación conocida: **no pueden
predecir valores más allá del rango observado durante el entrenamiento**. Al
enfrentarse a temperaturas que nunca vieron en train, quedan atrapados en las
hojas terminales asociadas a los días más cálidos del entrenamiento (que
casualmente coinciden con momentos de alta actividad y alto consumo). El
resultado es una sobreestimación sistemática:

| Modelo             | Media de predicción | Valor real medio | R²      |
|--------------------|--------------------:|-----------------:|--------:|
| Random Forest      |            ≈ 276 Wh |         ≈ 96 Wh  |  ~-4.8  |
| Gradient Boosting  |            ≈ 280 Wh |         ≈ 96 Wh  |  ~-5.8  |
| Regresión lineal   |             ≈ 96 Wh |         ≈ 96 Wh  |   ~0.09 |

La regresión lineal, al basarse en una fórmula continua, mantiene la capacidad
de extrapolación y por eso obtiene el mejor desempeño relativo bajo este
escenario.

### Validación con split aleatorio

Para confirmar que la mala performance de los modelos de árboles se debe al
efecto estacional y no a un problema intrínseco de los modelos, se ejecutó
también un split aleatorio 80/20 (donde train y test comparten el mismo rango
de condiciones ambientales). Los resultados invierten completamente el orden:

| Modelo             | R² (cronológico) | R² (aleatorio) |
|--------------------|-----------------:|---------------:|
| Regresión lineal   |            ~0.09 |          ~0.15 |
| Random Forest      |            ~-4.8 |           ~0.54 |
| Gradient Boosting  |            ~-5.8 |           ~0.45 |

Bajo un escenario estable, Random Forest y Gradient Boosting sí superan por
amplio margen a la regresión lineal, tal como se espera en la literatura.

### Conclusión

La elección del modelo idóneo depende del caso de uso real:

- Si el objetivo es **predecir dentro del rango histórico** (mismo régimen
  estacional o condiciones similares), Random Forest y Gradient Boosting son
  claramente superiores.
- Si el objetivo es **predecir en condiciones nuevas** (temperaturas o climas
  fuera del rango de entrenamiento), la regresión lineal es más robusta, y en
  un despliegue real convendría **reentrenar los modelos periódicamente** con
  datos recientes que capturen la estacionalidad actual.

Este hallazgo también refuerza la importancia de elegir cuidadosamente la
estrategia de validación en problemas con dependencia temporal.

---

## Reproducibilidad

Todas las operaciones aleatorias usan `random_state=42`, por lo que los
resultados numéricos son reproducibles en cualquier máquina con las mismas
versiones de las librerías.

---

## Equipo

Equipo 7 - Modelatón 2026