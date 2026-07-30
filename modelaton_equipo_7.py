# ============================================
# importación de librerías
# ============================================
import pandas as pd                                             # manejo de dataframes y CSVs
import numpy as np                                              # cálculos numéricos (raíz cuadrada, arrays, etc.)
import matplotlib.pyplot as plt                                 # librería base de gráficas
import seaborn as sns                                           # gráficas estadísticas más elegantes (heatmaps)
from sklearn.linear_model import LinearRegression               # modelo 1: regresión lineal
from sklearn.preprocessing import StandardScaler                # estandariza variables (media 0, desv. 1)
from sklearn.ensemble import GradientBoostingRegressor          # modelo 3: gradient boosting
from sklearn.ensemble import RandomForestRegressor              # modelo 2: random forest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # métricas de evaluación
from sklearn.model_selection import train_test_split            # split aleatorio para análisis comparativo

# ============================================
# carga del dataset
# ============================================
url = (                                                         # URL oficial del repositorio UCI Machine Learning
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/00374/"
    "energydata_complete.csv"
)
df = pd.read_csv(url)                                           # lee el CSV directamente desde la web

# ============================================
# exploración inicial del dataset
# ============================================
print(df.head())                                                # muestra las primeras 5 filas (rv1 y rv2 son ruido aleatorio)
print(df.shape)                                                 # dimensiones: (filas, columnas)
print(df.isnull().sum())                                        # cuenta valores faltantes por columna (deberían ser 0)
print(df.describe())                                            # estadísticas descriptivas: mean, std, min, max, cuartiles
print(df.dtypes)                                                # tipo de dato de cada columna

# ============================================
# transformación de la columna date
# ============================================
df['date'] = pd.to_datetime(df['date'])                         # convierte texto a formato datetime nativo
df['fecha'] = df['date'].dt.date                                # extrae solo la fecha (año-mes-día)
df['hora'] = df['date'].dt.time                                 # extrae solo la hora del día
print(df.head())                                                # verificamos que se agregaron 'fecha' y 'hora'


# ============================================
# separación cronológica del dataset (80% train / 20% test)
# ============================================
df = df.sort_values('date').reset_index(drop=True)              # ordenamos cronológicamente y reiniciamos índices

n = len(df)                                                     # total de registros en el dataset
corte = int(n * 0.8)                                            # posición del corte para separar el 80%

train = df.iloc[:corte].copy()                                  # primer 80% -> entrenamiento (pasado)
test = df.iloc[corte:].copy()                                   # último 20% -> prueba (simula el futuro)

# imprimimos rangos temporales para verificar que el split se hizo correctamente
print("Train:", train.shape, "de", train['date'].min(), "a", train['date'].max())
print("Test:", test.shape, "de", test['date'].min(), "a", test['date'].max())

# ============================================
# selección de variables
# ============================================

# derivamos variables temporales útiles como predictoras
for d in (train, test):                                         # aplicamos lo mismo a train y test
    d['hora_num'] = d['date'].dt.hour                           # hora del día (0-23)
    d['dia_semana'] = d['date'].dt.dayofweek                    # día de la semana (0=lunes ... 6=domingo)
    d['es_fin_semana'] = d['dia_semana'].isin([5, 6]).astype(int)  # 1 si es sábado/domingo, 0 en otro caso

# definimos qué queremos predecir
target = 'Appliances'                                           # variable objetivo: consumo energético en Wh

# excluimos columnas que NO son predictoras válidas:
# - date, fecha, hora: ya descompuestas en hora_num/dia_semana/es_fin_semana
# - rv1, rv2: variables aleatorias, sin valor predictivo real
# - Appliances: es el target, no puede predecirse a sí mismo
excluir = ['date', 'fecha', 'hora', 'rv1', 'rv2', target]

# lista final de features (variables predictoras)
features = [col for col in train.columns if col not in excluir]
print("Variables seleccionadas como features:", features)

# calculamos la correlación lineal de cada feature con el target
corr_target = train[features + [target]].corr()[target].sort_values(ascending=False)
print("\nCorrelación de cada variable con Appliances:\n", corr_target)

# separamos matriz de features (X) y vector objetivo (y) para ambos conjuntos
X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

print("\nX_train:", X_train.shape, "| X_test:", X_test.shape)

# ============================================
# heatmap de correlación
# ============================================

# heatmap 1: matriz completa de correlaciones entre todas las features y el target
# sirve para detectar multicolinealidad (features muy correlacionadas entre sí)
plt.figure(figsize=(14, 10))                                    # tamaño de la figura
matriz_corr = train[features + [target]].corr()                 # matriz de correlaciones de Pearson

sns.heatmap(
    matriz_corr,
    annot=False,                                                # sin números encima (satura con tantas variables)
    cmap='coolwarm',                                            # paleta: azul=negativo, rojo=positivo
    center=0,                                                   # 0 queda en color neutro (blanco)
    linewidths=0.3                                              # líneas finas entre celdas
)
plt.title('Matriz de correlación - variables vs Appliances')
plt.tight_layout()                                              # evita que se corten etiquetas
plt.show()

# heatmap 2: solo la columna de correlación con el target (más fácil de leer)
plt.figure(figsize=(6, 10))
sns.heatmap(
    corr_target.to_frame(),                                     # convertimos la Serie en DataFrame de 1 columna
    annot=True,                                                 # ahora sí mostramos los números
    fmt='.2f',                                                  # formato con 2 decimales
    cmap='coolwarm',
    center=0
)
plt.title('Correlación de cada variable con Appliances')
plt.tight_layout()
plt.show()

# ============================================
# modelo 1: regresión lineal
# ============================================

# creamos un escalador que transforma cada variable a media 0 y desv. estándar 1
# necesario para que los coeficientes de la regresión sean comparables entre sí
scaler = StandardScaler()

# el scaler se ENTRENA solo con train (aprende sus medias y desviaciones)
# a test se le APLICA la misma transformación, sin recalcular
# esto evita que información futura "contamine" el escalado
X_train_scaled = scaler.fit_transform(X_train)                  # fit + transform en train
X_test_scaled = scaler.transform(X_test)                        # solo transform en test

# el scaler devuelve arrays de numpy; los volvemos a DataFrame para conservar nombres de columnas
X_train_scaled = pd.DataFrame(X_train_scaled, columns=features, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=features, index=X_test.index)

# creamos y entrenamos el modelo de regresión lineal
modelo_lr = LinearRegression()
modelo_lr.fit(X_train_scaled, y_train)                          # ajusta la ecuación de la recta

# generamos predicciones sobre el conjunto de prueba
y_pred_lr = modelo_lr.predict(X_test_scaled)

# extraemos los coeficientes: cuánto pesa cada variable en la predicción
# ordenamos por magnitud absoluta (un coef. muy negativo es tan importante como uno muy positivo)
coeficientes = pd.Series(modelo_lr.coef_, index=features).sort_values(key=abs, ascending=False)
print("Coeficientes de la regresión lineal (ordenados por magnitud):\n", coeficientes)

# ============================================
# modelo 2: Random Forest
# ============================================

modelo_rf = RandomForestRegressor(
    n_estimators=100,                                           # número de árboles en el bosque
    random_state=42,                                            # semilla fija para reproducibilidad
    n_jobs=-1                                                   # usa todos los núcleos del CPU (más rápido)
)

modelo_rf.fit(X_train, y_train)                                 # entrena con datos SIN escalar (RF no lo necesita)
y_pred_rf = modelo_rf.predict(X_test)                           # predicciones sobre test

# importancia de variables: qué tanto reduce el error cada feature en promedio
importancia_rf = pd.Series(modelo_rf.feature_importances_, index=features).sort_values(ascending=False)
print("Importancia de variables - Random Forest:\n", importancia_rf)


# ============================================
# modelo 3: Gradient Boosting
# ============================================

modelo_gb = GradientBoostingRegressor(
    n_estimators=100,                                           # número de árboles secuenciales
    learning_rate=0.1,                                          # tasa de aprendizaje (cuánto corrige cada árbol al anterior)
    max_depth=3,                                                # profundidad máxima de cada árbol (evita sobreajuste)
    random_state=42                                             # semilla fija para reproducibilidad
)

modelo_gb.fit(X_train, y_train)                                 # entrena secuencialmente (más lento que RF)
y_pred_gb = modelo_gb.predict(X_test)

# importancia de variables (mismo criterio que RF)
importancia_gb = pd.Series(modelo_gb.feature_importances_, index=features).sort_values(ascending=False)
print("Importancia de variables - Gradient Boosting:\n", importancia_gb)

# verificación: los 4 arrays deben tener el mismo tamaño (el 20% de test)
print("Regresión lineal:", y_pred_lr.shape)
print("Random Forest:", y_pred_rf.shape)
print("Gradient Boosting:", y_pred_gb.shape)
print("y_test real:", y_test.shape)

# ============================================
# evaluación de resultados (split cronológico)
# ============================================

def evaluar_modelo(nombre, y_real, y_pred):
    """
    Calcula las 3 métricas de regresión estándar e imprime un resumen.
    - MAE: error absoluto medio (mismas unidades que el target, en Wh).
    - RMSE: raíz del error cuadrático medio (penaliza más los errores grandes).
    - R²: proporción de varianza explicada (1=perfecto, 0=predice la media, <0=peor que la media).
    """
    mae = mean_absolute_error(y_real, y_pred)                   # promedio de |real - predicho|
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))          # raíz del promedio de (real - predicho)²
    r2 = r2_score(y_real, y_pred)                               # 1 - (SS_res / SS_tot)
    print(f"{nombre}")
    print(f"  MAE:  {mae:.2f} Wh")
    print(f"  RMSE: {rmse:.2f} Wh")
    print(f"  R²:   {r2:.4f}")
    print()
    return {"modelo": nombre, "MAE": mae, "RMSE": rmse, "R2": r2}

# evaluamos los 3 modelos y guardamos los resultados en una lista de diccionarios
resultados = []
resultados.append(evaluar_modelo("Regresión lineal", y_test, y_pred_lr))
resultados.append(evaluar_modelo("Random Forest", y_test, y_pred_rf))
resultados.append(evaluar_modelo("Gradient Boosting", y_test, y_pred_gb))

# convertimos la lista en un DataFrame para presentación tabular
comparacion = pd.DataFrame(resultados)
print(comparacion)

# ============================================
# diagnóstico: por qué RF y GB fallan en el split cronológico
# ============================================
# imprimimos distribuciones de real y predicho para confirmar el fenómeno
# de sobreestimación sistemática de RF y GB por incapacidad de extrapolar

print("y_test real:")
print(y_test.describe())

print("\nPredicciones Random Forest:")
print(pd.Series(y_pred_rf).describe())

print("\nPredicciones Gradient Boosting:")
print(pd.Series(y_pred_gb).describe())

# confirmamos que las medias del TARGET son similares entre train y test:
# el problema no está en el target, está en las FEATURES (estacionalidad)
print("Appliances en TRAIN:")
print(y_train.describe())

print("\nAppliances en TEST:")
print(y_test.describe())

# ============================================
# análisis comparativo: split aleatorio
# ============================================
# Objetivo: demostrar que Random Forest y Gradient Boosting SÍ superan
# a la regresión lineal cuando el split no está sesgado por estacionalidad.
# Esto refuerza el hallazgo de que el mal desempeño de los modelos de árboles
# en el split cronológico se debe a su incapacidad de extrapolar fuera
# del rango de entrenamiento.

# preparamos el dataframe completo con las variables temporales derivadas
X_all = df.copy()
X_all['hora_num'] = X_all['date'].dt.hour
X_all['dia_semana'] = X_all['date'].dt.dayofweek
X_all['es_fin_semana'] = X_all['dia_semana'].isin([5, 6]).astype(int)

# extraemos las mismas features y target
X_full = X_all[features]
y_full = X_all[target]

# split aleatorio 80/20 (con semilla fija para reproducibilidad)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)

print("\n" + "="*50)
print("COMPARACIÓN: Split aleatorio")
print("="*50)

# --- regresión lineal con escalado (mismo enfoque que en cronológico) ---
scaler_r = StandardScaler()
X_train_r_scaled = scaler_r.fit_transform(X_train_r)            # ajustamos scaler al train aleatorio
X_test_r_scaled = scaler_r.transform(X_test_r)                  # aplicamos al test aleatorio

modelo_lr_r = LinearRegression()
modelo_lr_r.fit(X_train_r_scaled, y_train_r)
y_pred_lr_r = modelo_lr_r.predict(X_test_r_scaled)

# --- Random Forest (mismos hiperparámetros que en cronológico) ---
modelo_rf_r = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
modelo_rf_r.fit(X_train_r, y_train_r)
y_pred_rf_r = modelo_rf_r.predict(X_test_r)

# --- Gradient Boosting (mismos hiperparámetros que en cronológico) ---
modelo_gb_r = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
)
modelo_gb_r.fit(X_train_r, y_train_r)
y_pred_gb_r = modelo_gb_r.predict(X_test_r)

# evaluamos los 3 modelos bajo split aleatorio
resultados_r = []
resultados_r.append(evaluar_modelo("Regresión lineal (aleatorio)", y_test_r, y_pred_lr_r))
resultados_r.append(evaluar_modelo("Random Forest (aleatorio)", y_test_r, y_pred_rf_r))
resultados_r.append(evaluar_modelo("Gradient Boosting (aleatorio)", y_test_r, y_pred_gb_r))

comparacion_aleatorio = pd.DataFrame(resultados_r)
print(comparacion_aleatorio)

# ============================================
# tabla comparativa final: cronológico vs aleatorio
# ============================================
# unimos ambas tablas etiquetadas para poder comparar lado a lado
comparacion['split'] = 'cronológico'                            # etiquetamos qué tipo de split usó cada fila
comparacion_aleatorio['split'] = 'aleatorio'
tabla_final = pd.concat([comparacion, comparacion_aleatorio], ignore_index=True)  # unimos verticalmente
print("\n" + "="*50)
print("TABLA COMPARATIVA FINAL")
print("="*50)
print(tabla_final)

# ============================================
# generar graficas e insights
# ============================================

# ---------------------------------------------
# 0. Consumo de Appliances a lo largo del tiempo con marca del corte train/test
# ---------------------------------------------
# Muestra el patrón temporal del consumo y visualiza dónde está el corte cronológico.
# Sirve como contexto general antes de mostrar los resultados de los modelos.
plt.figure(figsize=(15, 5))
plt.plot(df['date'], df['Appliances'], alpha=0.5)               # serie completa con transparencia
plt.axvline(df['date'].iloc[corte], color='red', linestyle='--', label='Corte train/test')  # línea del corte
plt.legend()
plt.title('Consumo de Appliances a lo largo del tiempo')
plt.xlabel('Fecha')
plt.ylabel('Wh')
plt.show()


# ---------------------------------------------
# 1. Predicho vs real: los 3 modelos en ambos splits
# ---------------------------------------------
# Rejilla 2x3: fila superior = cronológico, fila inferior = aleatorio.
# Muestra visualmente por qué RF y GB fallan con split cronológico
# (nubes de puntos flotando lejos de la línea roja) y funcionan bien
# con split aleatorio (nubes abrazando la línea).

fig, axes = plt.subplots(2, 3, figsize=(18, 10))                # 2 filas x 3 columnas

# fila superior: resultados del split cronológico
for ax, y_pred, nombre in zip(
    axes[0],                                                    # eje de cada columna en la fila 0
    [y_pred_lr, y_pred_rf, y_pred_gb],                          # predicciones de cada modelo
    ['Regresión lineal', 'Random Forest', 'Gradient Boosting']  # nombres para el título
):
    ax.scatter(y_test, y_pred, alpha=0.3, s=10)                 # dispersión real vs predicho
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],  # línea diagonal ideal
            'r--', label='Predicción perfecta')
    ax.set_xlabel('Consumo real (Wh)')
    ax.set_ylabel('Consumo predicho (Wh)')
    ax.set_title(f'{nombre} - split cronológico')
    ax.legend()

# fila inferior: resultados del split aleatorio
for ax, y_pred, nombre in zip(
    axes[1],
    [y_pred_lr_r, y_pred_rf_r, y_pred_gb_r],
    ['Regresión lineal', 'Random Forest', 'Gradient Boosting']
):
    ax.scatter(y_test_r, y_pred, alpha=0.3, s=10)
    ax.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()],
            'r--', label='Predicción perfecta')
    ax.set_xlabel('Consumo real (Wh)')
    ax.set_ylabel('Consumo predicho (Wh)')
    ax.set_title(f'{nombre} - split aleatorio')
    ax.legend()

plt.tight_layout()
plt.show()

# ---------------------------------------------
# 2. Consumo real vs predicho a lo largo del tiempo (split cronológico)
# ---------------------------------------------
# Muestra cómo cada modelo sigue (o no) el patrón temporal real del consumo.
# Aquí se aprecia claramente la sobreestimación sistemática de RF y GB.

fechas_test = test['date'].values                               # eje X: fechas del conjunto de prueba

plt.figure(figsize=(15, 6))
plt.plot(fechas_test, y_test.values, label='Consumo real', color='black', alpha=0.6, linewidth=1)
plt.plot(fechas_test, y_pred_lr, label='Regresión lineal', alpha=0.7)
plt.plot(fechas_test, y_pred_rf, label='Random Forest', alpha=0.7)
plt.plot(fechas_test, y_pred_gb, label='Gradient Boosting', alpha=0.7)
plt.title('Consumo real vs predicho a lo largo del tiempo (test set cronológico)')
plt.xlabel('Fecha')
plt.ylabel('Appliances (Wh)')
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------
# 3. Top 10 variables más importantes (Random Forest y Gradient Boosting)
# ---------------------------------------------
# Usamos las importancias del split ALEATORIO porque ahí los modelos SÍ funcionan bien,
# entonces las importancias son más confiables como insight de "qué mueve el consumo".

importancia_rf_r = pd.Series(modelo_rf_r.feature_importances_, index=features).sort_values(ascending=False)
importancia_gb_r = pd.Series(modelo_gb_r.feature_importances_, index=features).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))                 # 1 fila x 2 columnas

# Random Forest a la izquierda
importancia_rf_r.head(10).plot(kind='barh', ax=axes[0], color='steelblue')  # gráfica de barras horizontales
axes[0].set_title('Top 10 variables - Random Forest')
axes[0].set_xlabel('Importancia relativa')
axes[0].invert_yaxis()                                          # invertimos para que la más importante quede arriba

# Gradient Boosting a la derecha
importancia_gb_r.head(10).plot(kind='barh', ax=axes[1], color='darkorange')
axes[1].set_title('Top 10 variables - Gradient Boosting')
axes[1].set_xlabel('Importancia relativa')
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()

# ---------------------------------------------
# 4. Comparación visual de métricas: cronológico vs aleatorio
# ---------------------------------------------
# Barras agrupadas: cada modelo tiene una barra por tipo de split.
# Refuerza el hallazgo mostrando el contraste directo entre ambos escenarios.

fig, axes = plt.subplots(1, 3, figsize=(15, 5))                 # una gráfica por métrica

for ax, metrica, titulo in zip(axes, ['MAE', 'RMSE', 'R2'], ['MAE (Wh)', 'RMSE (Wh)', 'R²']):
    ancho = 0.35                                                # ancho de cada barra
    x_pos = range(len(comparacion))                             # posiciones base en el eje X (uno por modelo)
    
    # barras del split cronológico, desplazadas ligeramente a la izquierda
    ax.bar([p - ancho/2 for p in x_pos], comparacion[metrica],
           width=ancho, label='Cronológico', color='steelblue')
    # barras del split aleatorio, desplazadas a la derecha
    ax.bar([p + ancho/2 for p in x_pos], comparacion_aleatorio[metrica],
           width=ancho, label='Aleatorio', color='darkorange')
    
    ax.set_xticks(list(x_pos))                                  # etiquetas del eje X
    ax.set_xticklabels(['Reg. lineal', 'Random Forest', 'Gradient Boosting'], rotation=15)
    ax.set_title(titulo)
    ax.legend()
    ax.axhline(0, color='black', linewidth=0.5)                 # línea horizontal en 0 (útil para R² negativo)

plt.tight_layout()
plt.show()

# ---------------------------------------------
# 5. Insights numéricos y recomendaciones de ahorro energético
# ---------------------------------------------

print("\n" + "="*60)
print("INSIGHTS Y RECOMENDACIONES DE AHORRO ENERGÉTICO")
print("="*60)

# imprimimos las 5 variables más influyentes según el Random Forest aleatorio
print("\nTop 5 variables más influyentes en el consumo (según Random Forest):")
print(importancia_rf_r.head(5).to_string())

# y su correlación lineal con el target, para complementar el análisis
print("\nCorrelación de las variables más importantes con Appliances:")
top5_vars = importancia_rf_r.head(5).index.tolist()             # extraemos los nombres de las top 5
print(corr_target.loc[top5_vars].to_string())                   # buscamos sus correlaciones en la serie ya calculada

# bloque de recomendaciones aterrizadas en las variables detectadas por el modelo
print("""
RECOMENDACIONES DE AHORRO ENERGÉTICO
------------------------------------

1. HORARIOS DE MAYOR CONSUMO (variable más influyente):
   La variable 'hora_num' es, por amplio margen, la más importante para ambos
   modelos (Random Forest y Gradient Boosting). Esto confirma que el consumo
   energético tiene una fuerte dependencia del momento del día.
   Recomendación: identificar franjas horarias pico y desplazar cargas
   flexibles (lavadora, secadora, lavavajillas, plancha) a horas valle.
   Automatizar el encendido programado de estos electrodomésticos puede
   generar ahorros significativos sin afectar la comodidad del hogar.

2. USO DE ILUMINACIÓN:
   La variable 'lights' aparece entre las más importantes en Gradient Boosting
   y tiene una correlación lineal positiva con el consumo (r = 0.22). Esto
   indica que el uso de luces es un buen proxy de la actividad general del
   hogar y, por lo tanto, del consumo de electrodomésticos.
   Recomendación: reemplazar iluminación por LED, instalar sensores de
   presencia en zonas de paso y evaluar el uso de reguladores de intensidad
   (dimmers) en áreas comunes.

3. CONDICIONES AMBIENTALES DE ZONAS ESPECÍFICAS:
   Las temperaturas y humedades de la lavandería (T3 y RH_3) aparecen
   consistentemente entre las variables más importantes en ambos modelos,
   junto con la habitación 2 (sala) y otras zonas interiores. Esto sugiere
   que estas áreas concentran electrodomésticos con alto consumo (lavadora,
   secadora, aparatos de sala) cuyo funcionamiento afecta directamente las
   condiciones ambientales locales.
   Recomendación: priorizar la eficiencia energética en electrodomésticos
   de la lavandería y sala (etiqueta A+++ o equivalente), revisar aislamiento
   térmico en estas zonas y evitar el uso simultáneo de aparatos de alto
   consumo en la misma habitación.

4. INFLUENCIA DEL CLIMA EXTERIOR:
   La humedad exterior (RH_out) presenta la correlación negativa más fuerte
   con el consumo (r = -0.15), lo que sugiere que en días más secos hay
   mayor uso de electrodomésticos, posiblemente por mayor actividad general
   o uso de ventiladores/climatización.
   Recomendación: implementar sistemas de ventilación pasiva y aprovechar
   la ventilación natural en días húmedos para reducir la dependencia de
   equipos eléctricos de climatización.

5. HALLAZGO SOBRE EL DÍA DE LA SEMANA:
   Contrario a lo esperado inicialmente, las variables 'dia_semana' y
   'es_fin_semana' NO resultaron relevantes para los modelos (correlaciones
   cercanas a cero y ausencia del top 10 de importancias). Esto sugiere que
   el patrón de consumo del hogar analizado es relativamente estable a lo
   largo de la semana, dominado por rutinas diarias (horas de comida, sueño,
   trabajo) más que por el tipo de día.
   Implicación: las estrategias de ahorro basadas en horarios son más
   efectivas que las basadas en el día de la semana, al menos para este
   hogar en particular.

6. LIMITACIÓN METODOLÓGICA:
   El desempeño de Random Forest y Gradient Boosting en el split cronológico
   expone que estos modelos no extrapolan bien fuera de su rango de
   entrenamiento. Para una implementación real, sería necesario reentrenar
   el modelo periódicamente con datos recientes que capturen la
   estacionalidad actual, o considerar técnicas específicas para series
   temporales (ARIMA, LSTM, etc.).
""")