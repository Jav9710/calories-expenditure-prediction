# Predicción de Gasto Calórico con Machine Learning

Proyecto de desarrollo de un modelo predictor del gasto calórico durante la actividad física, utilizando técnicas de Machine Learning y siguiendo la metodología CRISP-DM.

## Tabla de Contenidos
- [Objetivo del Proyecto](#objetivo-del-proyecto)
- [Metodología CRISP-DM](#metodología-crisp-dm)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Análisis Exploratorio de Datos](#análisis-exploratorio-de-datos)
- [Feature Engineering](#feature-engineering)
- [Modelos Implementados](#modelos-implementados)
- [Resultados](#resultados)
- [Requisitos](#requisitos)
- [Uso](#uso)

---

## Objetivo del Proyecto

### Objetivo Principal
Crear un modelo predictivo que estime con precisión las **calorías quemadas** por una persona durante el ejercicio físico, basándose en características fisiológicas y parámetros de actividad.

### Objetivos Específicos
- Analizar la relación entre variables fisiológicas (edad, peso, altura, frecuencia cardíaca, temperatura corporal) y el gasto calórico
- Identificar las características más relevantes para la predicción del consumo energético
- Comparar diferentes algoritmos de Machine Learning (KNN, Regresión Ridge/Lasso, Random Forest, XGBoost, MLP)
- Optimizar el modelo seleccionado para maximizar su capacidad predictiva
- Proporcionar una herramienta útil para aplicaciones de salud y fitness

### Aplicaciones Potenciales
- **Aplicaciones de fitness**: Estimación precisa de calorías quemadas durante entrenamientos
- **Dispositivos wearables**: Mejora de algoritmos en smartwatches y pulseras de actividad
- **Programas de salud**: Planificación de ejercicio personalizado para control de peso
- **Investigación deportiva**: Análisis del gasto energético en diferentes tipos de actividad física

---

## Metodología CRISP-DM

El proyecto sigue la metodología **CRISP-DM** (Cross-Industry Standard Process for Data Mining):

### 1. Comprensión del Negocio
Definición del problema: predecir el gasto calórico durante el ejercicio físico para aplicaciones de salud y fitness.

### 2. Comprensión de los Datos ([01_data_understanding.ipynb](01_investigacion/01_data_understanding.ipynb))
- **Dataset**: 750,000 registros de entrenamiento + 250,000 de prueba
- **Variables**: 8 features (Age, Height, Weight, Duration, Heart_Rate, Body_Temp, Sex) + 1 target (Calories)
- **Calidad**: Excelente (0% valores nulos, sin duplicados)
- **Balance**: Distribución balanceada por género (50% F / 50% M)

### 3. Preparación de Datos ([02_data_preparation.ipynb](01_investigacion/02_data_preparation.ipynb))
- Limpieza de duplicados
- Validación de rangos fisiológicos
- Codificación One-Hot de variable categórica (Sex)
- División train (80%) / validation (20%)

### 4. Análisis Exploratorio ([03_exploratory_analysis.ipynb](01_investigacion/03_exploratory_analysis.ipynb))
- Análisis de distribuciones y outliers
- Estudio de correlaciones
- Identificación de variables clave

### 5. Feature Engineering ([04_feature_engineering.ipynb](01_investigacion/04_feature_engineering.ipynb))
- Creación de features derivadas
- Normalización con StandardScaler

### 6. Modelado ([02_modelos_baseline/](02_modelos_baseline/))
- Entrenamiento de múltiples algoritmos
- Optimización de hiperparámetros con GridSearch/RandomizedSearch
- Validación y comparación de modelos

---

## Estructura del Proyecto

```
.
├── data/
│   ├── raw/                      # Datos originales (sin procesar)
│   │   ├── train.csv            # Dataset de entrenamiento original
│   │   └── test.csv             # Dataset de prueba original
│   └── processed/                # Datos procesados y enriquecidos
│       ├── train.csv            # Train después de limpieza
│       ├── val.csv              # Validación (20% del train)
│       ├── test.csv             # Test procesado
│       ├── train_fe.csv         # Train con feature engineering
│       ├── val_fe.csv           # Validación con feature engineering
│       ├── test_fe.csv          # Test con feature engineering
│       ├── train_fe_scaled.csv  # Train escalado (listo para ML)
│       └── val_fe_scaled.csv    # Validación escalado
│
├── 01_investigacion/             # Notebooks de análisis CRISP-DM
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_exploratory_analysis.ipynb
│   └── 04_feature_engineering.ipynb
│
├── 02_modelos_baseline/          # Notebooks de modelado
│   ├── knn.ipynb
│   ├── ridge.ipynb
│   ├── lasso.ipynb
│   ├── random_forest.ipynb
│   ├── xgb_regressor.ipynb
│   └── mlp.ipynb
│
├── results/
│   ├── models/                   # Modelos entrenados (.joblib)
│   ├── figures/                  # Visualizaciones generadas
│   └── baseline_results.csv      # Comparación de resultados
│
└── README.md                     # Este archivo
```

---

## Análisis Exploratorio de Datos

### Características del Dataset

#### Variables Numéricas
- **Age**: 20-79 años (media: 41.4 años)
- **Height**: 126-222 cm (media: 174.7 cm)
- **Weight**: 36-132 kg (media: 75.1 kg)
- **Duration**: 1-30 minutos (media: 15.4 min)
- **Heart_Rate**: 67-128 BPM (media: 95.5 BPM)
- **Body_Temp**: 37.1-41.5°C (media: 40.0°C)
- **Calories**: 1-314 kcal (media: 88.3 kcal)

#### Variable Categórica
- **Sex**: male (49.9%) / female (50.1%)

### Hallazgos Clave del EDA

#### 1. Correlaciones con la Variable Objetivo (Calories)
Las variables más correlacionadas con el gasto calórico son:

| Variable    | Correlación | Interpretación |
|-------------|-------------|----------------|
| Duration    | 0.9599      | Correlación MUY FUERTE ⭐ |
| Heart_Rate  | 0.9087      | Correlación MUY FUERTE ⭐ |
| Body_Temp   | 0.8287      | Correlación MUY FUERTE ⭐ |
| Age         | 0.1457      | Correlación DÉBIL |
| Weight      | 0.0159      | Correlación MUY DÉBIL |
| Height      | -0.0040     | Correlación MUY DÉBIL |

**Interpretación**: La duración del ejercicio, la frecuencia cardíaca y la temperatura corporal son los predictores más importantes del gasto calórico.

#### 2. Multicolinealidad Detectada
Se identificaron 4 pares de variables con alta correlación (>0.7):
- Height vs Weight (0.958)
- Duration vs Heart_Rate (0.875)
- Duration vs Body_Temp (0.903)
- Heart_Rate vs Body_Temp (0.796)

**Implicación**: Considerar regularización (Ridge/Lasso) o selección de features para modelos sensibles a multicolinealidad.

#### 3. Distribuciones
- **Calories**: Distribución asimétrica a la derecha (skewness: 0.54)
- Mayor concentración de valores bajos de calorías (mediana: 77 cal)
- Ninguna variable sigue distribución perfectamente normal

#### 4. Outliers
- **Body_Temp**: 1.99% outliers (14,919 registros)
- Resto de variables: <0.02% outliers (aceptable)

---

## Feature Engineering

### Features Derivadas Creadas

#### 1. BMI (Body Mass Index)
```python
BMI = Weight / (Height_m)²
```
- Indicador de composición corporal
- Relevante para metabolismo basal

#### 2. Altura en Metros
```python
Height_m = Height / 100
```
- Conversión de cm a metros para cálculo del BMI

#### 3. Frecuencia Cardíaca Normalizada
```python
HR_per_min = Heart_Rate / (Duration/60)
```
- Representa la intensidad del ejercicio
- Captura el ritmo cardíaco por minuto de actividad

#### 4. Interacción HR × Duration
```python
HRxDuration = Heart_Rate * Duration
```
- Captura el efecto combinado de intensidad y duración
- Feature altamente predictivo del gasto calórico

### Normalización
- **Método**: StandardScaler (media=0, std=1)
- **Features normalizadas**: Age, Height, Weight, Duration, Heart_Rate, Body_Temp, BMI, HR_per_min, HRxDuration
- **Scaler guardado**: `results/models/scaler.joblib`

---

## Modelos Implementados

### Algoritmos Evaluados

1. **K-Nearest Neighbors (KNN)**
   - Algoritmo no paramétrico basado en distancia
   - Hiperparámetros: n_neighbors, weights

2. **Ridge Regression**
   - Regresión lineal con regularización L2
   - Maneja bien la multicolinealidad

3. **Lasso Regression**
   - Regresión lineal con regularización L1
   - Selección automática de features

4. **Random Forest**
   - Ensamble de árboles de decisión
   - Robusto a outliers y no linealidades

5. **XGBoost Regressor**
   - Gradient Boosting optimizado
   - Estado del arte en competencias de ML

6. **Multi-Layer Perceptron (MLP)**
   - Red neuronal feedforward
   - Captura relaciones complejas no lineales

### Proceso de Entrenamiento
- **Validación**: 3-fold Cross-Validation
- **Optimización**: RandomizedSearchCV / GridSearchCV
- **Métrica principal**: RMSE (Root Mean Squared Error)
- **Métricas secundarias**: MAE, R²

---

## Resultados

### Ejemplo: K-Nearest Neighbors (KNN)

**Mejores hiperparámetros encontrados:**
- `n_neighbors`: 7
- `weights`: 'distance'

**Métricas de desempeño:**
- **MAE**: 2.71 calorías
- **RMSE**: 4.28 calorías
- **R²**: 0.9953

**Interpretación:**
- El modelo explica el 99.53% de la varianza en el gasto calórico
- Error promedio de ±4.28 calorías (muy bajo considerando rango 1-314 cal)
- Excelente capacidad predictiva

### Comparación de Modelos
Los resultados completos de todos los modelos se encuentran en:
- [results/baseline_results.csv](results/baseline_results.csv)
- Notebooks individuales en [02_modelos_baseline/](02_modelos_baseline/)

---

## Requisitos

### Dependencias Principales
```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
scipy
joblib
```

Instalar todas las dependencias:
```bash
pip install -r requirements.txt
```

---

## Uso

### 1. Análisis Exploratorio Completo
Ejecutar los notebooks en orden:
```
01_investigacion/
  └── 01_data_understanding.ipynb    # Exploración inicial
  └── 02_data_preparation.ipynb      # Limpieza y preparación
  └── 03_exploratory_analysis.ipynb  # EDA profundo
  └── 04_feature_engineering.ipynb   # Creación de features
```

### 2. Entrenamiento de Modelos
Ejecutar cualquier notebook de la carpeta `02_modelos_baseline/`:
```python
# Ejemplo: KNN
jupyter notebook 02_modelos_baseline/knn.ipynb
```

### 3. Predicción con Modelo Entrenado
```python
import joblib
import pandas as pd

# Cargar modelo y scaler
model = joblib.load('results/models/KNeighbors.joblib')
scaler = joblib.load('results/models/scaler.joblib')

# Preparar datos de entrada
X_new = pd.DataFrame({
    'Age': [30],
    'Height': [175],
    'Weight': [70],
    'Duration': [20],
    'Heart_Rate': [120],
    'Body_Temp': [39.5],
    'Sex_male': [1],  # 1=male, 0=female
    'BMI': [22.86],
    'HR_per_min': [360],
    'HRxDuration': [2400]
})

# Escalar features
X_scaled = scaler.transform(X_new[numeric_feats])

# Predecir
calories_predicted = model.predict(X_scaled)
print(f"Calorías estimadas: {calories_predicted[0]:.2f} kcal")
```

---

## Flujo de Trabajo

```
data/raw/ (datos originales)
    ↓
[01_data_understanding.ipynb] → Análisis inicial
    ↓
[02_data_preparation.ipynb] → Limpieza + División
    ↓
data/processed/ (datos limpios)
    ↓
[03_exploratory_analysis.ipynb] → EDA profundo
    ↓
[04_feature_engineering.ipynb] → Features + Normalización
    ↓
data/processed/*_fe_scaled.csv (listos para ML)
    ↓
[02_modelos_baseline/*.ipynb] → Entrenamiento de modelos
    ↓
results/models/*.joblib (modelos entrenados)
```

---

## Conclusiones

### Fortalezas del Proyecto
- Dataset de alta calidad (750K registros, 0% nulos)
- Metodología rigurosa (CRISP-DM)
- Features altamente correlacionados con el target
- Modelos con excelente capacidad predictiva (R² > 0.99)

### Hallazgos Principales
1. **Duration**, **Heart_Rate** y **Body_Temp** son los predictores más importantes
2. Las interacciones entre variables mejoran significativamente la predicción
3. Los modelos no lineales (KNN, RF, XGBoost) capturan mejor las relaciones complejas
4. El feature engineering con variables derivadas (BMI, HRxDuration) aporta valor predictivo

### Próximos Pasos (Trabajo Futuro)
- Implementar técnicas de ensamble (stacking, blending)
- Explorar redes neuronales más profundas
- Realizar análisis de feature importance detallado
- Validar modelos con datos de diferentes fuentes
- Desplegar modelo en producción (API REST)

---

## Autor

Proyecto desarrollado como Trabajo Fin de Máster (TFM)

## Licencia

Este proyecto es de uso académico y educativo.

---

**Última actualización**: 2025-10-31
