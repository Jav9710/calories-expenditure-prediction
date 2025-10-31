# Estructura de Datos

Este directorio contiene los datasets utilizados en el proyecto de predicción de gasto calórico.

## 🎯 Finalidad de la Investigación

Este proyecto tiene como objetivo desarrollar un **modelo predictor del gasto calórico** durante la actividad física, utilizando técnicas de Machine Learning y siguiendo la metodología CRISP-DM.

### Objetivos del Proyecto

**Objetivo Principal:**
Crear un modelo predictivo que estime con precisión las calorías quemadas por una persona durante el ejercicio físico, basándose en características fisiológicas y parámetros de actividad.

**Objetivos Específicos:**
- Analizar la relación entre variables fisiológicas (edad, peso, altura, frecuencia cardíaca) y el gasto calórico
- Identificar las características más relevantes para la predicción del consumo energético
- Comparar diferentes algoritmos de Machine Learning (KNN, regresión, árboles de decisión, etc.)
- Optimizar el modelo seleccionado para maximizar su capacidad predictiva
- Proporcionar una herramienta útil para aplicaciones de salud y fitness

### Aplicaciones Potenciales

Este modelo puede ser utilizado en:
- **Aplicaciones de fitness**: Estimación precisa de calorías quemadas durante entrenamientos
- **Dispositivos wearables**: Mejora de algoritmos en smartwatches y pulseras de actividad
- **Programas de salud**: Planificación de ejercicio personalizado para control de peso
- **Investigación deportiva**: Análisis del gasto energético en diferentes tipos de actividad física

### Metodología

El proyecto sigue la metodología **CRISP-DM** (Cross-Industry Standard Process for Data Mining):
1. **Comprensión del negocio**: Definición del problema y objetivos
2. **Comprensión de los datos**: Análisis exploratorio del dataset
3. **Preparación de datos**: Limpieza, transformación y feature engineering
4. **Modelado**: Desarrollo y entrenamiento de modelos predictivos
5. **Evaluación**: Validación y comparación de modelos
6. **Despliegue**: Implementación del modelo final

## 📁 Estructura de Carpetas

### `raw/`
Contiene el dataset original a analizar. Los datos en esta carpeta deben permanecer inalterados para mantener la integridad de los datos fuente.

**Propósito:**
- Almacenar los datos sin procesar tal como se obtienen de la fuente original
- Servir como punto de referencia para cualquier análisis o procesamiento posterior
- Mantener la trazabilidad del origen de los datos

### `processed/`
Contiene los datasets procesados y enriquecidos generados durante las diferentes fases del proyecto.

**Propósito:**
- Almacenar datasets transformados después de la limpieza de datos
- Guardar datasets con variables derivadas o características nuevas (feature engineering)
- Mantener versiones intermedias de los datos para diferentes etapas del análisis
- Almacenar los datasets finales listos para el modelado

## 🔄 Flujo de Trabajo

```
raw/ (datos originales)
  ↓
  [Procesamiento y limpieza]
  ↓
processed/ (datos procesados)
  ↓
  [Feature engineering]
  ↓
processed/ (datos enriquecidos)
  ↓
  [Modelado]
```

## 📝 Convenciones de Nomenclatura

Se recomienda usar nombres descriptivos para los archivos en `processed/` que indiquen el tipo de procesamiento aplicado, por ejemplo:
- `dataset_cleaned.csv` - Datos después de limpieza
- `dataset_featured.csv` - Datos con features adicionales
- `dataset_final.csv` - Datos listos para modelado
