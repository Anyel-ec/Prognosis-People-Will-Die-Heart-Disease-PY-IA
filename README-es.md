# Heart Failure Prediction Project

## Descripción del Proyecto

Este proyecto se enfoca en el pronóstico de la mortalidad en pacientes con enfermedades cardíacas. Utiliza un conjunto de datos que contiene varias características clínicas de los pacientes para entrenar y evaluar modelos de clasificación, con el objetivo de predecir si un paciente fallecerá debido a complicaciones cardíacas.

## Conjunto de Datos

El conjunto de datos utilizado en este proyecto se llama `heart_failure.csv` y contiene los siguientes campos:

- `age`: Edad del paciente.
- `anaemia`: Indicador de anemia (0: No, 1: Sí).
- `creatinine_phosphokinase`: Nivel de la enzima creatina fosfoquinasa en sangre.
- `diabetes`: Indicador de diabetes (0: No, 1: Sí).
- `ejection_fraction`: Porcentaje de sangre que sale del corazón en cada contracción.
- `high_blood_pressure`: Indicador de hipertensión (0: No, 1: Sí).
- `platelets`: Número de plaquetas en la sangre.
- `serum_creatinine`: Nivel de creatinina en sangre.
- `serum_sodium`: Nivel de sodio en sangre.
- `sex`: Sexo del paciente (0: Mujer, 1: Hombre).
- `smoking`: Indicador de tabaquismo (0: No, 1: Sí).
- `time`: Periodo de seguimiento (días).
- `DEATH_EVENT`: Evento de muerte (0: No, 1: Sí).

## Instalación

Para ejecutar el código, asegúrate de tener instaladas las siguientes bibliotecas de Python:

```bash
pip install pandas scikit-learn
```

## Uso

### Árbol de Decisión

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Carga el conjunto de datos en un DataFrame de Pandas
df = pd.read_csv('heart_failure.csv')

# Asegúrate de que no haya valores nulos en el DataFrame
df = df.dropna()

# Divide los datos en características (X) y etiquetas (y)
X = df.drop('DEATH_EVENT', axis=1)  # características
y = df['DEATH_EVENT']  # etiquetas

# Divide los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Entrena el clasificador de árbol de decisiones
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Realiza predicciones sobre los datos de prueba
predictions = clf.predict(X_test)

# Imprime el informe de clasificación
print(classification_report(y_test, predictions))
```

### SVM (Support Vector Machine)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, precision_score

# Carga el conjunto de datos en un DataFrame de Pandas
df = pd.read_csv('heart_failure.csv')

# Asegúrate de que no haya valores nulos en el DataFrame
df = df.dropna()

# Divide los datos en características (X) y etiquetas (y)
X = df.drop('DEATH_EVENT', axis=1)  # características
y = df['DEATH_EVENT']  # etiquetas

# Divide los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Entrena el clasificador SVM
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Realiza predicciones sobre los datos de prueba
predictions = clf.predict(X_test)

# Calcula la precisión
precision = precision_score(y_test, predictions, pos_label=1)

# Imprime el informe de clasificación y la precisión
print("Precisión:", precision)
print(classification_report(y_test, predictions))
```

## Resultados

### Árbol de Decisión

```plaintext
              precision    recall  f1-score   support

           0       0.67      0.83      0.74        35
           1       0.65      0.44      0.52        25

    accuracy                           0.67        60
   macro avg       0.66      0.63      0.63        60
weighted avg       0.66      0.67      0.65        60
```

### SVM

```plaintext
Precisión: 0.8125
              precision    recall  f1-score   support

           0       0.73      0.91      0.81        35
           1       0.81      0.52      0.63        25

    accuracy                           0.75        60
   macro avg       0.77      0.72      0.72        60
weighted avg       0.76      0.75      0.74        60
```

### Interpretación de los Resultados

#### Árbol de Decisión

- **Precisión (precision)**: Mide la proporción de verdaderos positivos entre todos los casos positivos predichos. Para la clase 1 (muerte), la precisión es 0.65, lo que significa que el 65% de las predicciones positivas son correctas.
- **Recall (recall)**: Mide la proporción de verdaderos positivos entre todos los casos reales positivos. Para la clase 1, el recall es 0.44, indicando que el modelo identifica correctamente el 44% de los casos reales de muerte.
- **F1-score**: Es la media armónica de la precisión y el recall. Un valor más alto indica un mejor equilibrio entre precisión y recall.
- **Accuracy (exactitud)**: Proporción de predicciones correctas sobre el total de predicciones. En este caso, el modelo tiene una exactitud del 67%.

#### SVM

- **Precisión (precision)**: La precisión para la clase 1 es 0.81, lo que significa que el 81% de las predicciones positivas son correctas.
- **Recall (recall)**: El recall para la clase 1 es 0.52, indicando que el modelo identifica correctamente el 52% de los casos reales de muerte.
- **F1-score**: La F1-score es mejor para la clase 0 que para la clase 1, lo que sugiere que el modelo es más efectivo en predecir correctamente los casos de supervivencia.
- **Accuracy (exactitud)**: La exactitud del modelo es 75%, lo que indica una mejora con respecto al modelo de árbol de decisión.

En general, aunque ambos modelos tienen sus ventajas, el modelo SVM muestra una mayor precisión en la predicción de eventos de muerte, aunque con un menor recall comparado con la precisión. Esto sugiere que el SVM es mejor en evitar falsos positivos, pero puede perder algunos casos positivos reales.
