# **Select Language:** 🌍
- [Español (Spanish)](README-es.md)
- [English](README.md)

# Heart Failure Prediction Project

## Project Description

This project focuses on predicting mortality in patients with heart disease. It utilizes a dataset containing various clinical features of patients to train and evaluate classification models, aiming to predict whether a patient will die due to heart-related complications.

## Dataset

The dataset used in this project is called `heart_failure.csv` and contains the following fields:

- `age`: Age of the patient.
- `anaemia`: Indicator of anaemia (0: No, 1: Yes).
- `creatinine_phosphokinase`: Level of the enzyme creatine phosphokinase in blood.
- `diabetes`: Indicator of diabetes (0: No, 1: Yes).
- `ejection_fraction`: Percentage of blood leaving the heart with each contraction.
- `high_blood_pressure`: Indicator of high blood pressure (0: No, 1: Yes).
- `platelets`: Number of platelets in the blood.
- `serum_creatinine`: Level of creatinine in blood.
- `serum_sodium`: Level of sodium in blood.
- `sex`: Sex of the patient (0: Female, 1: Male).
- `smoking`: Indicator of smoking (0: No, 1: Yes).
- `time`: Follow-up period (days).
- `DEATH_EVENT`: Death event (0: No, 1: Yes).

## Installation

To run the code, make sure you have the following Python libraries installed:

```bash
pip install pandas scikit-learn
```

## Usage

### Decision Tree

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Load the dataset into a Pandas DataFrame
df = pd.read_csv('heart_failure.csv')

# Ensure there are no missing values in the DataFrame
df = df.dropna()

# Split the data into features (X) and labels (y)
X = df.drop('DEATH_EVENT', axis=1)  # features
y = df['DEATH_EVENT']  # labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train the decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Make predictions on the test data
predictions = clf.predict(X_test)

# Print the classification report
print(classification_report(y_test, predictions))
```

### SVM (Support Vector Machine)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, precision_score

# Load the dataset into a Pandas DataFrame
df = pd.read_csv('heart_failure.csv')

# Ensure there are no missing values in the DataFrame
df = df.dropna()

# Split the data into features (X) and labels (y)
X = df.drop('DEATH_EVENT', axis=1)  # features
y = df['DEATH_EVENT']  # labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train the SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Make predictions on the test data
predictions = clf.predict(X_test)

# Calculate the precision
precision = precision_score(y_test, predictions, pos_label=1)

# Print the classification report and precision
print("Precision:", precision)
print(classification_report(y_test, predictions))
```

## Results

### Decision Tree

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
Precision: 0.8125
              precision    recall  f1-score   support

           0       0.73      0.91      0.81        35
           1       0.81      0.52      0.63        25

    accuracy                           0.75        60
   macro avg       0.77      0.72      0.72        60
weighted avg       0.76      0.75      0.74        60
```

### Interpretation of Results

#### Decision Tree

- **Precision**: Measures the proportion of true positives among all positive predictions. For class 1 (death), the precision is 0.65, meaning 65% of positive predictions are correct.
- **Recall**: Measures the proportion of true positives among all actual positive cases. For class 1, the recall is 0.44, indicating the model correctly identifies 44% of actual death cases.
- **F1-score**: The harmonic mean of precision and recall. A higher value indicates a better balance between precision and recall.
- **Accuracy**: The proportion of correct predictions out of the total predictions. In this case, the model has an accuracy of 67%.

#### SVM

- **Precision**: The precision for class 1 is 0.81, meaning 81% of positive predictions are correct.
- **Recall**: The recall for class 1 is 0.52, indicating the model correctly identifies 52% of actual death cases.
- **F1-score**: The F1-score is better for class 0 than for class 1, suggesting the model is more effective at correctly predicting survival cases.
- **Accuracy**: The model's accuracy is 75%, showing an improvement over the decision tree model.

Overall, while both models have their advantages, the SVM model shows higher precision in predicting death events, though with lower recall compared to precision. This suggests the SVM is better at avoiding false positives but may miss some true positive cases.
