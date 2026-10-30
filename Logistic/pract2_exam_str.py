import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# 1. Завантаження даних
df = pd.read_csv('assets/internship_candidates_cefr_final.csv')

# 2. Вхідні та цільові змінні
X = df[['Experience', 'Grade', 'EnglishLevel', 'Age', 'EntryTestScore']]
y = df['Accepted']

# 3. Попередня обробка: кодування категоріальних ознак
categorical_features = ['EnglishLevel']
numeric_features = ['Experience', 'Grade', 'Age', 'EntryTestScore']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# 4. Побудова пайплайну
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# 5. Розділення
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Навчання моделі
model.fit(X_train, y_train)

# 7. Передбачення
y_pred = model.predict(X_test)

# 8. Візуалізація
plt.scatter(X_test['EntryTestScore'], X_test['EnglishLevel'], c=y_pred, cmap='coolwarm', edgecolor='k', s=100)
plt.title('Logistic Regression Predictions')
plt.xlabel('Entry Test Score')
plt.ylabel('EnglishLevel')
plt.colorbar(label='Predicted Class')
plt.show()

plt.scatter(X_test['EntryTestScore'], X_test['Grade'], c=y_pred, cmap='coolwarm', edgecolor='k', s=100)
plt.title('Logistic Regression Predictions')
plt.xlabel('EntryTestScore')
plt.ylabel('Grade')
plt.colorbar(label='Predicted Class')
plt.show()

# 9. Передбачення для нових даних
new_data = pd.DataFrame({
    'Experience': [3.5, 2.8],
    'Grade': [10, 8],
    'EnglishLevel': ['Pre-Intermediate', 'Upper-Intermediate'],
    'Age': [18, 30],
    'EntryTestScore': [796, 868],
})
predictions = model.predict(new_data)
print("Predictions for new data:", predictions)

# 10. Оцінка
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
