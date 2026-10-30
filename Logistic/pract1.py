import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# 1. Завантаження даних
df = pd.read_csv('assets/internship_candidates_final_numeric.csv')

# 2. Вхідні та цільові змінні
X = df[['Experience', 'Grade', 'EnglishLevel', 'Age', 'EntryTestScore']]
y = df['Accepted']

# 3. Розділення
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Побудова моделі
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Передбачення
y_pred = model.predict(X_test)

# 6. Візуалізація
import matplotlib.pyplot as plt 
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

# 7. Передбачення для нових даних
new_data = pd.DataFrame({
    'Experience': [3.5, 2.8],
    'Grade': [10, 8],
    'EnglishLevel': [2, 2],
    'Age': [18, 30],
    'EntryTestScore': [796, 868],
    
})
predictions = model.predict(new_data)
print("Predictions for new data:", predictions)

# 6. Оцінка
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))