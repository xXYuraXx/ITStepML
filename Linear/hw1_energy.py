import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Завантаження даних
df = pd.read_csv("assets/energy_usage_plus.csv")

# 2. Вибір ознак і цільової змінної
X = df[['temperature', 'humidity', 'hour', 'is_weekend', 'season', 'district_type']]
y = df['consumption']

# 3. Попередня обробка: кодування категоріальних ознак
categorical_features = ['season', 'district_type']
numeric_features = ['temperature', 'humidity', 'hour', 'is_weekend']

# 4. Побудова пайплайну
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'  # Пропускає числові ознаки без змін
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 5. Розділення на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Навчання моделі
model.fit(X_train, y_train)

# 7. Прогноз на тестових даних
y_pred = model.predict(X_test)

# 8. Оцінка моделі
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f"MAPE: {mape:.2f}%")

# 9. Прогноз на нових даних
new_district = pd.DataFrame([{
    'temperature': 25.0,
    'humidity': 60,
    'hour': 14,
    'is_weekend': 0,
    'season': 'Summer',
    'district_type': 'Residential'
}])

predicted_consumption = model.predict(new_district)[0]
print(f"\nПрогнозоване споживання електроенергії: {predicted_consumption:,.2f} кВт·год")

# Візуалізація: справжні значення vs прогноз
plt.scatter(y_test, y_pred)
plt.xlabel("Справжнє споживання")
plt.ylabel("Прогнозоване споживання")
plt.title(f"Справжнє vs Прогнозоване споживання (Похибка {mape:.2f}%)")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')
plt.show()

