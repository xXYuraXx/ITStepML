import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Завантаження даних
df = pd.read_csv("assets/cars.csv")
print(df.head())

# Вибір ознак та цільової змінної
X = df[['year', 'engine_volume', 'mileage', 'horsepower']]  # features
y = df['price']                                             # target

print("Features: ", X)
print("Target:", y)

# Розділення на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 20%

# Створення та навчання моделі
model = LinearRegression()
model.fit(X_train, y_train) # start supervised learning

# Прогноз та оцінка
your_car = pd.DataFrame([{
    'year': 2006,
    'engine_volume': 2.6,
    'mileage': 37,
    'horsepower': 325
}])

# Прогноз ціни
predicted_price = model.predict(your_car)
print(f"Прогнозована ціна машини: {predicted_price[0]:,.2f} $")

y_pred = model.predict(X_test)

# Evaluate the model
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f"MAPE: {mape:.2f}%")

# Візуалізація: справжні ціни vs прогноз
plt.scatter(y_test, y_pred)
plt.xlabel("Справжня ціна")
plt.ylabel("Прогнозована ціна")
plt.title(f"Справжня vs Прогнозована ціна (Похибка {mape:.2f}%)")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')
plt.show()