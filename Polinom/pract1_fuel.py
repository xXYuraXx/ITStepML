import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
 
# 2. Генеруємо дані для навчання
data = pd.read_csv("assets/fuel_consumption_vs_speed.csv")
print(data.head())

X = data[["speed_kmh"]]
Y = data["fuel_consumption_l_per_100km"]
 
# 3. Створюємо модель: Поліноміальна регресія ступеня 3
degree = 3
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
 
# 4. Навчаємо модель на всіх даних
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model.fit(X_train, Y_train)

# 6. Передбачення на тестових даних
y_pred = model.predict(X_test)
 
# 7. Побудова графіку
plt.figure(figsize=(10, 6))
plt.plot(X, Y, label='Real data', color='blue')
plt.plot(X, model.predict(X), label=f'Predicted data (Poly degree {degree})', color='red', linestyle='--')
plt.xlabel('Speed km/h')
plt.ylabel('Fuel consumption 1l/100km')
plt.title('Real vs Predicted data')
plt.legend()
plt.grid(True)
plt.show()
 
# 8. Передбачення для конкретної точки
x_value = 55
predicted_value = model.predict(np.array([[x_value]]))
print(f"Predicted value at x={x_value}: {predicted_value[0]:.2f}")

mae = mean_absolute_error(Y, model.predict(X))
mse = mean_squared_error(Y, model.predict(X))

print(f"Mean Absolute Error (MAE): {mae:.4f}") # MAE — середня абсолютна помилка. Показує середню відстань між передбаченим та реальним значеннями.
print(f"Mean Squared Error (MSE): {mse:.4f}") # Коефіцієнт детермінації. Показує, яку частину варіації цільової змінної пояснює модель.