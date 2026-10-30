import matplotlib.pyplot as plt
import numpy as np

# ==========================================

x = np.linspace(-10, 10, 500)
y = x**2 * np.sin(x)

plt.figure()
plt.plot(x, y)
plt.title("Завдання 1: f(x) = x^2 * sin(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()

# ==========================================

data = np.random.normal(loc=5, scale=2, size=1000)

plt.figure()
plt.hist(data, bins=30, edgecolor="black")
plt.title("Завдання 2: Гістограма")
plt.xlabel("Значення")
plt.ylabel("Частота")
plt.show()

# ==========================================

hobbies = ["Програмування", "Серіали", "Спорт", "Відеоігри", "Музика"]
shares = [35, 20, 15, 15, 15]

plt.figure()
plt.pie(shares, labels=hobbies, autopct="%1.1f%%")
plt.title("Завдання 3: Мої хобі")
plt.show()

# ==========================================

apples = np.random.normal(loc=170, scale=20, size=100)
bananas = np.random.normal(loc=130, scale=15, size=100)
oranges = np.random.normal(loc=200, scale=25, size=100)
lemons = np.random.normal(loc=100, scale=10, size=100)

plt.figure()
plt.boxplot([apples, bananas, oranges, lemons], tick_labels=["Яблука", "Банани", "Апельсини", "Лимони"])
plt.title("Завдання 4: Розподіл маси фруктів (г)")
plt.ylabel("Маса (г)")
plt.show()

# ==========================================

x_points = np.random.uniform(0, 1, 100)
y_points = np.random.uniform(0, 1, 100)

plt.figure()
plt.scatter(x_points, y_points, color="green", alpha=0.6)
plt.title("Завдання 5: Точкова діаграма")
plt.xlabel("Вісь X")
plt.ylabel("Вісь Y")
plt.show()

