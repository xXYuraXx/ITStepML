from sklearn import metrics
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

model = keras.Sequential(
    [
        layers.Dense(4, activation="relu", input_shape=(2,)),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.summary()

X = np.random.rand(1000, 2).astype("float32")

y = (X[:, 0] + X[:, 1] >= 1).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16, 
    validation_data=(X_test, y_test)
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"\nAccuracy на тестовій вибірці: {test_acc * 100:.2f}%")
print(f"Loss на тестовій вибірці: {test_loss:.4f}")

new_points = np.random.rand(5, 2).astype("float32")
probabilities = model.predict(new_points)
predicted_classes = (probabilities >= 0.5).astype(int)
true_classes = (new_points[:, 0] + new_points[:, 1] >= 1).astype(int)

print("\n--- Результати для 5 нових точок ---")
for i in range(5):
    pt = new_points[i]
    sum_coords = pt[0] + pt[1]
    print(
        f"Точка {i+1}: ({pt[0]:.2f}, {pt[1]:.2f}) | "
        f"x1 + x2 = {sum_coords:.2f} | "
        f"Ймовірність: {probabilities[i][0]:.3f} | "
        f"Передбачений: {predicted_classes[i][0]} | "
        f"Справжній: {true_classes[i]}"
    )

plt.figure(figsize=(12, 4))

# Графік Loss (функція втрат)
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Зміна (Loss)')
plt.xlabel('Епоха')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Графік Accuracy (точність)
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Зміна (Accuracy)')
plt.xlabel('Епоха')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()