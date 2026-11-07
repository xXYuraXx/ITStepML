from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers

LABELS = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}


def load_dataset(assets_dir: Path):
    df_train = pd.read_csv(assets_dir / "fashion-mnist_train.csv")
    df_test = pd.read_csv(assets_dir / "fashion-mnist_test.csv")

    X_train = df_train.drop(columns=['label']).values.astype(np.float32) / 255.0
    y_train = df_train['label'].values

    X_test = df_test.drop(columns=['label']).values.astype(np.float32) / 255.0
    y_test = df_test['label'].values

    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train, X_test, y_test, model_path: Path, history_path: Path, epochs: int = 20):
    model = tf.keras.Sequential(
        [
            layers.Dense(128, activation="relu", input_shape=(28*28,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(10),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=64,
        validation_data=(X_test, y_test)
    )

    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    pd.DataFrame(history.history).to_csv(history_path, index=False)
    print(f"History saved to {history_path}")
    
    return model, history


def get_or_train_model(assets_dir: Path, model_path: Path, history_path: Path, epochs: int = 20):
    if model_path.exists():
        print(f"Loading existing model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        history = None
        if history_path.exists():
            history = pd.read_csv(history_path).to_dict(orient="list")
    else:
        print("Training new model...")
        X_train, y_train, X_test, y_test = load_dataset(assets_dir)
        model, raw_history = train_model(X_train, y_train, X_test, y_test, model_path, history_path, epochs=epochs)
        history = raw_history.history
    return model, history

def predict_custom_image(model, image_path: Path):
    if not image_path.exists():
        print(f"Файл {image_path} не знайдено.")
        return

    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))

    # Білий малюнок на чорному фоні
    img_array = np.array(img, dtype=np.float32)

    img_array = img_array / 255.0
    sample = img_array.reshape(1, 28*28)
    
    pred_logits = model.predict(sample)
    pred_class = int(np.argmax(pred_logits, axis=1)[0])

    print(f"Зображення: {image_path.name}")
    print("Logits:", pred_logits)
    print("Predicted class:", pred_class, "=", LABELS[pred_class])


cur_dir = Path(__file__).resolve().parent
assets_path = cur_dir / "assets"
saved_model_path = cur_dir / "fashion_model.keras"
history_path = cur_dir / "fashion_model_history.csv"

model, history = get_or_train_model(assets_path, saved_model_path, history_path, epochs=20)

custom_img = cur_dir / "assets" / "test_img.png"
if custom_img.exists():
    predict_custom_image(model, custom_img)

if history is not None:
    epochs = len(history['accuracy'])
    x = range(1, epochs + 1)
    y_acc = history['accuracy']
    val_acc = history['val_accuracy']
    y_loss = history['loss']
    val_loss = history['val_loss']

    fig = plt.figure(figsize=(9, 4), num="Detect type clothes")

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(x, y_acc, label='Train accuracy')
    ax1.plot(x, val_acc, label='Val accuracy', color="orange")
    ax1.set_title("Accuracy over epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="upper left")

    ax2.plot(x, y_loss, label='Train Loss')
    ax2.plot(x, val_loss, label='Val Loss', color="orange")
    ax2.set_title("Loss over epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()