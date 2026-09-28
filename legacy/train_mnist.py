"""Train the original MNIST digit CNN used by legacy/app.py.

Run from the repo root:  python legacy/train_mnist.py
"""
from pathlib import Path

from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

SEED = 42
MODEL_PATH = Path(__file__).parent / "mnist_cnn_model.keras"

# Seeds Python, NumPy and TensorFlow in one call, so runs are repeatable
# (some TensorFlow ops can still add tiny run-to-run differences).
keras.utils.set_random_seed(SEED)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Scale pixels to [0, 1] and add the channel dimension Conv2D expects.
x_train = (x_train / 255.0).reshape(-1, 28, 28, 1)
x_test = (x_test / 255.0).reshape(-1, 28, 28, 1)

model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
model.summary()

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# validation_split holds out the last 10% of the training data for validation.
# The test set is never seen during training; it's only used once, below.
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

predicted_labels = model.predict(x_test, verbose=0).argmax(axis=1)
print("Confusion matrix (rows = true digit, columns = predicted):")
print(confusion_matrix(y_test, predicted_labels))

model.save(MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")
