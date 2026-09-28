# Legacy: MNIST Digit Recognizer

The original version of this project, before it became SketchSolve.

A CNN trained on MNIST that classifies one handwritten digit (0–9), with a Streamlit app where you draw a digit and get a live prediction.

## Run it

From the repo root, using Python 3.12:

```bash
pip install -r legacy/requirements.txt
python legacy/train_mnist.py   # optional: retrains legacy/mnist_cnn_model.keras
streamlit run legacy/app.py
```

## Model

`Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dense(128) → Dense(10, softmax)`, 225,034 parameters. Adam optimizer, sparse categorical crossentropy loss, 5 epochs, seed 42.

## Results (last run of `legacy/train_mnist.py`)

- Training on 54,000 images, validating on 6,000 held out from the training set (`validation_split=0.1`)
- Validation accuracy: 98.43% after epoch 1, 98.98% after epoch 5
- **Test accuracy: 98.78%** on the 10,000-image MNIST test set, which is used only for this final evaluation
- Most common confusion: 4 predicted as 9 (22 of 982 fours)

An earlier version used the test set as its validation data, so its score wasn't a fair test result. The number above comes from the fixed script.

## Known limitation

The app shrinks the whole 280×280 canvas to 28×28. MNIST digits are cropped, fit into a 20×20 box, and centered by center of mass, so small or off-center drawings are often misclassified. SketchSolve fixes this with shared preprocessing in `src/preprocessing.py`.
