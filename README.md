# LSTM Time Series Prediction Example

This repository demonstrates a **simple Long Short-Term Memory (LSTM) model** using Python and TensorFlow/Keras
to predict the next value in a time series.

LSTMs are a type of **recurrent neural network (RNN)** designed to handle sequential data. They can "remember"
patterns over time, making them ideal for tasks like:

* Time series forecasting
* Anomaly detection
* Natural Language Processing (NLP)

In this example, we use a **sine wave** as a sample dataset to illustrate how an LSTM predicts the next value
based on previous observations.

---

## Requirements

* Python 3.8+
* NumPy
* TensorFlow
* Matplotlib (optional for plotting)

Install dependencies using pip:

```bash
pip install numpy tensorflow matplotlib
```

---

## Code Overview

The code is structured in the following steps:

1. **Import Libraries**
2. **Generate Example Data**
3. **Prepare Dataset for LSTM**
4. **Build and Compile LSTM Model**
5. **Train Model**
6. **Predict Next Value**
7. **Optional: Visualize Predictions**

---

### 1. Import Libraries

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
```

### 2. Generate Example Data

```python
x = np.linspace(0, 100, 500)
y = np.sin(x)
```

### 3. Prepare Dataset for LSTM

```python
window_size = 10
X, Y = [], []
for i in range(len(y) - window_size):
    X.append(y[i:i + window_size])
    Y.append(y[i + window_size])
X = np.array(X).reshape((len(X), window_size, 1))
Y = np.array(Y)
```

### 4. Build and Compile LSTM Model

```python
model = Sequential([
    LSTM(50, activation='tanh', input_shape=(window_size, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
```

### 5. Train Model

```python
model.fit(X, Y, epochs=20, verbose=1)
```

### 6. Predict Next Value

```python
test_input = y[-window_size:].reshape((1, window_size, 1))
predicted = model.predict(test_input, verbose=0)
print("Predicted next value:", predicted[0][0])
```

### 7. Optional: Visualize Predictions

```python
predictions = []
for i in range(len(X)):
    pred = model.predict(X[i].reshape(1, window_size, 1), verbose=0)
    predictions.append(pred[0][0])

plt.plot(y[window_size:], label="Actual")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.show()
```

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/MohamedSLITI/lstm.git
```

2. Navigate to the folder and run:

```bash
python lstm_example.py
```
