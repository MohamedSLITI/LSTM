import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Example time series: y = sin(x)
x = np.linspace(0, 100, 500)
y = np.sin(x)

# Prepare dataset for LSTM
window_size = 10
X, Y = [], []
for i in range(len(y) - window_size):
    X.append(y[i:i + window_size])
    Y.append(y[i + window_size])

X = np.array(X)
Y = np.array(Y)

# Reshape for LSTM: [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build LSTM model
model = Sequential([
    LSTM(50, activation='tanh', input_shape=(window_size, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X, Y, epochs=20, verbose=1)

# Predict next value
test_input = y[-window_size:]
test_input = test_input.reshape((1, window_size, 1))
predicted = model.predict(test_input, verbose=0)

print("Predicted next value:", predicted[0][0])
