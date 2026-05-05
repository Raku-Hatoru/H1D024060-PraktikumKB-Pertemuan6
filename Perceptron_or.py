import numpy as np
import Perceptron as p 

# 1. Inisialisasi Input (X1, X2) dan Target (t) secara bipolar
X = np.array([    [1, 1],    [1, -1],    [-1, 1],    [-1, -1]])

t = np.array([[1],[1],[1],[-1]])

# 2. Pemanggilan model Perceptron
model = p.Perceptron(alpha=0.1, epoch=10)

# 3. Training model
model.fit(X, t)
