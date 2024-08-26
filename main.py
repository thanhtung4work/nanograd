import math
import random

import numpy
from sklearn import datasets

from engine import Value, MLP, loss

digit_dataset = datasets.load_digits()

x = digit_dataset["data"]
y = digit_dataset["target"]
n = MLP(64, [2, 10])

for i in range(10):
    # Forward pass
    y_pred = [n(row) for row in x]
    print(f"[=== epoch {i + 1} ===]")
    l = 0
    for rowp, rowth in zip(y_pred, y):
        l += loss.cross_entropy(rowp, rowth, n_classes=10)
    
    # Reset gradient
    for p in n.parameters():
        p.gradient = 0

    # Back propagation
    l.backward()
    for p in n.parameters():
        p.data += -0.05 * p.gradient

    # Get loss
    print("Loss:", l.data)

# Test trained data
y_pred = [n(row) for row in x]
for y in y_pred:
    print(numpy.argmax(y))