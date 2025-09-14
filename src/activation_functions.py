# visually seeing the diff between relu and gelu - nothing fancy 

from DummyGPTModel import *
import matplotlib.pyplot as plt
import torch.nn as nn

# initializing GELU with the class initialized in DummyGPTModel
# Relu is a built in pytorch class
gelu, relu = GELU(), nn.ReLU()

x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)


plt.figure(figsize=(8,  3))
for i, (y, label) in enumerate(zip[y_gelu, y_relu], ["GELU", "RELU"], 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label} {x}")
    plt.grid(True)
    
plt.tight_layout()
plt.show()