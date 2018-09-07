import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms


# Create Tensors
x = torch.tensor(1., requires_grad = True)
w = torch.tensor(2., requires_grad = True)
b = torch.tensor(3., requires_grad = True)

# Build a computational graph
y = w * x + b

# Compute Gradients
y.backward()

# Print out the Gradients
print (x.grad)
print (w.grad)
print (b.grad)
