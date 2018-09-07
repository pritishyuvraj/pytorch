import torch
import torch.nn as nn

# Create tensors of dimension 10*3, 10*2
x = torch.randn(10, 3)
y = torch.randn(10, 2)

# Build a fully connected layer
linear = nn.Linear(3, 2)
print('w: ', linear.weight)
print('b: ', linear.bias)

# Build a loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# Forward Pass
pred = linear(x)

# Computer loss
loss = criterion(pred, y)
print('loss: ', loss.item())

# Backward Pass
loss.backward()

# Print out the gradients
print('dL/dW: ', linear.weight.grad)
print('dL/db: ', linear.bias.grad)

# print out the loss after 1-step gradient descent
pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization', loss.item())
