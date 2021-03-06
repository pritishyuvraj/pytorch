import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Hyper parameters
input_size = 784
output_size = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

# Loading Dataset
train_dataset = torchvision.datasets.MNIST(
    root='../../data',
    train=True,
    download=True,
    transform=transforms.ToTensor())

test_dataset = torchvision.datasets.MNIST(
    root='../../data', train=False, transform=transforms.ToTensor())

# Data Loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Logistic Regression
model = nn.Linear(input_size, output_size)

# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28)

        # Forward pass
        outputs = model(images)
        print(outputs.size(), labels.size())
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch {}/{}, Step {}/{}, Loss: {:.4f}'.format(
                epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 784)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

        print("Accuracy of the model on 1000 test images: {} %".format(
            100 * correct / total))
