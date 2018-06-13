import torch
import torch.nn as nn
import torchvision
import numpy as np
from collections import defaultdict

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # initialize file path or list of file name
        self.file_path = '../../data/iris/iris.data'
        self.toy_dataset = np.random.rand(10, 4)
        self.read_file = [
            line.split(',') for line in open(self.file_path).read().split('\n')
        ]

        temp = []

        # File cleaning
        for index, _line in enumerate(self.read_file):
            if self.read_file[index][0] == '' or self.read_file[index][0] == ' ':
                pass
            else:
                temp.append(self.read_file[index])

        self.read_file = temp

        self.constants = defaultdict(int)
        start_index = 0

        # Defininig constants
        for line in self.read_file:
            if line[-1] not in self.constants:
                self.constants[line[-1]] = start_index
                start_index += 1

        print(self.constants)
        # print(self.read_file)

        # Modifying the file
        for index, _line in enumerate(self.read_file):
            # print (self.read_file[index][-1], self.read_file[index])
            self.read_file[index][-1] = self.constants[self.read_file[index][
                -1]]
            # print(self.read_file[index][-1], self.read_file[index])
            self.read_file[index] = [float(x) for x in self.read_file[index]]
            # print(self.read_file[index])

        self.input_size = len(self.read_file[0][0:-1])
        self.output_size = len(self.constants)

        self.database = np.asarray(self.read_file)
        # print(self.database)
        print(self.database.shape)

    def __getitem__(self, index):
        # Return a data pair(eg. image and label)
        return self.database[index][0:-1], int(self.database[index][-1])

    def __len__(self):
        # total size of the dataset
        return self.database.shape[0]


# Fully connected Neural Network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


if __name__ == '__main__':
    custom_dataset = CustomDataset()
    train_loader = torch.utils.data.DataLoader(
        dataset=custom_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=custom_dataset, batch_size=64, shuffle=False)

    data_iter = iter(train_loader)

    # Defining model
    model = NeuralNet(custom_dataset.input_size, 10,
                      custom_dataset.output_size)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for i in range(5000):
        for images, labels in train_loader:
            images = images.float().to(device)
            labels = labels.float().to(device)
            # Forward Pass
            outputs = model(images.float())
            # outputs = outputs.float()
            loss = criterion(outputs, labels.long())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 1000 == 0:
                print("Loss ", loss.item())

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.float()
            labels = labels.long()
            # _, labels = torch.max(labels.data, 1)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # print(predicted, labels)
            total += images.size(0)
            correct += (predicted == labels).sum()
            print("Accuracy -> {}/{}: {:.4f}".format(
                correct, total,
                float(correct) / float(total) * 100.0))
