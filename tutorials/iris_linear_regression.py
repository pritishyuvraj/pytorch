import torch
import torch.nn as nn
import torchvision
import numpy as np
from collections import defaultdict


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
        one_hot_encoded_output = [0 for x in range(self.output_size)]
        one_hot_encoded_output[int(self.database[index][-1])] = 1
        one_hot_encoded_output = np.asarray(one_hot_encoded_output)
        # return self.database[index][0:-1], int(self.database[index][-1])
        return self.database[index][0:-1], one_hot_encoded_output

    def __len__(self):
        # total size of the dataset
        return self.database.shape[0]


if __name__ == '__main__':
    custom_dataset = CustomDataset()
    train_loader = torch.utils.data.DataLoader(
        dataset=custom_dataset, batch_size=64, shuffle=True)
    # print(dir(train_loader))

    data_iter = iter(train_loader)

    # Defining model
    model = nn.Linear(custom_dataset.input_size, custom_dataset.output_size)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for i in range(50000):
        for images, labels in train_loader:
            images = images.float()
            labels = labels.float()
            # Forward Pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 1000 == 0:
                print("Loss ", loss.item())

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.float()
            labels = labels.float()
            _, labels = torch.max(labels.data, 1)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += images.size(0)
            correct += (predicted == labels).sum()
            print("Accuracy -> {}/{}: {:.4f}".format(
                correct, total,
                float(correct) / float(total) * 100.0))
