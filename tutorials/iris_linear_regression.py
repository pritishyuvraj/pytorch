import torch
import torch.nn as nn
import torchvision
import numpy as np


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # initialize file path or list of file name
        self.file_path = '../../data/iris/iris.data'
        self.toy_dataset = np.random.rand(10, 4)
        pass

    def __getitem__(self, index):
        # Return a data pair(eg. image and label)
        return self.toy_dataset[index, 0:3], self.toy_dataset[index, 3]

    def __len__(self):
        # total size of the dataset
        return len(self.toy_dataset)


if __name__ == '__main__':
    custom_dataset = CustomDataset()
    train_loader = torch.utils.data.DataLoader(
        dataset=custom_dataset, batch_size=64, shuffle=True)
    print (dir(train_loader))

    data_iter = iter(train_loader)

    for images, labels in train_loader:
        print (images.size(), labels.size())
