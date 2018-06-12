import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

#download and construct cifar-10 dataset
train_dataset = torchvision.datasets.CIFAR10(
    root='../../data/',
    train=True,
    transform=transforms.ToTensor(),
    download=True)


#Fetch one data pair (read data from disk)
image, label = train_dataset[0]
print("iamge size", image.size())
print("label", label)

# Data loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=64, shuffle=True)

data_iter = iter(train_loader)

images, labels = data_iter.next()

# the model
linear = torch.nn.Linear(3072, 10)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr =0.01)

for images, labels in train_loader:
    images = images.view(-1, 3072)
    # print(images.size(), labels.size())
    one_hot_labels = np.zeros((labels.size()[0], 10))
    one_hot_labels[np.arange(labels.size()[0]), labels] = 1
    one_hot_labels = torch.from_numpy(one_hot_labels).double()
    one_hot_labels = one_hot_labels.double()
    temp_pred = linear(images).double()
    # temp_pred = temp_pred.type(torch.FloatTensor)
    # print (temp_pred, one_hot_labels)
    # print(type(temp_pred), type(one_hot_labels))
    loss = criterion(temp_pred, one_hot_labels)
    print("loss: ", loss.item())

# Pretrained the model
resnet = torchvision.models.resnet18(pretrained=True)

for param in resnet.parameters():
    param.requires_grad = False

resnet.fc = torch.nn.Linear(resnet.fc.in_features, 100)

# Forward pass
images = torch.randn(64, 3, 224, 224)

output = resnet(images)
print(output.size())

torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))
