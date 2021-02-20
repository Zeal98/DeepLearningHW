# Task1-2

import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision as tv

import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage


# pre-processing
root = 'E:\\Zqc Documents\\CU\\2021 Spring\\8430_DeepLearning\\HW1\\source\\cifar-10-batches-py' # path to CIFAR-10
# root = input("input the path to CIFER-10")

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])

trainset = tv.datasets.CIFAR10(
                    root=root,
                    train=True,
                    download=True,
                    transform=transform)

trainloader = torch.utils.data.DataLoader(
                    trainset,
                    batch_size=4,
                    shuffle=True,
                    num_workers=2)

testset = tv.datasets.CIFAR10(
                    root,
                    train=False,
                    download=True,
                    transform=transform)

testloader = torch.utils.data.DataLoader(
                    testset,
                    batch_size=4,
                    shuffle=False,
                    num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Module_0(torch.nn.Module):
    def __init__(self):
        super(Module_0, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.hidden1 = torch.nn.Linear(16*5*5, 120)
        self.hidden2 = torch.nn.Linear(120, 84)
        self.hidden3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        return x


class Module_1(torch.nn.Module):
    def __init__(self):
        super(Module_1, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.hidden1 = torch.nn.Linear(16*5*5, 120)
        self.hidden2 = torch.nn.Linear(120, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return x


class Module_2(torch.nn.Module):
    def __init__(self):
        super(Module_2, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 10, 5)
        self.conv3 = torch.nn.Conv2d(10, 16, 2)
        self.hidden1 = torch.nn.Linear(16*2*2, 120)
        self.hidden2 = torch.nn.Linear(120, 84)
        self.hidden3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        return x


def train():
    loss_func = torch.nn.CrossEntropyLoss()
    torch.set_num_threads(8)
    num_epochs = 30
    learning_rate = 0.001
    momentum = 0.9

    module_0 = Module_0()
    print("Module_0:", module_0)
    optimizer_0 = torch.optim.SGD(module_0.parameters(), lr=learning_rate, momentum=momentum)

    module_1 = Module_1()
    print("Module_1:", module_1)
    optimizer_1 = torch.optim.SGD(module_1.parameters(), lr=learning_rate, momentum=momentum)

    module_2 = Module_2()
    print("Module_2:", module_2)
    optimizer_2 = torch.optim.SGD(module_2.parameters(), lr=learning_rate, momentum=momentum)

    epoch = torch.unsqueeze(torch.linspace(0, num_epochs, num_epochs), dim=1)
    loss_seq_0 = torch.unsqueeze(torch.linspace(0, 0, num_epochs), dim=1)
    loss_seq_1 = torch.unsqueeze(torch.linspace(0, 0, num_epochs), dim=1)
    loss_seq_2 = torch.unsqueeze(torch.linspace(0, 0, num_epochs), dim=1)

    accuracy_0 = torch.unsqueeze(torch.linspace(0, 0, num_epochs), dim=1)
    accuracy_1 = torch.unsqueeze(torch.linspace(0, 0, num_epochs), dim=1)
    accuracy_2 = torch.unsqueeze(torch.linspace(0, 0, num_epochs), dim=1)

    epoch = Variable(epoch)
    loss_seq_0, loss_seq_1, loss_seq_2 = Variable(loss_seq_0), Variable(loss_seq_1), Variable(loss_seq_2)
    accuracy_0, accuracy_1, accuracy_2 = Variable(accuracy_0), Variable(accuracy_1), Variable(accuracy_2)

    plt.figure(1)
    plt.figure(2)
    for t in range(num_epochs):
        loss_0 = 0.0
        print("Current epochs finished: ", t)
        # training loop
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer_0.zero_grad()
            outputs_0 = module_0(inputs)
            loss = loss_func(outputs_0, labels)
            loss_seq_0[t] = loss
            loss.backward()
            optimizer_0.step()

            optimizer_1.zero_grad()
            outputs_1 = module_1(inputs)
            loss = loss_func(outputs_1, labels)
            loss_seq_1[t] = loss
            loss.backward()
            optimizer_1.step()

            optimizer_2.zero_grad()
            outputs_2 = module_0(inputs)
            loss = loss_func(outputs_2, labels)
            loss_seq_2[t] = loss
            loss.backward()
            optimizer_2.step()

        # testing loop
        total = 0
        correct_0 = 0
        correct_1 = 0
        correct_2 = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                total += labels.size(0)

                outputs_0 = module_0(images)
                _, predicted_0 = torch.max(outputs_0, 1)
                correct_0 += (predicted_0 == labels).sum()

                outputs_1 = module_1(images)
                _, predicted_1 = torch.max(outputs_1, 1)
                correct_1 += (predicted_1 == labels).sum()

                outputs_2 = module_2(images)
                _, predicted_2 = torch.max(outputs_2, 1)
                correct_2 += (predicted_2 == labels).sum()
        accuracy_0[t] = correct_0/total
        accuracy_1[t] = correct_1/total
        accuracy_2[t] = correct_2/total

        plt.figure(1)
        plt.cla()
        plt.plot(epoch.data.numpy(), loss_seq_0.data.numpy(), 'g-', lw=5)
        plt.plot(epoch.data.numpy(), loss_seq_1.data.numpy(), 'r-', lw=5)
        plt.plot(epoch.data.numpy(), loss_seq_2.data.numpy(), 'b-', lw=5)

        plt.figure(2)
        plt.cla()
        plt.plot(epoch.data.numpy(), accuracy_0.data.numpy(), 'g-', lw=5)
        plt.plot(epoch.data.numpy(), accuracy_1.data.numpy(), 'r-', lw=5)
        plt.plot(epoch.data.numpy(), accuracy_2.data.numpy(), 'b-', lw=5)

        plt.pause(0.2)

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    train()
