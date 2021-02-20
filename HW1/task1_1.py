# Task1-1
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F


class Module_0(torch.nn.Module):
    def __init__(self):
        super(Module_0, self).__init__()
        self.hidden1 = torch.nn.Linear(1, 190)
        self.predict = torch.nn.Linear(190, 1)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = self.predict(x)
        return x


class Module_1(torch.nn.Module):
    def __init__(self):
        super(Module_1, self).__init__()
        self.hidden1 = torch.nn.Linear(1, 5)
        self.hidden2 = torch.nn.Linear(5, 10)
        self.hidden3 = torch.nn.Linear(10, 10)
        self.hidden4 = torch.nn.Linear(10, 10)
        self.hidden5 = torch.nn.Linear(10, 10)
        self.hidden6 = torch.nn.Linear(10, 10)
        self.hidden7 = torch.nn.Linear(10, 10)
        self.hidden8 = torch.nn.Linear(10, 10)
        self.hidden9 = torch.nn.Linear(10, 5)
        self.predict = torch.nn.Linear(5, 1)
        #

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.hidden5(x))
        x = F.relu(self.hidden6(x))
        x = F.relu(self.hidden7(x))
        x = F.relu(self.hidden8(x))
        x = F.relu(self.hidden9(x))
        x = self.predict(x)
        return x


class Module_2(torch.nn.Module):
    def __init__(self):
        super(Module_2, self).__init__()
        self.hidden1 = torch.nn.Linear(1, 10)
        self.hidden2 = torch.nn.Linear(10, 18)
        self.hidden3 = torch.nn.Linear(18, 15)
        self.hidden4 = torch.nn.Linear(15, 4)
        self.predict = torch.nn.Linear(4, 1)
        #

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = self.predict(x)
        return x


def train():
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1) # 100 samples between (-1, 1)
    y = x*x + x
    x, y = Variable(x), Variable(y)

    learning_rate = 0.001
    module_0 = Module_0()
    print("Module_0:", module_0)
    loss_func = F.mse_loss
    optimizer_0 = torch.optim.SGD(module_0.parameters(), lr=learning_rate)

    module_1 = Module_1()
    print("Module_1:", module_1)
    optimizer_1 = torch.optim.SGD(module_1.parameters(), lr=learning_rate)

    module_2 = Module_2()
    print("Module_2:", module_2)
    optimizer_2 = torch.optim.SGD(module_2.parameters(), lr=learning_rate)

    num_epochs = 20000
    epoch = torch.unsqueeze(torch.linspace(0, num_epochs, num_epochs), dim=1)
    loss_seq_0 = torch.unsqueeze(torch.linspace(0, num_epochs, num_epochs), dim=1)
    loss_seq_1 = torch.unsqueeze(torch.linspace(0, num_epochs, num_epochs), dim=1)
    loss_seq_2 = torch.unsqueeze(torch.linspace(0, num_epochs, num_epochs), dim=1)
    epoch, loss_seq_0, loss_seq_1, loss_seq_2 = Variable(epoch), Variable(loss_seq_0), Variable(loss_seq_1), Variable(loss_seq_2)
    for t in range(num_epochs):
        prediction_0 = module_0(x)
        prediction_1 = module_1(x)
        prediction_2 = module_2(x)
        loss_0 = loss_func(prediction_0, y)
        loss_1 = loss_func(prediction_1, y)
        loss_2 = loss_func(prediction_2, y)
        loss_seq_0[t] = loss_0
        loss_seq_1[t] = loss_1
        loss_seq_2[t] = loss_2

        optimizer_0.zero_grad()
        loss_0.backward()
        optimizer_0.step()

        optimizer_1.zero_grad()
        loss_1.backward()
        optimizer_1.step()

        optimizer_2.zero_grad()
        loss_2.backward()
        optimizer_2.step()

    # plot the result each 50 epochs
    # if t % 50 == 0:
    print("prediction curve:\n")
    plt.figure()
    plt.cla()
    plt.scatter(x.data.numpy(), y.data.numpy())  # real curve
    plt.plot(x.data.numpy(), prediction_0.data.numpy(), 'g-', lw=5)
    plt.plot(x.data.numpy(), prediction_1.data.numpy(), 'r-', lw=5)   # predicted curve 1
    plt.plot(x.data.numpy(), prediction_2.data.numpy(), 'b-', lw=5)  # predicted curve 2

    # plot the loss_epoch curve
    plt.figure()
    plt.cla()
    plt.plot(epoch.data.numpy(), loss_seq_0.data.numpy(), 'g-', lw=5)
    plt.plot(epoch.data.numpy(), loss_seq_1.data.numpy(), 'r-', lw=5)
    plt.plot(epoch.data.numpy(), loss_seq_2.data.numpy(), 'b-', lw=5)

    plt.ioff()
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
