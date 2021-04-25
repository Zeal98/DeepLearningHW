import torch as t
from torch import nn
from torch.autograd import Variable
from torch.optim import RMSprop
from torch.optim import Adam
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torchvision.datasets import CIFAR10
from pylab import plt
import numpy as np

import os

class Config:
    lr = 0.0002
    nz = 100  # noise dimension
    image_size = 64
    image_size2 = 64
    nc = 3  # chanel of img
    ngf = 64  # generate channel
    ndf = 64  # discriminative channel
    beta1 = 0.5
    batch_size = 32
    max_epoch = 1  # =1 when debug, =50 when training
    workers = 2
    gpu = t.cuda.is_available()  # use gpu or not
    clamp_num = 0.01  # WGAN clip gradient


opt = Config()

class Discriminator(nn.Module):
    def __init__(self, save_path='epoch_acnetd.pth'):
        super().__init__()
        self.num_classes = 10

        net = []
        # 1:predefine
        channels_in = [3 + self.num_classes, 64, 128, 256]
        channels_out = [64, 128, 256, 512]
        padding = [1, 1, 1, 0]
        for i in range(len(channels_in)):
            net.append(nn.Conv2d(in_channels=channels_in[i], out_channels=channels_out[i],
                                 kernel_size=4, stride=2, padding=padding[i], bias=False))
            if i == 0:
                net.append(nn.LeakyReLU(0.2))
            else:
                net.append(nn.BatchNorm2d(num_features=channels_out[i]))
                net.append(nn.LeakyReLU(0.2))
                net.append(nn.Dropout(0.5))

        self.classify = nn.Linear(in_features=3 * 3 * 512, out_features=10)
        self.softmax = nn.Softmax(dim=1)
        self.disciminate = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.discriminator = nn.Sequential(*net)

        self.save_path = save_path
        try:
            self.load_state_dict(t.load(save_path))
        except:
            self.apply(self.weight_init)

    def weight_init(self, m):
        # weight_initialization: important for wgan
        class_name = m.__class__.__name__
        if class_name.find('Conv') != -1:
            m.weight.data.normal_(0, 0.02)
        elif class_name.find('Norm') != -1:
            m.weight.data.normal_(1.0, 0.02)

    def forward(self, input, label):
        class_onehot = t.zeros(label.size(0), self.num_classes, input.size(2), input.size(3))
        for i in range(label.size(0)):
            for j in range(self.num_classes):
                if j == label.data[i]:
                    class_onehot.data[i][j] = t.ones(input.size(2), input.size(3))
        if opt.gpu:
            class_onehot = class_onehot.cuda()

        # label = label.repeat(1, 1, x.size(2), x.size(3))

        data = t.cat(tensors=(input, class_onehot), dim=1)
        out = self.discriminator(data)
        out_ = out.view(input.size(0), -1)
        classsify = self.softmax(self.classify(out_))
        real_or_fake = self.sigmoid(self.disciminate(out))
        return real_or_fake.view(input.size(0), -1), classsify


class Generator(nn.Module):
    def __init__(self, save_path='epoch_acnetg.pth'):
        super(Generator, self).__init__()

        # first linear layer
        self.fc1 = nn.Linear(110, 384)
        # Transposed Convolution 2
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(384, 192, 4, 1, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        # Transposed Convolution 3
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(192, 96, 4, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 4, 2, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, 2, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(True),
        )
        # Transposed Convolution 5
        self.tconv6 = nn.Sequential(
            nn.ConvTranspose2d(24, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

        self.save_path = save_path
        try:
            self.load_state_dict(t.load(save_path))
            print("module loaded")
        except:
            self.apply(self.weight_init)

    def weight_init(self, m):
        # weight_initialization: important for wgan
        class_name = m.__class__.__name__
        if class_name.find('Conv') != -1:
            m.weight.data.normal_(0, 0.02)
        elif class_name.find('Norm') != -1:
            m.weight.data.normal_(1.0, 0.02)

    def noise_addclass(self, noise):
        rlabels = np.random.randint(0, 10, noise.size(0))
        rclass = t.zeros(noise.size(0), 10, noise.size(2), noise.size(3))
        for i in range(noise.size(0)):
            for j in range(10):
                if j == rlabels[i]:
                    rclass.data[i][j] = t.ones(noise.size(2), noise.size(3))
        if noise.device == 'cuda:0':
            rclass = rclass.cuda()
        return t.cat(tensors=(noise, rclass), dim=1), t.from_numpy(rlabels)

    def forward(self, input):
        # print(input.size())
        input = input.view(-1, 110)

        fc1 = self.fc1(input)
        fc1 = fc1.view(-1, 384, 1, 1)
        tconv2 = self.tconv2(fc1)
        tconv3 = self.tconv3(tconv2)
        tconv4 = self.tconv4(tconv3)
        tconv5 = self.tconv5(tconv4)
        tconv6 = self.tconv6(tconv5)
        output = tconv6
        return output


def train(dataloader, netg_path, netd_path, num_epochs=50):
    netg = Generator(netg_path)
    netd = Discriminator(netd_path)
    os.makedirs("Images/ACGAN", exist_ok=True)

    # optimizer
    optimizerD = Adam(netd.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizerG = Adam(netg.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # criterion
    criterion = nn.BCELoss()
    criterion_c = nn.NLLLoss()

    fix_noise = Variable(t.FloatTensor(opt.batch_size, opt.nz, 1, 1).normal_(0, 1))
    fix_noise, fix_label = netg.noise_addclass(fix_noise)

    if opt.gpu:
        print("using gpu")
        fix_noise = fix_noise.cuda()
        netd.cuda()
        netg.cuda()

    # begin training
    print('begin training, be patient...')
    # one = t.FloatTensor([1])
    # mone = -1 * one
    one = t.ones(32, 1, 1, 1)
    mone = 0 * one

    for epoch in range(num_epochs):
        print("Current epoch:", epoch)
        for ii, data in enumerate(dataloader, 0):
            real, label = data
            input = Variable(real)
            label = Variable(label)
            noise = t.randn(input.size(0), opt.nz, 1, 1)
            noise = Variable(noise)
            noise, rlabel = netg.noise_addclass(noise)
            rlabel = rlabel.to(rlabel.device, dtype=t.int64)
            one = t.ones(input.size()[0], 1)
            mone = 0 * one

            if opt.gpu:
                one = one.cuda()
                mone = mone.cuda()
                noise = noise.cuda()
                rlabel = rlabel.cuda()
                input = input.cuda()
                label = label.cuda()

            # ----- train netd -----
            netd.zero_grad()
            ## train netd with real img
            is_real_r, class_pre_r = netd(input, label)
            # print(is_real_r.size(), class_pre_r.size())
            # loss_D = criterion(output, one)
            err_r = criterion(is_real_r, one) + criterion_c(class_pre_r, label)
            err_r.backward()
            # output.backward(one)

            ## train netd with fake img

            fake_pic = netg(noise).detach()
            # print(fake_pic.size())
            is_real_f, class_pre_f = netd(fake_pic, rlabel)
            # loss_D = criterion(output, mone)
            err_f = criterion(is_real_f, mone) + criterion_c(class_pre_f, rlabel)
            loss_D = err_r + err_f
            # output2.backward(mone)
            # loss_D = -t.mean(output) + t.mean(output2)# loss_D = netd(fake) - netd(real)
            err_f.backward()
            optimizerD.step()

            # ------ train netg -------
            # train netd more: because the better netd is,
            # the better netg will be

            netg.zero_grad()
            # noise.data.normal_(0, 1)
            fake_pic = netg(noise)
            is_real, class_pre = netd(fake_pic, rlabel)
            loss_G = criterion(is_real, one) + criterion_c(class_pre, rlabel)
            # output.backward(one)
            # loss_G = -t.mean(output)
            loss_G.backward()

            optimizerG.step()
            # if ii % 100 == 0: pass
            if ii % 100 == 4:
                print("loss_D", loss_D.data, end='\t')
                print("loss_G", loss_G.data)

        fake_u = netg(fix_noise)
        imgs = make_grid(fake_u.data * 0.5 + 0.5).cpu()  # CHW
        plt.imshow(imgs.permute(1, 2, 0).numpy())  # HWC
        plt.show()
        t.save(netd.state_dict(), netd_path)
        t.save(netg.state_dict(), netg_path)
        if ((epoch + 1) % 10 == 0):
            t.save(netd.state_dict(), 'Images/ACGAN/ac_d%s' % epoch)
            t.save(netg.state_dict(), 'Images/ACGAN/ac_g%s' % epoch)
        save_image(fake_u.data[:32], "Images/ACGAN/%d.png" % epoch, nrow=8, normalize=True)

    t.save(netd.state_dict(), netd_path)
    t.save(netg.state_dict(), netg_path)

