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

import os

class Config:
    lr = 0.00005
    nz = 100  # noise dimension
    image_size = 64
    image_size2 = 64
    nc = 3  # chanel of img
    ngf = 64  # generate channel
    ndf = 64  # discriminative channel
    beta1 = 0.5
    batch_size = 32
    max_epoch = 1  # =1 when debug, =50 for standard training
    workers = 2
    gpu = t.cuda.is_available()  # use gpu or not
    clamp_num = 0.01  # WGAN clip gradient


opt = Config()

class Discriminator(nn.Module):
    def __init__(self, save_path='epoch_wnetd.pth'):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(opt.nc, opt.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(opt.ndf * 8, 1, 4, 1, 0, bias=False),

            # remove sigmoid
            # nn.Sigmoid()
        )

        self.save_path = save_path
        try:
            self.net.load_state_dict(t.load(save_path))
        except:
            self.net.apply(self.weight_init)

    def weight_init(self, m):
        # weight_initialization: important for wgan
        class_name = m.__class__.__name__
        if class_name.find('Conv') != -1:
            m.weight.data.normal_(0, 0.02)
        elif class_name.find('Norm') != -1:
            m.weight.data.normal_(1.0, 0.02)

    def forward(self, input):
        output = self.net(input)
        return output


class Generator(nn.Module):
    def __init__(self, save_path='epoch_wnetg.pth'):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(opt.nz, opt.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(opt.ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(opt.ngf * 8, opt.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(opt.ngf, opt.nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self.save_path = save_path
        try:
            self.net.load_state_dict(t.load(save_path))
        except:
            self.net.apply(self.weight_init)

    def weight_init(self, m):
        # weight_initialization: important for wgan
        class_name = m.__class__.__name__
        if class_name.find('Conv') != -1:
            m.weight.data.normal_(0, 0.02)
        elif class_name.find('Norm') != -1:
            m.weight.data.normal_(1.0, 0.02)

    def forward(self, input):
        output = self.net(input)
        return output


def train(dataloader, netg_path, netd_path, num_epochs=50):
    netg = Generator(netg_path)
    netd = Discriminator(netd_path)
    os.makedirs("Images/WGAN", exist_ok=True)

    optimizerD = RMSprop(netd.parameters(), lr=opt.lr)
    optimizerG = RMSprop(netg.parameters(), lr=opt.lr)

    fix_noise = Variable(t.FloatTensor(opt.batch_size, opt.nz, 1, 1).normal_(0, 1))
    if opt.gpu:
        print("using gpu")
        fix_noise = fix_noise.cuda()
        netd.cuda()
        netg.cuda()

    # begin training
    print('begin training, be patient...')
    one = t.FloatTensor([1])
    mone = -1 * one

    for epoch in range(num_epochs):
        print("Current epoch:", epoch)
        for ii, data in enumerate(dataloader, 0):
            real, _ = data
            input = Variable(real)
            noise = t.randn(input.size(0), opt.nz, 1, 1)
            noise = Variable(noise)

            if opt.gpu:
                one = one.cuda()
                mone = mone.cuda()
                noise = noise.cuda()
                input = input.cuda()

            # modification: clip param for discriminator
            for parm in netd.parameters():
                parm.data.clamp_(-opt.clamp_num, opt.clamp_num)

            # ----- train netd -----
            netd.zero_grad()
            ## train netd with real img
            output = netd(input)
            # output.backward(one)
            ## train netd with fake img
            fake_pic = netg(noise).detach()
            output2 = netd(fake_pic)
            # output2.backward(mone)
            loss_D = -t.mean(output) + t.mean(output2)  # loss_D = netd(fake) - netd(real)
            loss_D.backward()
            optimizerD.step()

            # ------ train netg -------
            # train netd more: because the better netd is,
            # the better netg will be
            if (ii + 1) % 5 == 0:
                netg.zero_grad()
                noise.data.normal_(0, 1)
                fake_pic = netg(noise)
                output = netd(fake_pic)
                # output.backward(one)
                loss_G = -t.mean(output)
                loss_G.backward()

                optimizerG.step()
                if ii % 100 == 0: pass
            if ii % 100 == 4:
                print("loss_D", loss_D, end='\t')
                print("loss_G", loss_G)

        fake_u = netg(fix_noise)
        imgs = make_grid(fake_u.data * 0.5 + 0.5).cpu()  # CHW
        plt.imshow(imgs.permute(1, 2, 0).numpy())  # HWC
        plt.show()
        t.save(netd.state_dict(), netd_path)
        t.save(netg.state_dict(), netg_path)
        if ((epoch + 1) % 10 == 0):
            t.save(netd.state_dict(), 'Images/WGAN/w_d%s' % epoch)
            t.save(netg.state_dict(), 'Images/WGAN/w_g%s' % epoch)
        save_image(fake_u.data[:32], "Images/WGAN/%d.png" % epoch, nrow=8, normalize=True)

    t.save(netd.state_dict(), netd_path)
    t.save(netg.state_dict(), netg_path)