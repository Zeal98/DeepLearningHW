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
import Modules.DCGAN as DCGAN
import Modules.WGAN as WGAN
import Modules.ACGAN as ACGAN

import os

os.makedirs("TestImages/CIFER10", exist_ok=True)
os.makedirs("TestImages/DCGAN", exist_ok=True)
os.makedirs("TestImages/WGAN", exist_ok=True)
os.makedirs("TestImages/ACGAN", exist_ok=True)
os.makedirs("TestImages/Noise", exist_ok=True)
os.makedirs("TestImages/DCGAN_2", exist_ok=True)

class Config():
    batch_size = 32
    image_size = 64
    image_size2 = 64
    nc = 3  # chanel of img
    gpu = t.cuda.is_available()  # use gpu or not
    workers = 2
    nz = 100    # dim of noise

opt = Config()

# data preprocess
transform = transforms.Compose([
    transforms.Resize(opt.image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

dataset = CIFAR10(root='cifar10/', transform=transform, download=True)
# dataloader with multiprocessing
dataloader = t.utils.data.DataLoader(dataset,
                                     opt.batch_size,
                                     shuffle=True,
                                     num_workers=opt.workers)


def cifer10(save_path, num_sample):
    os.makedirs(save_path[len(save_path)-1], exist_ok=True)
    for ii, data in enumerate(dataloader):
        image, _ = data
        image = Variable(image)
        for i in range(image.size(0)):
            save_image(image[i], save_path+"%s.png"%(ii*opt.batch_size+i), normalize=True)
        if (ii + 1) * opt.batch_size >= num_sample:
            break
    return save_path


def dcgan(save_path, num_sample, module_path="./Modules/weights/dcg"):
    os.makedirs(save_path[len(save_path)-1], exist_ok=True)
    netg = DCGAN.Generator(save_path=module_path)
    if opt.gpu:
        netg.cuda()
    for ii in range(int(num_sample/opt.batch_size)):
        noise = t.randn(opt.batch_size, opt.nz, 1, 1)
        noise = Variable(noise)
        if opt.gpu:
            noise = noise.cuda()
        output = netg(noise)
        for i in range(output.size(0)):
            save_image(output[i], save_path+"%s.png"%(ii*opt.batch_size+i), normalize=True)
    return save_path


def wgan(save_path, num_sample, module_path="./Modules/weights/wg"):
    os.makedirs(save_path[len(save_path)-1], exist_ok=True)
    netg = WGAN.Generator(save_path=module_path)
    if opt.gpu:
        netg.cuda()
    for ii in range(int(num_sample/opt.batch_size)):
        noise = t.randn(opt.batch_size, opt.nz, 1, 1)
        noise = Variable(noise)
        if opt.gpu:
            noise = noise.cuda()
        output = netg(noise)
        for i in range(output.size(0)):
            save_image(output[i], save_path+"%s.png"%(ii*opt.batch_size+i), normalize=True)
    return save_path


def acgan(save_path, num_sample, module_path="./Modules/weights/acg"):
    os.makedirs(save_path[len(save_path)-1], exist_ok=True)
    netg = ACGAN.Generator(save_path=module_path)
    if opt.gpu:
        netg.cuda()
    for ii in range(int(num_sample/opt.batch_size)):
        noise = t.randn(opt.batch_size, opt.nz, 1, 1)
        noise = Variable(noise)
        noise, rlabel = netg.noise_addclass(noise)
        if opt.gpu:
            noise = noise.cuda()
        output = netg(noise)
        for i in range(output.size(0)):
            save_image(output[i], save_path+"%s.png"%(ii*opt.batch_size+i), normalize=True)
    return save_path


if __name__ == "__main__":
    num_images = 2560
    cifer10("TestImages/CIFER10/", num_images)
    dcgan("TestImages/DCGAN/", num_images, "Modules/weights/dc_g19")
    wgan("TestImages/WGAN/", num_images, "Modules/weights/w_g49.pth")
    acgan("TestImages/ACGAN/", num_images, "Modules/weights.ac_g19")
    dcgan("TestImages/DCGAN_2/", num_images, "Modules/weights/dc_g39")
    dcgan("TestImages/Noise/", num_images, "Modules/weights/dcg")





