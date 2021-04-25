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

os.makedirs("Modules/weights", exist_ok=True)


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

if __name__ == "__main__":
    d_path = ["./Modules/weights/dcd", "./Modules/weights.wd", "./Modules/weights/acd"]
    g_path = ["./Modules/weights/dcg", "./Modules/weights/wg", "./Modules/weights/acg"]

    # WGAN.train(dataloader, netg_path=g_path[1], netd_path=d_path[1], num_epochs=1)
    # DCGAN.train(dataloader, netg_path=g_path[0], netd_path=d_path[0], num_epochs=1)
    ACGAN.train(dataloader, netg_path=g_path[2], netd_path=d_path[2], num_epochs=1)
