from __future__ import print_function
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as dset
from torchvision import transforms


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # 定义编码器
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(16,32,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2,inplace=True),
        )
        self.encoder_fc1=nn.Linear(32*7*7,nz)
        self.encoder_fc2=nn.Linear(32*7*7,nz)
        self.Sigmoid = nn.Sigmoid()
        self.decoder_fc = nn.Linear(nz+10,32 * 7 * 7)
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def noise_reparameterize(self,mean,logvar):
        eps = torch.randn(mean.shape).to(device)
        z = mean + eps * torch.exp(logvar)
        return z

    def forward(self, x):
        z = self.encoder(x)
        output = self.decoder(z)
        return output
    def encoder(self,x):
        out1, out2 = self.encoder_conv(x), self.encoder_conv(x)
        mean = self.encoder_fc1(out1.view(out1.shape[0], -1))
        logstd = self.encoder_fc2(out2.view(out2.shape[0], -1))
        z = self.noise_reparameterize(mean, logstd)
        return z,mean,logstd
    def decoder(self,z):
        out3 = self.decoder_fc(z)
        out3 = out3.view(out3.shape[0], 32, 7, 7)
        out3 = self.decoder_deconv(out3)
        return out3

if __name__ == '__main__':
    nz=100
    device='cpu'
    vae = VAE()
    vae.load_state_dict(torch.load('./result_train_20230727/CVAE-GAN-VAE.pth'))
    outputs = []
    for num in range(10):
        label = torch.Tensor([num]).repeat(10).long()
        label_onehot = torch.zeros((10, 10))
        label_onehot[torch.arange(10), label] = 1
        z = torch.randn((10, nz))
        z = torch.cat([z, label_onehot], 1)
        outputs.append(vae.decoder(z).view(z.shape[0], 1, 28, 28))
    outputs = torch.cat(outputs)
    img = make_grid(outputs, nrow=10, normalize=False).clamp(0, 1).detach().numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)), interpolation='nearest')
    plt.show()
 
