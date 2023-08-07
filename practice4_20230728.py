import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#定义网络结构
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
        # 定义解码器
        self.decoder_fc = nn.Linear(nz+10,32 * 7 * 7)
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Sigmoid(),
        )
    # 定义噪声
    def noise_reparameterize(self,mean,logvar):
        eps = torch.randn(mean.shape).to(device)
        z = mean + eps * torch.exp(logvar)
        return z
    # 前向计算
    def forward(self, x):
        z = self.encoder(x)
        output = self.decoder(z)
        return output
    # 编码器
    def encoder(self,x):
        out1, out2 = self.encoder_conv(x), self.encoder_conv(x)
        # 均值
        mean = self.encoder_fc1(out1.view(out1.shape[0], -1))
        # 标准差
        logstd = self.encoder_fc2(out2.view(out2.shape[0], -1))
        # 噪声
        z = self.noise_reparameterize(mean, logstd)
        return z,mean,logstd
    # 解码器
    def decoder(self,z):
        out3 = self.decoder_fc(z)
        out3 = out3.view(out3.shape[0], 32, 7, 7)
        out3 = self.decoder_deconv(out3)
        return out3
    

# 定义损失函数
def loss_function(recon_x,x,mean,logstd):
    MSE = MSECriterion(recon_x,x)
    var = torch.pow(torch.exp(logstd),2)
    KLD = -0.5 * torch.sum(1+torch.log(var)-torch.pow(mean,2)-var)
    return MSE+KLD

if __name__ == '__main__':
    dataset = 'mnist'
    batchSize = 128
    nz=100
    nepoch=30
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 可以优化运行效率
    cudnn.benchmark = True
    dataset = dset.MNIST(root='./data',
                         train=True,
                         transform=transforms.Compose([transforms.ToTensor()]),
                         download=True
                         )
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batchSize,
                                             shuffle=True)

    vae = VAE().to(device)
    criterion = nn.BCELoss().to(device)
    MSECriterion = nn.MSELoss().to(device)
    optimizerVAE = optim.Adam(vae.parameters(), lr=0.0001)
    loss_history = []
    for epoch in range(nepoch):
        for i, (data,label) in enumerate(dataloader, 0):
            # 先处理一下数据
            data = data.to(device)
            label_onehot = torch.zeros((data.shape[0], 10)).to(device)
            label_onehot[torch.arange(data.shape[0]), label] = 1
            batch_size = data.shape[0]
           
            # 更新VAE
            z,mean,logstd = vae.encoder(data)
            z = torch.cat([z,label_onehot],1)
            recon_data = vae.decoder(z)
            vae_loss1 = loss_function(recon_data,data,mean,logstd)
           
            vae.zero_grad()
            vae_loss = vae_loss1
            vae_loss.backward()
            optimizerVAE.step()
            
            if i%100==0:
                print('[%d/%d][%d/%d]Loss: %.4f'
                      % (epoch+1, nepoch, i, len(dataloader),vae_loss.item()))
        loss_history.append(vae_loss.item())
                
torch.save(vae.state_dict(), './result_train_20230728/CVAE-GAN-VAE_v3.pth')


plt.figure(figsize=(10, 5))
plt.plot(range(nepoch), loss_history)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()