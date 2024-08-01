import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConnectedLayer(nn.Module):
    def __init__(self, in_dim, out_dim, init_bias=0.0):
        super(FullyConnectedLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, init_bias)

    def forward(self, x):
        return self.fc(x)

class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, init_bias=0.0):
        super(Conv2dLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.constant_(self.conv.bias, init_bias)

    def forward(self, x):
        return self.conv(x)

class Deconv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, init_bias=0.0):
        super(Deconv2dLayer, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        nn.init.xavier_uniform_(self.deconv.weight)
        nn.init.constant_(self.deconv.bias, init_bias)

    def forward(self, x):
        return self.deconv(x)

class FCNet(nn.Module):
    def __init__(self, layers, activations):
        super(FCNet, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(FullyConnectedLayer(layers[i], layers[i+1]))
        self.activations = activations

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.activations[i] is not None:
                x = self.activations[i](x)
        return x

def gaussian_kl(mu_p, cov_p, mu_q, cov_q):
    kl = torch.log(cov_q / cov_p) + (cov_p + (mu_p - mu_q).pow(2)) / cov_q - 1
    return 0.5 * kl.sum(dim=-1).mean()

def bernoulli_kl(u_p, u_q):
    kl = u_p * (torch.log(u_p + 1e-10) - torch.log(u_q + 1e-10)) + (1 - u_p) * (torch.log(1 - u_p + 1e-10) - torch.log(1 - u_q + 1e-10))
    return kl.mean()

def gaussian_nll(data, mu, cov):
    nll = 0.5 * (torch.log(2 * torch.pi) + torch.log(cov) + (data - mu).pow(2) / cov)
    return nll.sum(dim=-1).mean()

def recons_loss(cost, real, recons):
    if cost == 'l2':
        loss = torch.sqrt(1e-10 + (real - recons).pow(2).sum(dim=-1))
    elif cost == 'l2sq':
        loss = (real - recons).pow(2).sum(dim=-1)
    elif cost == 'l1':
        loss = (real - recons).abs().sum(dim=-1)
    elif cost == 'cross_entropy':
        loss = -(real * torch.log(recons + 1e-10) + (1 - real) * torch.log(1 - recons + 1e-10)).sum(dim=-1)
    return loss.mean()
