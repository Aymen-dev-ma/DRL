import torch
import torch.nn as nn
import torch.optim as optim
from utils import FCNet, gaussian_kl, bernoulli_kl, gaussian_nll, recons_loss

class ModelDecon(nn.Module):
    def __init__(self, opts):
        super(ModelDecon, self).__init__()
        self.opts = opts
        self.fc_net = FCNet([opts['input_dim']] + opts['fc_layers'], [nn.ReLU()] * len(opts['fc_layers']))

    def forward(self, x):
        return self.fc_net(x)

    def p_z(self, batch_size, z_dim):
        mu_z = torch.zeros(batch_size, z_dim)
        cov_z = torch.ones(batch_size, z_dim)
        eps = torch.randn(batch_size, z_dim)
        z = mu_z + torch.sqrt(cov_z) * eps
        return z

    def p_x_g_z_u(self, z, u):
        zu_fea = torch.cat([z, u], dim=-1)
        mu, sigma = self.fc_net(zu_fea), self.fc_net(zu_fea)
        return mu, sigma

    def p_a_g_z_u(self, z, u):
        zu_fea = torch.cat([z, u], dim=-1)
        mu, sigma = self.fc_net(zu_fea), self.fc_net(zu_fea)
        return mu, sigma

    def p_r_g_z_a_u(self, z, a, u):
        zau_fea = torch.cat([z, a, u], dim=-1)
        mu, sigma = self.fc_net(zau_fea), self.fc_net(zau_fea)
        return mu, sigma

    def p_z_g_z_a(self, z, a):
        za_fea = torch.cat([z, a], dim=-1)
        mu, sigma = self.fc_net(za_fea), self.fc_net(za_fea)
        return mu, sigma

    def q_z_g_z_x_a_r(self, x_seq, a_seq, r_seq):
        xar_fea = torch.cat([x_seq, a_seq, r_seq], dim=-1)
        mu, sigma = self.fc_net(xar_fea), self.fc_net(xar_fea)
        return mu, sigma

    def q_u_g_x_a_r(self, x_seq, a_seq, r_seq):
        xar_fea = torch.cat([x_seq, a_seq, r_seq], dim=-1)
        logits = self.fc_net(xar_fea)
        prediction = torch.sigmoid(logits)
        return logits, prediction

    def q_a_g_x(self, x):
        mu, sigma = self.fc_net(x), self.fc_net(x)
        return mu, sigma

    def q_r_g_x_a(self, x, a):
        xa_fea = torch.cat([x, a], dim=-1)
        mu, sigma = self.fc_net(xa_fea), self.fc_net(xa_fea)
        return mu, sigma

    def neg_elbo(self, x_seq, a_seq, r_seq, u_seq, anneal=1):
        mu_q, cov_q = self.q_z_g_z_x_a_r(x_seq, a_seq, r_seq)
        eps = torch.randn_like(mu_q)
        z_q_samples = mu_q + eps * torch.sqrt(cov_q)

        mu_p, cov_p = self.p_z_g_z_a(z_q_samples, a_seq)
        mu_prior = torch.zeros_like(mu_p)
        cov_prior = torch.ones_like(cov_p)

        kl_divergence = gaussian_kl(mu_prior, cov_prior, mu_q, cov_q)
        u_logits, u_prediction = self.q_u_g_x_a_r(x_seq, a_seq, r_seq)
        u_prior = torch.full_like(u_prediction, 0.5)
        u_kl_divergence = bernoulli_kl(u_prior, u_prediction)

        mu_pxgz, cov_pxgz = self.p_x_g_z_u(z_q_samples, u_prediction)
        mu_pagz, cov_pagz = self.p_a_g_z_u(z_q_samples, u_prediction)
        mu_prgza, cov_prgza = self.p_r_g_z_a_u(z_q_samples, a_seq, u_prediction)

        mu_qagx, cov_qagx = self.q_a_g_x(x_seq)
        mu_qrgxa, cov_qrgxa = self.q_r_g_x_a(x_seq, a_seq)

        nll_pxgz = gaussian_nll(x_seq, mu_pxgz, cov_pxgz)
        nll_pagz = gaussian_nll(a_seq, mu_pagz, cov_pagz)
        nll_prgza = gaussian_nll(r_seq, mu_prgza, cov_prgza)
        nll_qagx = gaussian_nll(a_seq, mu_qagx, cov_qagx)
        nll_qrgxa = gaussian_nll(r_seq, mu_qrgxa, cov_qrgxa)

        nll = nll_pxgz + nll_pagz + nll_prgza + anneal * kl_divergence + nll_qagx + nll_qrgxa + u_kl_divergence

        correct_prediction = torch.eq(torch.round(u_prediction), u_seq).float()
        u_accuracy = correct_prediction.mean()

        return nll, kl_divergence, u_kl_divergence, u_accuracy

    def train_model(self, data):
        optimizer = optim.Adam(self.parameters(), lr=self.opts['lr'])
        for epoch in range(self.opts['epochs']):
            for batch in data:
                x_seq, a_seq, r_seq, u_seq = batch
                optimizer.zero_grad()
                loss, _, _, _ = self.neg_elbo(x_seq, a_seq, r_seq, u_seq, anneal=1)
                loss.backward()
                optimizer.step()
                print(f"Epoch {epoch}, Loss: {loss.item()}")
