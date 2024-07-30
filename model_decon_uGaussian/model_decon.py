import numpy as np
import tensorflow as tf
import os
import time
from utils import *

class Model_Decon(object):

    def __init__(self, opts):
        self.opts = opts
        np.random.seed(self.opts['seed'])
        self.fb = 'forward'
        self.u = None
        self.u_x_list = []
        self.u_a_list = []
        self.u_r_list = []

    def build_model(self):
        self.x_seq = tf.keras.Input(shape=(self.opts['nsteps'], self.opts['x_dim']))
        self.a_seq = tf.keras.Input(shape=(self.opts['nsteps'], self.opts['a_dim']))
        self.r_seq = tf.keras.Input(shape=(self.opts['nsteps'], self.opts['r_dim']))
        self.u_seq = tf.keras.Input(shape=(self.opts['nsteps'], self.opts['u_dim']))
        self.mask = tf.keras.Input(shape=(self.opts['nsteps'], self.opts['mask_dim']))

        self.z, self.mu_q, self.cov_q = self.q_z_g_z_x_a_r(self.x_seq, self.a_seq, self.r_seq, self.mask)
        self.mu_u, self.cov_u = self.q_u_g_x_a_r(self.x_seq, self.a_seq, self.r_seq, self.mask)

        self.mu_pxgz, self.cov_pxgz = self.p_x_g_z_u(self.z, self.mu_u)
        self.mu_pagz, self.cov_pagz = self.p_a_g_z_u(self.z, self.mu_u)
        self.mu_prgza, self.cov_prgza = self.p_r_g_z_a_u(self.z, self.a_seq, self.mu_u)

        self.loss = self.neg_elbo(self.x_seq, self.a_seq, self.r_seq, self.u_seq, mask=self.mask)
        self.model = tf.keras.Model(inputs=[self.x_seq, self.a_seq, self.r_seq, self.u_seq, self.mask], outputs=self.loss)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.opts['lr']), loss=lambda y_true, y_pred: y_pred)

    def p_z(self):
        mu_z = tf.zeros([self.opts['batch_size'], self.opts['z_dim']], tf.float32)
        cov_z = tf.ones([self.opts['batch_size'], self.opts['z_dim']], tf.float32)
        eps = tf.random.normal((self.opts['batch_size'], self.opts['z_dim']),
                               0., 1., dtype=tf.float32)
        z = mu_z + tf.multiply(eps, tf.sqrt(1e-8 + cov_z))
        return z

    def p_x_g_z_u(self, z, u):
        if len(z.shape) > 2:
            u = tf.expand_dims(u, 1)
            u = tf.tile(u, [1, z.shape[1], 1])

        z_fea = fc_net(self.opts, z, self.opts['pxgz_net_layers'], self.opts['pxgz_net_outlayers'], 'pxgz_net')
        u_fea = fc_net(self.opts, u, self.opts['pxgu_net_layers'], self.opts['pxgu_net_outlayers'], 'pxgu_net')

        if len(z.shape) > 2:
            zu_fea = tf.concat([z_fea, u_fea], 2)
        else:
            zu_fea = tf.concat([z_fea, u_fea], 1)

        if self.opts['is_conv']:
            z_fea = fc_net(self.opts, zu_fea, self.opts['pxgzu_prenet_layers'],
                           self.opts['pxgzu_prenet_outlayers'], 'pxgz_prenet')
            z_fea = tf.reshape(z_fea, z_fea.shape[:-1] + [4, 4, 32])
            mu, sigma = decoder(self.opts, z_fea, self.opts['pxgzu_in_shape'],
                                self.opts['pxgzu_out_shape'], 'pxgzu_conv_net')
            mu = tf.reshape(mu, mu.shape[:-3] + [-1])
            sigma = tf.reshape(sigma, sigma.shape[:-3] + [-1])
        else:
            mu, sigma = fc_net(self.opts, zu_fea, self.opts['pxgzu_net_layers'],
                               self.opts['pxgzu_net_outlayers'], 'pxgzu_net')
        return mu, sigma

    def p_a_g_z_u(self, z, u):
        if len(z.shape) > 2:
            u = tf.expand_dims(u, 1)
            u = tf.tile(u, [1, z.shape[1], 1])

        z_fea = fc_net(self.opts, z, self.opts['pagz_net_layers'], self.opts['pagz_net_outlayers'], 'pagz_net')
        u_fea = fc_net(self.opts, u, self.opts['pagu_net_layers'], self.opts['pagu_net_outlayers'], 'pagu_net')

        if len(z.shape) > 2:
            zu_fea = tf.concat([z_fea, u_fea], 2)
        else:
            zu_fea = tf.concat([z_fea, u_fea], 1)

        mu, sigma = fc_net(self.opts, zu_fea, self.opts['pagzu_net_layers'],
                           self.opts['pagzu_net_outlayers'], 'pagzu_net')
        mu = mu * self.opts['a_range']

        return mu, sigma

    def p_r_g_z_a_u(self, z, a, u):
        if len(z.shape) > 2:
            u = tf.expand_dims(u, 1)
            u = tf.tile(u, [1, z.shape[1], 1])

        z_fea = fc_net(self.opts, z, self.opts['prgz_net_layers'], self.opts['prgz_net_outlayers'], 'prgz_net')
        a_fea = fc_net(self.opts, a, self.opts['prga_net_layers'], self.opts['prga_net_outlayers'], 'prga_net')
        u_fea = fc_net(self.opts, u, self.opts['prgu_net_layers'], self.opts['prgu_net_outlayers'], 'pagu_net')

        if len(z.shape) > 2:
            zau_fea = tf.concat([z_fea, a_fea, u_fea], 2)
        else:
            zau_fea = tf.concat([z_fea, a_fea, u_fea], 1)

        mu, sigma = fc_net(self.opts, zau_fea, self.opts['prgzau_net_layers'],
                           self.opts['prgzau_net_outlayers'], 'prgzau_net')

        mu = mu * (self.opts['r_range_upper'] - self.opts['r_range_lower']) + self.opts['r_range_lower']
        return mu, sigma

    def p_z_g_z_a(self, z, a):
        z_fea = fc_net(self.opts, z, self.opts['pzgz_net_layers'], self.opts['pzgz_net_outlayers'], 'pzgz_net')
        a_fea = fc_net(self.opts, a, self.opts['pzga_net_layers'], self.opts['pzga_net_outlayers'], 'pzga_net')

        if len(z.shape) > 2:
            az_fea = tf.concat([z_fea, a_fea], 2)
        else:
            az_fea = tf.concat([z_fea, a_fea], 1)

        h_az_fea = fc_net(self.opts, az_fea, self.opts['pzgza_net_layers'],
                          self.opts['pzgza_net_outlayers'], 'pzgza_net')
        h_mu = fc_net(self.opts, h_az_fea, self.opts['pzgza_mu_net_layers'],
                      self.opts['pzgza_mu_net_outlayers'], 'pzgza_mu_net')

        if self.opts['gated']:
            hg_az_fea = fc_net(self.opts, az_fea, self.opts['pzgza_pregate_net_layers'],
                               self.opts['pzgza_pregate_net_outlayers'], 'pzgza_pregate_net')
            gate = fc_net(self.opts, hg_az_fea, self.opts['pzgza_gate_net_layers'],
                          self.opts['pzgza_gate_net_outlayers'], 'pzgza_gate_net')
            mu = gate * h_mu + (1 - gate) * fc_net(self.opts, az_fea, self.opts['pzgza_gate_mu_net_layers'],
                                                   self.opts['pzgza_gate_mu_net_outlayers'], 'pzgza_gate_mu_net')
        else:
            mu = h_mu

        sigma = fc_net(self.opts, h_az_fea, self.opts['pzgza_sigma_net_layers'],
                       self.opts['pzgza_sigma_net_outlayers'], 'pzgza_sigma_net')

        return mu, sigma

    def q_z_g_z_x_a_r(self, x_seq, a_seq, r_seq, mask=None):
        if len(x_seq.shape) == 2:
            x_seq = tf.expand_dims(x_seq, 1)
            a_seq = tf.expand_dims(a_seq, 1)
            r_seq = tf.expand_dims(r_seq, 1)

        if self.opts['is_conv']:
            x_reshape = tf.reshape(x_seq, x_seq.shape[:-1] + [28, 28, 1])
            x_encoded = encoder(self.opts, x_reshape, self.opts['qzgx_in_channels'],
                                self.opts['qzgx_out_channel'], 'qzgx_conv_net')
            x_fea = fc_net(self.opts, x_encoded, self.opts['qzgx_encoded_net_layers'],
                           self.opts['qzgx_encoded_net_outlayers'], 'qzgx_encoded_net')
        else:
            x_fea = fc_net(self.opts, x_seq, self.opts['qzgx_net_layers'],
                           self.opts['qzgx_net_outlayers'], 'qzgx_net')

        a_fea = fc_net(self.opts, a_seq, self.opts['qzga_net_layers'], self.opts['qzga_net_outlayers'], 'qzga_net')
        r_fea = fc_net(self.opts, r_seq, self.opts['qzgr_net_layers'], self.opts['qzgr_net_outlayers'], 'qzgr_net')

        concat_xar = tf.concat([x_fea, a_fea, r_fea], 2)
        xar_fea = fc_net(self.opts, concat_xar, self.opts['qzgxar_net_layers'],
                         self.opts['qzgxar_net_outlayers'], 'qzgxar_net')

        h_r = self.lstm_net(xar_fea, 'R', mask)

        if self.opts['inference_model_type'] == 'LR':
            h_l = self.lstm_net(xar_fea, 'L', mask)
            h = (h_r + h_l)/2.
        else:
            h = h_r

        z_0 = tf.zeros([self.opts['batch_size'], self.opts['z_dim']], tf.float32)
        mu_0 = tf.zeros([self.opts['batch_size'], self.opts['z_dim']])
        cov_0 = tf.ones([self.opts['batch_size'], self.opts['z_dim']])

        h = h[:, tf.newaxis]
        a_fea = fc_net(self.opts, a_fea, self.opts['qagh_net_layers'], self.opts['qagh_net_outlayers'], 'qagh_net')
        a_fea = tf.transpose(a_fea, [1, 0, 2])
        a_fea = tf.concat([tf.ones([1, self.opts['batch_size'], tf.shape(a_fea)[2]]), a_fea[:-1, :, :]], 0)
        a_fea = a_fea[:, tf.newaxis]

        ha_concat = tf.concat([h, a_fea], 1)
        ha_split = tf.split(ha_concat, x_seq.shape[1], 0)
        ha_list = []
        for i in range(x_seq.shape[1]):
            ha_list.append(tf.reshape(ha_split[i], [2, self.opts['batch_size'], self.opts['lstm_dim']]))

        elements = tf.convert_to_tensor(ha_list)

        output_q = tf.scan(
            self.st_approx,
            elements,
            initializer=(z_0, mu_0, cov_0)
        )

        z = tf.transpose(output_q[0], [1, 0, 2])
        mu = tf.transpose(output_q[1], [1, 0, 2])
        cov = tf.transpose(output_q[2], [1, 0, 2])

        if len(x_seq.shape) == 2:
            z = tf.squeeze(z, [1])
            mu = tf.squeeze(mu, [1])
            cov = tf.squeeze(cov, [1])

        return z, mu, cov

    def q_u_g_x_a_r(self, x_seq, a_seq, r_seq, mask=None):
        if len(x_seq.shape) == 2:
            x_seq = tf.expand_dims(x_seq, 1)
            a_seq = tf.expand_dims(a_seq, 1)
            r_seq = tf.expand_dims(r_seq, 1)

        if self.opts['is_conv']:
            x_reshape = tf.reshape(x_seq, x_seq.shape[:-1] + [28, 28, 1])
            x_encoded = encoder(self.opts, x_reshape, self.opts['qzgx_in_channels'],
                                self.opts['qzgx_out_channel'], 'qzgx_conv_net', reuse=True)
            x_fea = fc_net(self.opts, x_encoded, self.opts['qzgx_encoded_net_layers'],
                           self.opts['qzgx_encoded_net_outlayers'], 'qzgx_encoded_net', reuse=True)
        else:
            x_fea = fc_net(self.opts, x_seq, self.opts['qugx_net_layers'],
                           self.opts['qugx_net_outlayers'], 'qugx_net')

        a_fea = fc_net(self.opts, a_seq, self.opts['qzga_net_layers'],
                       self.opts['qzga_net_outlayers'], 'qzga_net', reuse=True)
        r_fea = fc_net(self.opts, r_seq, self.opts['qzgr_net_layers'],
                       self.opts['qzgr_net_outlayers'], 'qzgr_net', reuse=True)

        concat_xar = tf.concat([x_fea, a_fea, r_fea], 2)
        xar_fea = fc_net(self.opts, concat_xar, self.opts['qzgxar_net_layers'],
                         self.opts['qzgxar_net_outlayers'], 'qzgxar_net', reuse=True)

        h_r = self.lstm_net(xar_fea, 'UR', mask)
        h_r = tf.reverse(h_r, [0])

        if self.opts['inference_model_type'] == 'LR':
            h_l = self.lstm_net(xar_fea, 'UL', mask)
            h_l = tf.reverse(h_l, [0])
            h = (h_r[0] + h_l[0])/2.
        else:
            h = h_r[0]

        h_trans = tf.reshape(h, [self.opts['batch_size'], -1])
        mu, cov = fc_net(self.opts, h_trans, self.opts['qugh_net_layers'],
                         self.opts['qugh_net_outlayers'], 'qugh_net')
        return mu, cov

    def q_a_g_x(self, x):
        if self.opts['is_conv']:
            x_reshape = tf.reshape(x, x.shape[:-1] + [28, 28, 1])
            x_encoded = encoder(self.opts, x_reshape, self.opts['qagx_in_channels'], self.opts['qagx_out_channel'],
                                'qagx_conv_net')
            mu, sigma = fc_net(self.opts, x_encoded, self.opts['qagx_encoded_net_layers'],
                               self.opts['qagx_encoded_net_outlayers'], 'qagx_encoded_net')
        else:
            mu, sigma = fc_net(self.opts, x, self.opts['qagx_net_layers'],
                               self.opts['qagx_net_outlayers'], 'qagx_net')
        mu = mu * self.opts['a_range']
        return mu, sigma

    def q_r_g_x_a(self, x, a):
        if self.opts['is_conv']:
            x_reshape = tf.reshape(x, x.shape[:-1] + [28, 28, 1])
            x_encoded = encoder(self.opts, x_reshape, self.opts['qrgx_in_channels'], self.opts['qrgx_out_channel'],
                                'qrgx_conv_net')
            x_fea = fc_net(self.opts, x_encoded, self.opts['qrgx_encoded_net_layers'],
                           self.opts['qrgx_encoded_net_outlayers'], 'qrgx_encoded_net')
        else:
            x_fea = fc_net(self.opts, x, self.opts['qrgx_net_layers'], self.opts['qrgx_net_outlayers'], 'qrgx_net')

        a_fea = fc_net(self.opts, a, self.opts['qrga_net_layers'], self.opts['qrga_net_outlayers'], 'qrga_net')

        if len(x.shape) > 2:
            ax_fea = tf.concat([x_fea, a_fea], 2)
        else:
            ax_fea = tf.concat([x_fea, a_fea], 1)

        mu, sigma = fc_net(self.opts, ax_fea, self.opts['qrgxa_net_layers'],
                           self.opts['qrgxa_net_outlayers'], 'qrgxa_net')
        mu = mu * (self.opts['r_range_upper'] - self.opts['r_range_lower']) + self.opts['r_range_lower']

        return mu, sigma

    def neg_elbo(self, x_seq, a_seq, r_seq, u_seq, anneal=1, mask=None):
        z_q, mu_q, cov_q = self.q_z_g_z_x_a_r(x_seq, a_seq, r_seq, mask)
        eps = tf.random.normal((self.opts['batch_size'], self.opts['nsteps'], self.opts['z_dim']),
                               0., 1., dtype=tf.float32)
        z_q_samples = mu_q + tf.multiply(eps, tf.sqrt(1e-8 + cov_q))

        mu_p, cov_p = self.p_z_g_z_a(z_q_samples, a_seq)
        mu_prior = tf.concat([tf.zeros([self.opts['batch_size'], 1, self.opts['z_dim']]), mu_p[:, :-1, :]], 1)
        cov_prior = tf.concat([tf.ones([self.opts['batch_size'], 1, self.opts['z_dim']]), cov_p[:, :-1, :]], 1)
        kl_divergence = gaussianKL(mu_prior, cov_prior, mu_q, cov_q, mask)

        mu_u, cov_u = self.q_u_g_x_a_r(x_seq, a_seq, r_seq, mask)
        mu_u_prior = tf.zeros([self.opts['batch_size'], self.opts['u_dim']])
        cov_u_prior = tf.ones([self.opts['batch_size'], self.opts['u_dim']])
        u_kl_divergence = gaussianKL(mu_u_prior, cov_u_prior, mu_u, cov_u, mask)

        mu_pxgz, cov_pxgz = self.p_x_g_z_u(z_q_samples, mu_u)
        mu_pagz, cov_pagz = self.p_a_g_z_u(z_q_samples, mu_u)
        mu_prgza, cov_prgza = self.p_r_g_z_a_u(z_q_samples, a_seq, mu_u)

        mu_qagx, cov_qagx = self.q_a_g_x(x_seq)
        mu_qrgxa, cov_qrgxa = self.q_r_g_x_a(x_seq, a_seq)

        nll_pxgz = gaussianNLL(x_seq, mu_pxgz, cov_pxgz, mask)
        nll_pagz = gaussianNLL(a_seq, mu_pagz, cov_pagz, mask)
        nll_prgza = gaussianNLL(r_seq, mu_prgza, cov_prgza, mask)
        nll_qagx = gaussianNLL(a_seq, mu_qagx, cov_qagx, mask)
        nll_qrgxa = gaussianNLL(r_seq, mu_qrgxa, cov_qrgxa, mask)

        nll = nll_pxgz + nll_pagz + nll_prgza + anneal * kl_divergence + nll_qagx + nll_qrgxa + u_kl_divergence
        return nll

    def update_u(self, x_prev, a_prev, r_prev):
        self.u_x_list.append(x_prev)
        self.u_a_list.append(a_prev)
        self.u_r_list.append(r_prev)
        x_seq = tf.transpose(tf.convert_to_tensor(self.u_x_list), [1, 0, 2])
        a_seq = tf.transpose(tf.convert_to_tensor(self.u_a_list), [1, 0, 2])
        r_seq = tf.transpose(tf.convert_to_tensor(self.u_r_list), [1, 0, 2])
        self.u, _ = self.q_u_g_x_a_r(x_seq, a_seq, r_seq)

    def clear_u(self):
        self.u_x_list = []
        self.u_a_list = []
        self.u_r_list = []

    def gen_st_approx(self, prev, current):
        z_prev = prev[0]
        x_prev, _ = self.p_x_g_z_u(z_prev, self.u)
        a_prev = 2.*(2.*tf.random.uniform((self.opts['batch_size'], self.opts['a_dim']), 0., 1., dtype=tf.float32)-1)
        r_prev, _ = self.p_r_g_z_a_u(z_prev, a_prev, self.u)
        z_current_mu, _ = self.p_z_g_z_a(z_prev, a_prev)
        self.update_u(x_prev, a_prev, r_prev)
        return z_current_mu, x_prev, a_prev, r_prev

    def gen_xar_seq_g_z(self, z_0):
        z_0_shape = z_0.shape
        if len(z_0_shape) > 2:
            z_0 = tf.reshape(z_0, [z_0_shape[0], z_0_shape[2]])
        output_xar = tf.scan(
            self.gen_st_approx,
            tf.range(self.opts['nsteps']),
            initializer=(z_0, tf.zeros([self.opts['batch_size'], self.opts['x_dim']]),
                         tf.zeros([self.opts['batch_size'], self.opts['a_dim']]),
                         tf.zeros([self.opts['batch_size'], self.opts['r_dim']]))
        )
        self.clear_u()
        return tf.transpose(output_xar[1], [1, 0, 2])

    def recons_xar_seq_g_xar_seq(self, x_seq, a_seq, r_seq, mask):
        z_q, mu_q, cov_q = self.q_z_g_z_x_a_r(x_seq, a_seq, r_seq, mask)
        eps = tf.random.normal((self.opts['batch_size'], self.opts['nsteps'], self.opts['z_dim']),
                               0., 1., dtype=tf.float32)
        z_q_samples = mu_q + tf.multiply(eps, tf.sqrt(1e-8 + cov_q))
        mu_u, cov_u = self.q_u_g_x_a_r(x_seq, a_seq, r_seq, mask)
        mu_pxgz, cov_pxgz = self.p_x_g_z_u(z_q_samples, mu_u)
        mu_pagz, cov_pagz = self.p_a_g_z_u(z_q_samples, mu_u)
        mu_prgza, cov_prgza = self.p_r_g_z_a_u(z_q_samples, a_seq, mu_u)
        return mu_pxgz, mu_pagz, mu_prgza

    def gen_z_g_x(self, x):
        a, _ = self.q_a_g_x(x)
        r, _ = self.q_r_g_x_a(x, a)
        self.u, _ = self.q_u_g_x_a_r(x, a, r)
        _, z, _ = self.q_z_g_z_x_a_r(x, a, r)
        return z

    def train_model(self, data):
        batch_num = data.train_num // self.opts['batch_size']
        counter = self.opts['counter_start']
        for epoch in range(self.opts['epoch_start'], self.opts['epoch_start'] + self.opts['epoch_num']):
            ids_perm = np.random.permutation(data.train_num)
            for itr in range(batch_num):
                batch_ids = ids_perm[self.opts['batch_size'] * itr:self.opts['batch_size'] * (itr + 1)]
                x_batch = data.x_train[batch_ids]
                a_batch = data.a_train[batch_ids]
                r_batch = data.r_train[batch_ids]
                u_batch = data.rich_train[batch_ids]
                mask_batch = data.mask_train[batch_ids]
                loss = self.model.train_on_batch([x_batch, a_batch, r_batch, u_batch, mask_batch], np.zeros((self.opts['batch_size'], 1)))
                print(f'Epoch: {epoch}, Iter: {itr}, Loss: {loss}')
                counter += 1

        self.model.save_weights(os.path.join(self.opts['work_dir'], 'model_checkpoints', 'model_decon.h5'))