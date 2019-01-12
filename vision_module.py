import tensorflow as tf
import numpy as np
import os

from keras import backend as K
from keras.activations import softmax

from exp_parameters import DecayParam

from A2C.layers import orthogonal_initializer, noise_and_argmax
from A2C.layers import conv2d, flatten, dense

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def img_summary(name, img, max_outputs, type='simple'):
    def concat4(img1, img2, img3, img4):
        sum_img_top = tf.concat([img1, img2], 2)
        sum_img_bot = tf.concat([img3, img4], 2)
        sum_img = tf.concat([sum_img_top, sum_img_bot], 1)
        return sum_img

    if type == 'simple':
        return img
    elif type == 'ch4':
        img1 = img[:, :, :, 0, None]
        img2 = img[:, :, :, 1, None]
        img3 = img[:, :, :, 2, None]
        img4 = img[:, :, :, 3, None]

    elif type == 'list':
        img1, img2, img3, img4 = img
    else:
        raise Exception

    sum_img = concat4(img1, img2, img3, img4)
    tf.summary.image(name, sum_img, max_outputs)

    return sum_img


class BaseAutoEncoder(object):
    # Create model
    def __init__(self, exp_param):
        self.exp_param = exp_param
        self.dataset = exp_param.dataset
        self.latent_dim = exp_param.latent
        self.tb_num_images = 3

        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.img_channels = exp_param.net_dim[-1]

        self.raw_input, self.image, self.mask_in, self.mask_net, self.z_input = self.create_net_input()
        # tf.summary.image('image', self.image, self.tb_num_images)
        img_summary('image', self.image, self.tb_num_images, type='ch4')
        # tf.summary.image('mask_net', self.mask_net, self.tb_num_images)
        img_summary('mask_net', self.mask_net, self.tb_num_images, type='ch4')

    def create_net_input(self):
        # tf.placeholder(tf.float32, (None,) + exp_param.data_dim, name='image')
        # self.image = tf.placeholder(tf.float32, (None,)+exp_param.input_dim, name='image')
        raw_input = tf.placeholder(self.exp_param.raw_type, (None,) + self.exp_param.raw_dim, name='raw_input')

        net_input = tf.image.resize_images(
            raw_input,
            size=self.exp_param.net_dim[:2],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        net_input = tf.cast(net_input, tf.float32)
        if self.dataset == 'breakout':
            net_input = tf.div(net_input, 255., 'normalize')

        # mask_in = tf.placeholder(tf.uint8, (None,) + self.exp_param.raw_dim[:2] + (1,), 'Rec_loss_mask')
        mask_in = tf.placeholder(tf.uint8, (None,) + self.exp_param.raw_dim, 'Rec_loss_mask')
        mask_net = tf.image.resize_images(
            mask_in,
            size=self.exp_param.net_dim[:2],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        mask_net = tf.cast(mask_net, tf.float32)/255.
        if self.exp_param.g_size != 0:
            g_kernel = self.gaussian_kernel(size=self.exp_param.g_size, mean=0, std=self.exp_param.g_std)
            mask_net = tf.reduce_mean(mask_net, -1, keepdims=True)  # TODO: Very unsatisfying solution!
            mask_net = tf.nn.conv2d(mask_net, g_kernel, strides=[1, 1, 1, 1], padding="SAME")
        mask_net = mask_net * self.exp_param.g_std / 0.3989  # https://stats.stackexchange.com/questions/143631/height-of-a-normal-distribution-curve

        z_dim = np.prod(self.latent_dim[0])  # Number variables, values per variable
        z_input = tf.placeholder(tf.float32, shape=(None, z_dim))

        return raw_input, net_input, mask_in, mask_net, z_input

    def gaussian_kernel(self,
                        size: int,
                        mean: float,
                        std: float,):
        """ Makes 2D gaussian Kernel for convolution.
            https://stackoverflow.com/questions/52012657/how-to-make-a-2d-gaussian-filter-in-tensorflow
        """
        d = tf.distributions.Normal(float(mean), float(std))
        vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
        gauss_kernel = tf.einsum('i,j->ij', vals, vals)

        gauss_kernel = (gauss_kernel / tf.reduce_sum(gauss_kernel))[:, :, tf.newaxis, tf.newaxis]
        # gauss_kernel = tf.tile(gauss_kernel, [1, 1, 1, self.exp_param.net_dim[-1]])

        return gauss_kernel

    def setup_network(self):
        self.encoder_out = self.encoder(self.image)
        self.z, self.latent_var = self.latent(self.encoder_out)

        # self.reconstructions = self.decoder(self.z)
        with tf.variable_scope('decoder'):
            self.reconstructions = self.decoder(self.z)
        with tf.variable_scope('decoder', reuse=True):
            tf.get_variable_scope().reuse_variables()  # TODO: Why is this necessary? It shouldn't be, but it is...?
            self.reconstructions_from_z = self.decoder(self.z_input)

        # for x in tf.global_variables(): print(x.name)  # Debugging

        # tf.summary.image('reconstructions', self.reconstructions, self.tb_num_images)
        img_summary('reconstructions', self.reconstructions, self.tb_num_images, type='ch4')


        self.loss, self.loss_img = self.compute_loss()

        loss_img_3ch = self.loss_img/(tf.reduce_max(self.loss_img)+1e-9)

        mask_norm = self.mask_net

        if self.exp_param.dataset == 'breakout':
            # mask_norm = tf.tile(mask_norm, [1, 1, 1, 3])
            # loss_img_3ch = tf.tile(loss_img_3ch, [1, 1, 1, 3])
            img_reshape = tf.reduce_mean(self.image, -1, keepdims=True)
            rec_reshape = tf.reduce_mean(self.reconstructions, -1, keepdims=True)
            mask_norm = tf.reduce_mean(mask_norm, -1, keepdims=True)
        else:
            img_reshape = self.image
            rec_reshape = self.reconstructions

        mask_norm = mask_norm/(tf.reduce_max(mask_norm)+1e-9)

        self.sum_img = img_summary('awesome_summary', [img_reshape, rec_reshape, mask_norm, loss_img_3ch], self.tb_num_images, 'list')

        self.merged = tf.summary.merge_all()

    def encoder(self, x):
        print('Encoder')
        print(x)
        if self.dataset == 'mnist':
            # x = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=2, padding='valid', activation=tf.nn.relu)
            # print(x)
            # x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=2, padding='valid', activation=tf.nn.relu)
            # print(x)
            # x = tf.layers.conv2d(x, filters=128, kernel_size=3, strides=2, padding='valid', activation=tf.nn.relu)
            # print(x)
            # x = tf.layers.conv2d(x, filters=128, kernel_size=2, strides=1, padding='valid', activation=tf.nn.relu)
            # print(x)
            # x = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=2, padding='valid', activation=tf.nn.relu)
            # print(x)
            x = conv2d('conv1', x, num_filters=32, kernel_size=(8, 8),
                           padding='VALID', stride=(4, 4),
                           initializer=orthogonal_initializer(np.sqrt(2)), activation=tf.nn.relu,
                           is_training=self.is_training)
            print(x)

            x = conv2d('conv2', x, num_filters=64, kernel_size=(4, 4), padding='VALID', stride=(2, 2),
                           initializer=orthogonal_initializer(np.sqrt(2)), activation=tf.nn.relu,
                           is_training=self.is_training)
            print(x)
            x = conv2d('conv3', x, num_filters=64, kernel_size=(2,2), padding='VALID', stride=(1, 1),
                           initializer=orthogonal_initializer(np.sqrt(2)), activation=tf.nn.relu,
                           is_training=self.is_training)
            print(x)
            # x = tf.layers.conv2d(x, filters=128, kernel_size=2, strides=1, padding='valid', activation=tf.nn.relu)
            # print(x)

        # elif self.dataset == 'breakout':
        #     ## OLD VERSION THAT IS VERY BIG!
        #     x = tf.layers.conv2d(x, filters=16, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        #     print(x)
        #     x = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        #     print(x)
        #     x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        #     print(x)
        #     x = tf.layers.conv2d(x, filters=128, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        #     print(x)
        #     x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        #     print(x)
        elif self.dataset == 'breakout':
            conv1 = conv2d('conv1', x, num_filters=32, kernel_size=(8, 8),
                           padding='VALID', stride=(4, 4),
                           initializer=orthogonal_initializer(np.sqrt(2)), activation=tf.nn.relu,
                           is_training=self.is_training)
            print(conv1)

            conv2 = conv2d('conv2', conv1, num_filters=64, kernel_size=(4, 4), padding='VALID', stride=(2, 2),
                           initializer=orthogonal_initializer(np.sqrt(2)), activation=tf.nn.relu,
                           is_training=self.is_training)
            print(conv2)

            conv3 = conv2d('conv3', conv2, num_filters=64, kernel_size=(3, 3), padding='VALID', stride=(1, 1),
                           initializer=orthogonal_initializer(np.sqrt(2)), activation=tf.nn.relu,
                           is_training=self.is_training)
            print(conv3)
            x = conv3
        print()
        return x

    def decoder(self, z, reuse=False):
        print('Decoder')
        # first_conv_filters = 256
        first_conv_filters = 64
        decoder_input_size = self.encoder_out.shape[1]*self.encoder_out.shape[2]*first_conv_filters

        x = tf.layers.dense(z, decoder_input_size, activation=tf.nn.relu)
        print(x)
        # x = tf.reshape(x, [-1, 1, 1, decoder_input_size])
        x = tf.reshape(x, [
            -1,
            self.encoder_out.shape[1],
            self.encoder_out.shape[2],
            first_conv_filters
        ])
        # x = tf.layers.conv2d(x, filters=128, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        print(x)


        if self.dataset == 'mnist':
            x = tf.image.resize_images(x, (x.shape[1] * 6, x.shape[2] * 6))
            x = conv2d('conv_up1', x, num_filters=64, kernel_size=(3, 3), padding='VALID', stride=(1, 1),
                           initializer=orthogonal_initializer(np.sqrt(2)), activation=tf.nn.relu,
                           is_training=self.is_training)
            print(x)

            x = tf.image.resize_images(x, (x.shape[1] * 3, x.shape[2] * 3))
            x = conv2d('conv_up2', x, num_filters=64, kernel_size=(3, 3), padding='VALID', stride=(1, 1),
                           initializer=orthogonal_initializer(np.sqrt(2)), activation=tf.nn.relu,
                           is_training=self.is_training)
            print(x)

            x = tf.image.resize_images(x, (x.shape[1] * 3, x.shape[2] * 3))
            x = conv2d('conv_up3', x, num_filters=32, kernel_size=(3, 3), padding='VALID', stride=(1, 1),
                           initializer=orthogonal_initializer(np.sqrt(2)), activation=tf.nn.relu,
                           is_training=self.is_training)
            print(x)

            x = conv2d('conv_up4', x, num_filters=self.img_channels, kernel_size=(1, 1), padding='VALID', stride=(1, 1),
                           initializer=orthogonal_initializer(np.sqrt(2)), activation=None,
                           is_training=self.is_training)
            print(x)
        elif self.dataset == 'breakout':
            x = tf.image.resize_images(x, (x.shape[1] * 4, x.shape[2] * 4))
            # x = tf.layers.conv2d(x, filters=64, kernel_size=7, strides=1, padding='valid', activation=tf.nn.relu)
            x = conv2d('conv_up1', x, num_filters=64, kernel_size=(7, 7), padding='VALID', stride=(1, 1),
                           initializer=orthogonal_initializer(np.sqrt(2)), activation=tf.nn.relu,
                           is_training=self.is_training)
            print(x)
            x = tf.image.resize_images(x, (x.shape[1] * 4, x.shape[2] * 4))
            # x = tf.layers.conv2d(x, filters=64, kernel_size=7, strides=1, padding='same', activation=tf.nn.relu)
            x = conv2d('conv_up2', x, num_filters=64, kernel_size=(7, 7), padding='SAME', stride=(1, 1),
                           initializer=orthogonal_initializer(np.sqrt(2)), activation=tf.nn.relu,
                           is_training=self.is_training)
            print(x)
            # x = tf.layers.conv2d(x, filters=32, kernel_size=5, strides=1, padding='valid', activation=tf.nn.relu)
            x = conv2d('conv_up3', x, num_filters=32, kernel_size=(5, 5), padding='VALID', stride=(1, 1),
                           initializer=orthogonal_initializer(np.sqrt(2)), activation=tf.nn.relu,
                           is_training=self.is_training)
            print(x)
            # x = tf.layers.conv2d(x, filters=self.img_channels, kernel_size=1, strides=1, padding='same', activation=None)
            x = conv2d('conv_up4', x, num_filters=self.img_channels, kernel_size=(1, 1), padding='VALID', stride=(1, 1),
                           initializer=orthogonal_initializer(np.sqrt(2)), activation=None,
                           is_training=self.is_training)
            print(x)
        print()
        return x

        if 0:
            ## OLD LARGE CRAP

            x = tf.image.resize_images(x, (x.shape[1]*2, x.shape[2]*2))
            x = tf.layers.conv2d(x, filters=128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
            # x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=2, strides=2, padding='valid', activation=tf.nn.relu)
            print(x)
            # x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=2, strides=2, padding='valid', activation=tf.nn.relu)
            x = tf.image.resize_images(x, (x.shape[1]*2, x.shape[2]*2))
            x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
            print(x)
            # x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=2, strides=2, padding='valid', activation=tf.nn.relu)
            x = tf.image.resize_images(x, (x.shape[1]*2, x.shape[2]*2))
            x = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
            print(x)
            # x = tf.layers.conv2d_transpose(x, filters=16, kernel_size=2, strides=2, padding='valid', activation=tf.nn.relu)
            x = tf.image.resize_images(x, (x.shape[1]*2, x.shape[2]*2))
            x = tf.layers.conv2d(x, filters=16, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
            print(x)

            if self.dataset == 'mnist':
                # x = tf.layers.conv2d_transpose(x, filters=8, kernel_size=2, strides=2, padding='valid', activation=tf.nn.relu)
                x = tf.image.resize_images(x, (x.shape[1]*2, x.shape[2]*2))
                x = tf.layers.conv2d(x, filters=8, kernel_size=5, strides=1, padding='valid', activation=tf.nn.relu)
                print(x)
                x = tf.layers.conv2d(x, filters=self.img_channels, kernel_size=1, strides=1, padding='valid', activation=tf.nn.relu)
            elif self.dataset == 'breakout':
                # x = tf.layers.conv2d_transpose(x, filters=8, kernel_size=2, strides=2, padding='valid', activation=tf.nn.relu)
                x = tf.image.resize_images(x, (x.shape[1]*2, x.shape[2]*2))
                x = tf.layers.conv2d(x, filters=8, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
                print(x)
                x = tf.layers.conv2d(x, filters=self.img_channels, kernel_size=1, strides=1, padding='same', activation=tf.nn.relu)
            print(x)
            print()
            return x

    def sample_z(self, *args):
        raise NotImplementedError

    def latent(self, x):
        raise NotImplementedError

    def reconstruction_loss(self):
        # logits_flat = tf.layers.flatten(self.reconstructions)
        # labels_flat = tf.layers.flatten(self.image)
        # mask = tf.layers.flatten(self.mask)
        # err = logits_flat - labels_flat
        # err = err * mask

        err = (self.reconstructions - self.image)
        err = tf.square(err)
        err_mask = err*self.mask_net
        tf.summary.scalar("train/err_mask", tf.reduce_mean(err_mask))
        tf.summary.histogram('train_C/err_mask', err_mask)

        err = err + err_mask*self.exp_param.rec_loss_multiplier

        return tf.reduce_mean(err, axis=[1, 2, 3]), tf.reduce_mean(err, axis=-1, keepdims=True)

    def KL_loss(self):
        raise NotImplementedError

    def compute_loss(self):
        raise NotImplementedError

    def update_params(self, *args, **kwargs):
        pass

    def get_embedding(self, sess, observation):
        raise NotImplementedError

    def predict(self, sess, data):
        raise NotImplementedError

    def print_summary(self):
        print()


class ContinuousAutoEncoder(BaseAutoEncoder):
    def __init__(self, *args, **kwargs):
        super(ContinuousAutoEncoder, self).__init__(*args, **kwargs)

        self.KL_boost = DecayParam(x0=0.01, x_min=0.001, half_life=5e5)  # TODO: should be handled by config
        tf.summary.scalar("hyper/KL_boost_C", tf.reduce_mean(self.KL_boost.value))

        self.setup_network()

    def sample_z(self, mu, logvar):
        eps = tf.random_normal(shape=tf.shape(mu))
        z = mu + tf.exp(logvar / 2) * eps
        return z

    def latent(self, x):
        print("Latent: Continuous")

        x = tf.layers.flatten(x)
        print(x)
        z_mu = tf.layers.dense(x, units=self.latent_dim[0], name='z_mu')
        z_logvar = tf.layers.dense(x, units=self.latent_dim[0], name='z_logvar')
        print(z_mu)
        print(z_logvar)
        z = self.sample_z(z_mu, z_logvar)
        print(z)
        print()

        std = tf.sqrt(tf.exp(z_logvar))
        tf.summary.histogram('train_C/z_mu', z_mu)
        tf.summary.histogram('train_C/z_std', std)
        tf.summary.histogram('train_C/z', z)

        return z, (z_mu, z_logvar)

    def KL_loss(self):
        z_mu, z_logvar = self.latent_var
        kl_loss = 0.5 * tf.reduce_mean(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
        return kl_loss*self.KL_boost.value

    def compute_loss(self):
        rec_loss, err_img = self.reconstruction_loss()
        kl_loss = self.KL_loss()
        vae_loss = tf.reduce_mean(rec_loss + kl_loss)

        tf.summary.scalar("train/KL_loss", tf.reduce_mean(kl_loss))
        tf.summary.scalar("train/rec_loss", tf.reduce_mean(rec_loss))
        tf.summary.scalar("train/total_loss", tf.reduce_mean(vae_loss))
        tf.summary.image('Error_image', err_img, self.tb_num_images)
        tf.summary.histogram('train_C/err_vals', err_img)
        return vae_loss, err_img

    def update_params(self, step):
        self.KL_boost.update_param(step)

    def get_embedding(self, sess, observation):
        return sess.run(self.z, feed_dict={self.image: observation[None, :, :, :]})

    def predict(self, sess, data):
        print(sess)
        z_mu, z_logvar = self.latent_var
        pred, mu, z_logvar, z = sess.run([self.reconstructions, z_mu, z_logvar, self.z], feed_dict={self.image:data})
        sigma = np.exp(z_logvar/2)
        return pred, mu, sigma, z

    def print_summary(self):
        print("KL_boost {:5.4f}".format(K.get_value(self.KL_boost.value)))


class DiscreteAutoEncoder(BaseAutoEncoder):
    def __init__(self, *args, **kwargs):
        super(DiscreteAutoEncoder, self).__init__(*args, **kwargs)

        self.tau = DecayParam(x0=5.0, x_min=0.01, half_life=7.5e5)  # TODO: should be handled by config
        tf.summary.scalar("hyper/tau", tf.reduce_mean(self.tau.value))

        self.KL_boost = DecayParam(x0=0.5, x_min=0.1, half_life=5e5)  # TODO: should be handled by config
        tf.summary.scalar("hyper/KL_boost_D", tf.reduce_mean(self.KL_boost.value))

        self.setup_network()

    def sample_z(self, q_y):
        N, M = self.latent_dim[0]  # Number variables, values per variable

        # # TODO: should it be logits or log(softmax(logits))? From the paper (Cat. reparam.) it looks like the latter!
        # U = K.random_uniform(K.shape(logits), 0, 1)
        # y = logits - K.log(-K.log(U + 1e-20) + 1e-20)  # logits + gumbel noise
        # y = K.reshape(y, (-1, self.N, self.M))

        # Gumbel softmax trick
        log_q_y = K.log(q_y + 1e-20)
        U = K.random_uniform(K.shape(log_q_y), 0, 1)
        y = log_q_y - K.log(-K.log(U + 1e-20) + 1e-20)  # log_prob + gumbel noise

        # z = K.reshape(softmax(y / self.tau), (-1, N*M))

        def gumble_softmax(y):
            z = softmax(y / self.tau.value)
            z = tf.reshape(z, (-1, N*M))
            return z

        def hardsample(log_q_y):
            log_q_y = tf.reshape(log_q_y, (-1, M))
            z = tf.multinomial(log_q_y, 1)
            z = tf.one_hot(z, M)
            z = tf.reshape(z, (-1, N*M))
            return z

        # TODO: make sure that hard sample works with differnet shapes
        z = tf.cond(
            self.is_training,
            lambda: gumble_softmax(y),
            lambda: hardsample(log_q_y)
        )
        return z

    def latent(self, x):
        print("Latent: Discrete")
        N, M = self.latent_dim[0]

        x = tf.layers.flatten(x)
        print(x)

        logits = tf.layers.dense(x, units=N*M, name='z_logits')
        logits = K.reshape(logits, (-1, N, M))
        print(logits)

        q_y = softmax(logits)
        print(q_y)

        z = self.sample_z(q_y)
        print(z)

        # TODO: remove the 0th column.
        #z=
        print()

        # tf.summary.image('logits', tf.expand_dims(logits, -1), self.tb_num_images)
        # tf.summary.image('q_y', tf.expand_dims(q_y, -1), self.tb_num_images)
        tf.summary.image('z', K.reshape(z, (-1, N, M, 1)), self.tb_num_images)

        tf.summary.histogram('train_D/logits', logits)
        tf.summary.histogram('train_D/q_y', q_y)
        tf.summary.histogram('train_D/z', z)

        return z, (logits, q_y)

    def KL_loss(self):
        N, M = self.latent_dim[0]
        _, q_y = self.latent_var

        log_q_y = K.log(q_y + 1e-20)
        kl_loss = q_y * (log_q_y - K.log(1.0 / M))
        kl_loss = tf.reduce_mean(kl_loss, axis=(1, 2))
        return kl_loss*self.KL_boost.value

    def compute_loss(self):
        rec_loss, err_img = self.reconstruction_loss()
        kl_loss = self.KL_loss()
        elbo = tf.reduce_mean(rec_loss + kl_loss)

        tf.summary.scalar("train/KL_loss", tf.reduce_mean(kl_loss))
        tf.summary.scalar("train/rec_loss", tf.reduce_mean(rec_loss))
        tf.summary.scalar("train/total_loss", tf.reduce_mean(elbo))
        tf.summary.image('Error_image', err_img, self.tb_num_images)
        tf.summary.histogram('train_D/err_vals', err_img)
        return elbo, err_img

    def update_params(self, step):
        self.tau.update_param(step)
        self.KL_boost.update_param(step)

    def get_embedding(self, sess, observation):
        raise NotImplementedError

    def predict(self, sess, data):
        raise NotImplementedError

    def print_summary(self):
        print("tau {:5.4f}".format(K.get_value(self.tau.value)),
              "- KL_boost {:5.4f}".format(K.get_value(self.KL_boost.value)))
