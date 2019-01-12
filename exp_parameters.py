import time
import numpy as np
from keras import backend as K


class DecayParam():
    def __init__(self, x0, x_min, half_life, name=None):

        self.x0 = x0
        self.x_min = x_min
        self.half_life = half_life
        self.anneal_rate = np.log(2)/self.half_life
        self.value = K.variable(x_min, name=name)

    def update_param(self, step):
        K.set_value(self.value, np.max([self.x_min, self.x0 * np.exp(-self.anneal_rate * step)]))


class ExpParam:
    def __init__(self,
                 lat_type,
                 latent,
                 dataset,
                 raw_type,
                 raw_dim,  # the raw data
                 net_dim,  # the input to the newtork
                 name_prefix='',
                 rec_loss_multiplier=0.,
                 g_std=0,
                 learning_rate=0.001,
                 batch_size=64,
                 valid_inter=100,
                 max_epoch=0,
                 max_example=0
                 ):

        self.created = str(int(time.time()))
        valid_lat_type = ["continuous", "discrete"]
        assert lat_type in valid_lat_type, 'lat_type, ' + str(lat_type) + ' not understood.'
        self.lat_type = lat_type
        self.latent = latent

        valid_datasets = ['mnist', 'breakout']
        assert dataset in valid_datasets, 'dataset, ' + str(dataset) + ' not understood.'
        self.dataset = dataset
        self.raw_type = raw_type
        self.raw_dim = raw_dim
        self.net_dim = net_dim
        # self.data_dim = data_dim if data_dim is not None else input_dim

        self.name_prefix = name_prefix

        self.learning_rate = learning_rate
        self.rec_loss_multiplier = rec_loss_multiplier
        self.g_size = int(np.ceil(g_std)*3)
        self.g_std = g_std if g_std > 0 else 1

        self.valid_inter = valid_inter
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.max_example = max_example

    def toString(self):
        out = self.dataset + '_' + self.lat_type

        out += self.name_prefix

        out += '_LAT'
        if self.lat_type == 'discrete':
            for dim in self.latent:
                out += str(dim[0]) + '(' + str(dim[1]) + ')'
        elif self.lat_type == 'continuous':
            for dim in self.latent:
                out = out + str(dim)

        out += '_MADE' + self.created

        return out

    def copy(self):
        # TODO!
        raise NotImplementedError

    def save(self):
        # TODO!
        raise NotImplementedError

    def load(self):
        # TODO!
        raise NotImplementedError

    def print(self):
        # TODO: Actually print all the params in a reasonable way
        print(self.toString())


