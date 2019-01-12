import h5py
import numpy as np
import sys
import os

import tensorflow as tf


ball_col = [200, 72, 72]  # Color of the ball in Breakout

def data_iterator_mnist(data, batch_size):
    N = data.shape[0]
    epoch = 0
    while True:
        epoch += 1
        if batch_size == -1:
            yield None, N, data
            
        np.random.shuffle(data)
        for i in range(int(N/batch_size)):
            yield epoch, i*batch_size, data[i*batch_size:(i+1)*batch_size]


def data_iterator_atari(data, batch_size, shuffle=True):
    def shuffle_data(obs, obs_mask, action, reward, done):
        ## https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
        # p = np.random.permutation(obs.shape[0])
        # return obs[p], obs_mask[p], action[p], reward[p], done[p]
        rng_state = np.random.get_state()
        np.random.shuffle(obs)
        np.random.set_state(rng_state)
        np.random.shuffle(obs_mask)
        np.random.set_state(rng_state)
        np.random.shuffle(action)
        np.random.set_state(rng_state)
        np.random.shuffle(reward)
        np.random.set_state(rng_state)
        np.random.shuffle(done)
        np.random.set_state(rng_state)
        return obs, obs_mask, action, reward, done

    obs, obs_mask, action, reward, done = data
    N = obs.shape[0]
    epoch = 0
    while True:
        epoch += 1
        if batch_size == -1:
            yield None, N, (obs, obs_mask, action, reward, done)

        if shuffle:
            obs, obs_mask, action, reward, done = shuffle_data(obs, obs_mask, action, reward, done)
        for i in range(int(N / batch_size)):
            out_data = (
                obs[i * batch_size:(i + 1) * batch_size],
                obs_mask[i * batch_size:(i + 1) * batch_size],
                action[i * batch_size:(i + 1) * batch_size],
                reward[i * batch_size:(i + 1) * batch_size],
                done[i * batch_size:(i + 1) * batch_size],
            )
            yield epoch, i * batch_size, out_data


def load_data(train_batch_size, dataset='mnist', test_batch_size=64, shuffle=True):  #TODO: test_batch_size should be handled properly (=-1)
    if dataset == 'mnist':
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = np.expand_dims(x_train/255., -1)
        x_test = np.expand_dims(x_test/255., -1)

        train_iter = data_iterator_mnist(x_train, batch_size=train_batch_size)
        test_iter = data_iterator_mnist(x_test, batch_size=test_batch_size)

        return train_iter, test_iter
    elif dataset == 'breakout':
        # TODO: This probably causes meomry issues
        x_train = load_h5_as_array('Breakout_raw_train_')
        train_iter = data_iterator_atari(x_train, batch_size=train_batch_size, shuffle=shuffle)
        print('Train set loaded complete.', x_train[0].shape[0], 'data points in total.')
        print()

        x_test = load_h5_as_array('Breakout_raw_valid_')
        test_iter = data_iterator_atari(x_test, batch_size=test_batch_size, shuffle=False)
        print('Valid set loaded complete', x_test[0].shape[0], 'data points in total.')
        print()

        return train_iter, test_iter
    else:
        print(dataset)
        raise NotImplementedError


def normalize_observation(observation):
    # obs = np.copy(observation)/255. # uses a lot more space!
    obs = np.copy(observation)
    return obs


def save_np_array_as_h5(file_name, data_as_array):
    # print("Format: (obs, action, reward, done)")
    data_path = './data/'+file_name+'.h5'
    # print("Saving dataset at: {}".format(data_path), end=' ... ')

    h5f = h5py.File(data_path, 'w')
    h5f.create_dataset('obs',    data=np.array([i for i in data_as_array[:, 0]]))
    h5f.create_dataset('obs_mask',    data=np.array([i for i in data_as_array[:, 1]]))
    h5f.create_dataset('action', data=data_as_array[:, 2].astype(int))
    h5f.create_dataset('reward', data=data_as_array[:, 3].astype(float))
    h5f.create_dataset('done',   data=data_as_array[:, 4].astype(int))
    h5f.close()


def save_lists_as_h5(file_name, data_as_lists):
    # print("Format: (obs, action, reward, done)")
    data_path = './data/'+file_name+'.h5'

    obs, mask, action, reward, done = data_as_lists

    h5f = h5py.File(data_path, 'w')
    h5f.create_dataset('obs',         data=obs)
    h5f.create_dataset('obs_mask',    data=mask)
    h5f.create_dataset('action',      data=action.astype(int))
    h5f.create_dataset('reward',      data=reward.astype(float))
    h5f.create_dataset('done',        data=done.astype(int))
    h5f.close()


def load_h5_as_list(data_path):
    h5f = h5py.File(data_path, 'r')
    # data = {}
    # data['obs']    = h5f['obs'][:]      # float
    # data['action'] = h5f['action'][:]   # int
    # data['reward'] = h5f['reward'][:]   # float
    # data['done']   = h5f['done'][:]     # int
    data = [
        h5f['obs'][:],
        h5f['obs_mask'][:],
        h5f['action'][:],   # int
        h5f['reward'][:],   # float
        h5f['done'][:],     # int
        ]
    h5f.close()

    return data


def load_h5_as_array(file_name, num_chars=4):
    data = []
    i = 0
    while True:
        data_path = './data/' + file_name + '{:04}'.format(i) + '.h5'

        if os.path.isfile(data_path):
            data.append(load_h5_as_list(data_path))
        else:
            break
        print('Loaded:', data_path)
        i += 1

    # print("Format: (obs, action, reward, done)")
    obs = np.vstack([data[i][0] for i in range(len(data))])
    obs_mask = np.vstack([data[i][1] for i in range(len(data))])
    action = np.concatenate([data[i][2] for i in range(len(data))])
    reward = np.concatenate([data[i][3] for i in range(len(data))])
    done = np.concatenate([data[i][4] for i in range(len(data))])

    return obs, obs_mask, action, reward, done

def getSize_lol(lol):
    """ Get size from list of list of objects"""
    size = 0
    # for ll in lol:  # runs
    #     for l in ll:  # observations
    for l in lol:  # observations
        for t in l:  # individual elements
                if type(t) is np.ndarray:
                    size += t.nbytes
                else:
                    size += sys.getsizeof(t)

    return size


def sizeof_fmt(num, suffix='B'):
    """ From: https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size/1094933#1094933"""
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def find_unique_colors(img):
    # https://stackoverflow.com/questions/24780697/numpy-unique-list-of-colors-in-the-image
    img_r = img.reshape(-1, img.shape[-1])
    unique_col = np.unique(img_r, axis=0)
    return unique_col


def mask_col(img, col, mul, flat_output=False):
    if len(img.shape) == 3:
        img = np.expand_dims(img, 0)

    out = []
    for i in range(img.shape[0]):
        o = np.zeros_like(img[0])
        o[np.where((img[i, :, :, 0] == col[0]) & (img[i, :, :, 1] == col[1]) & (img[i, :, :, 2] == col[2]))] = mul
        if flat_output and len(o.shape) == 3:
            o = o[:, :, 0]
        out.append(o)

    return np.array(out).astype(np.float32)
