import numpy as np
import os
import tensorflow as tf
# import gym
# import matplotlib.pyplot as plt
# from IPython.display import clear_output
#
# import gym_utils
# from gen_raw_data import generate_action

import data_utils
from exp_parameters import ExpParam
from train_vision import create_or_load_vae


def gen_latent_batch(sess, network, exp_param, postfix):
    file_name = 'Breakout_raw_'+postfix+'_'
    i = 0
    while True:
        load_data_path = './data/' + file_name + '{:04}'.format(i) + '.h5'
        if os.path.isfile(load_data_path):
            print('Loading:', load_data_path, end='. ')


            ### Load data
            obs, obs_mask, action, reward, done = data_utils.load_h5_as_list(load_data_path)
            print('size =', obs.shape[0])


            ### Encode data
            print('\tEncoding...')
            z, [logit, q_y] = sess.run([network.z, network.latent_var], feed_dict={
                network.raw_input: obs,
                network.mask_in: obs_mask,
                network.is_training: False
            })

            if z.shape[-1] == 8192:
                z = np.reshape(z, [-1, 4096, 2])[:, :, -1]


            ### Save data
            new_train_data = [z, np.zeros(z.shape), action, reward, done]
            save_file_name = 'Breakout_latent_{}_{:04d}'.format(postfix, i)
            data_utils.save_lists_as_h5(save_file_name, new_train_data)
            print('\tSaved:', save_file_name)
            print()

        else:
            break
        i += 1

if __name__ == '__main__':
    ### Settings
    model_path = 'saved_model/'
    model_name = 'breakout_discrete_BLM1_STD0_LAT4096(2)_MADE1544445274'

    latent = [[32*128, 2]]
    # raw_dim = (210, 160, 3)
    # net_dim = (32*4, 32*3, 3)
    raw_dim = (84, 84, 4)
    net_dim = (84, 84, 4)

    ### Do stuff
    exp_param = ExpParam(
        lat_type="discrete",
        dataset='breakout',
        latent=[[32 * 128, 2]],
        raw_type=tf.uint8,
        raw_dim=raw_dim,
        net_dim=net_dim,  # very close to org aspect ration
        batch_size=2,  # for testing
    )

    ### Load model
    model_path += model_name

    sess, network, _ = create_or_load_vae(model_path, exp_param=exp_param, critical_load=True)

    print('\n\n\nBegin encoding.')
    gen_latent_batch(sess, network, exp_param, 'train')
    gen_latent_batch(sess, network, exp_param, 'valid')

    sess.close()
    print('done')

    file_name = 'Breakout_latent_train_'
    data_utils.load_h5_as_array(file_name, )

