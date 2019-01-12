# python gen_raw_data.py --config A2C/config/breakout.json

import pickle

import numpy as np
import multiprocessing as mp
import gym
import gym.spaces #TODO: remove this (only used to suppress warning


import scipy.ndimage
import matplotlib.pyplot as plt

import gym_utils
import data_utils
import tensorflow as tf

from A2C import train
from A2C.ActorCritic import A2C
from A2C.train import Trainer
from A2C.models.model import Model
from A2C.utils.utils import parse_args, create_experiment_dirs


def __observation_update(new_observation, old_observation):
    # Do frame-stacking here instead of the FrameStack wrapper to reduce IPC overhead
    updated_observation = np.roll(old_observation, shift=-1, axis=3)
    updated_observation[:, :, :, -1] = new_observation[:, :, :, 0]
    return updated_observation

def generate_action(env):
    random_action = True
    if random_action:
        a = env.action_space.sample()
    # else:
    # actions, values, states = self.model.step_policy.step(observation_s, states, dones)
    # genereate action using only the testing phase of A2C

    return a

def plot_4(observation_s):
    plt.figure()
    plt.imshow(observation_s[0,:,:,0])
    plt.figure()
    plt.imshow(observation_s[0,:,:,1])
    plt.figure()
    plt.imshow(observation_s[0,:,:,2])
    plt.figure()
    plt.imshow(observation_s[0,:,:,3])
    plt.show()


def gen_data(gen_args, render=False):
    """ Format: (obs, action, reward, done)
    """
    batch_num, postfix, max_steps, frame_skip = gen_args
    file_name = 'Breakout_raw_{}_{:04d}'.format(postfix, batch_num)

    # env = gym.make("BreakoutNoFrameskip-v4")
    # obs_data = []
    observation_list = []
    obs_mask_list = []
    actions_list = []
    values_list = []
    dones_list = []

    # configuration set-up
    config_args = parse_args()

    tf.reset_default_graph()

    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=config_args.num_envs,
                            inter_op_parallelism_threads=config_args.num_envs)

    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # args = config_args
    # model = Model(sess, optimizer_params={'learning_rate': args.learning_rate, 'alpha': 0.99, 'epsilon': 1e-5}, args=args)
    # a2c2 = Trainer(sess, model, args)
    # a2c2._init_model()
    # a2c2._load_model()
    #
    # states = a2c2.model.step_policy.initial_state
    #
    # dones = [False for _ in range(env.num_envs)]
    #
    # observation_s = np.zeros(
    #     (env.num_envs, a2c2.model.img_height, a2c2.model.img_width,
    #      a2c2.model.num_classes * a2c2.model.num_stack),
    #     dtype=np.uint8)
    # observation_s = __observation_update(env.reset(), observation_s)

    # Prepare Directories
    config_args.experiment_dir, config_args.summary_dir, config_args.checkpoint_dir, config_args.output_dir, config_args.test_dir = \
        create_experiment_dirs(config_args.experiment_dir)

    a2c = A2C(sess, config_args)

    # testing
    with open(a2c.args.experiment_dir + a2c.args.env_name + '.pkl', 'rb') as f:
        observation_space_shape, action_space_n = pickle.load(f)

    env = a2c.make_all_environments(num_envs=1, env_class=a2c.env_class, env_name=a2c.args.env_name, seed=a2c.args.env_seed)

    a2c.model.build(observation_space_shape, action_space_n)

    a2c.trainer._init_model()
    a2c.trainer._load_model()

    states = a2c.trainer.model.step_policy.initial_state

    dones = [False for _ in range(env.num_envs)]

    observation_s = np.zeros(
        (env.num_envs, a2c.trainer.model.img_height, a2c.trainer.model.img_width,
         a2c.trainer.model.num_classes * a2c.trainer.model.num_stack),
        dtype=np.uint8)
    observation_s = __observation_update(env.reset(), observation_s)
    mask_s = np.zeros_like(observation_s)

    i = 0
    while len(observation_list) < max_steps:

        actions, values, states = a2c.model.step_policy.step(observation_s, states, dones)
        observation, rewards, dones, _ = env.step(actions)
        for n, done in enumerate(dones):
            if done:
                observation_s[n] *= 0
                print(file_name, i, len(observation_list), max_steps, end='\r')
                # print(batch_num, len(observation_list), max_steps)

        # obs_mask = obs.astype(int) - obs
        obs_mask = observation.astype(int) - observation_s[:,:,:,-1,None]
        # obs_mask = observation_s.astype(int) - observation_s_new
        # obs_mask = np.abs(obs_mask)
        # obs_mask = np.expand_dims(obs_mask,-1)
        obs_mask = obs_mask * (obs_mask > 0)
        # obs_mask = np.mean(obs_mask, -1, keepdims=True).astype(np.uint8)
        # obs_mask = obs_mask / 255.
        # obs_mask = scipy.ndimage.filters.gaussian_filter(obs_mask, 5)
        # plt.imshow(obs_mask[:, :, 0]); plt.figure(); plt.imshow(obs); plt.show()
        # plt.imshow(obs_mask[0,:, :, 0]); plt.figure(); plt.imshow(observation[0,:,:,0]); plt.show()


        observation_s = __observation_update(observation, observation_s)
        mask_s = __observation_update(obs_mask, mask_s)
        # action = generate_action(env)
        # obs_, reward, done, info = env.step(action)


        if i % frame_skip == 0:
            # obs_data.append((obs, obs_mask, action, reward, done))
            # obs_data.append((observation, obs_mask, actions, values, dones))
            observation_list.append(observation_s)
            obs_mask_list.append(mask_s)
            actions_list.append(actions)
            values_list.append(values)
            dones_list.append(dones)

            if render: env.render()

        if dones:
            # obs, reward_sum, done = gym_utils.reset_env(env)
            # obs = data_utils.normalize_observation(obs)
            pass
        # else:
            # obs = data_utils.normalize_observation(observation_s)
            # observation_s = observation_s_new
        i += 1
    print()

    # data_as_array = np.concatenate(obs_data, 0)
    # data_as_array = np.vstack(obs_data)
    data_as_lists = [
        np.vstack(observation_list),
        np.vstack(obs_mask_list),
        np.asarray(actions_list),
        np.asarray(values_list),
        np.asarray(dones_list)]

    ### Compute memory usage of obs
    # size_of_data = data_utils.getSize_lol(obs_data)
    # actual_total_obs = len(obs_data[0]) * len(obs_data)
    # size_per_obs = int(size_of_data/actual_total_obs)
    # print(data_utils.sizeof_fmt(size_per_obs))  # 24.6KiB
    # print(size_per_obs)  # 25218 Bytes

    # data_utils.save_np_array_as_h5(file_name, data_as_array)
    data_utils.save_lists_as_h5(file_name, data_as_lists)

    # print('Generated dataset with ', data_as_array.shape[0], "observations.")
    # print("Format: (obs, obs_mask, action, reward, done)")
    print('Saved batch: {:4}'.format(batch_num), '-', file_name)

    env.close()
    # return obs_data
    return file_name


def generate_raw_data(total_frames, postfix='', frame_skip=1):
    total_frames = int(total_frames)

    # 256*32*obs_mem_size ~ 0.75 GB
    max_eps_len = 512  # doesn't actually matter - just max file size thing
    max_frames_per_thread = max_eps_len*32
    num_batches = total_frames // max_frames_per_thread
    frames_in_last_batch = total_frames - max_frames_per_thread * (total_frames // max_frames_per_thread)
    batch_len = [max_frames_per_thread]*num_batches + [frames_in_last_batch]
    if frames_in_last_batch != 0:
        num_batches += 1
    gen_args = [(i, postfix, batch_len[i], frame_skip) for i in range(num_batches)]

    num_threads = mp.cpu_count()-1
    # num_threads = 1
    print("Generating", postfix, "data for env CarRacing-v0")
    print("total_frames: ", total_frames)
    print('max_frames_per_thread', max_frames_per_thread)
    print("num_batches:", num_batches, '(',batch_len,')')
    print("num_threads:", num_threads)
    print('...')

    # with mp.Pool(num_threads) as p:
    #     # data = p.map(gen_data, gen_args)
    #     file_names = p.map(gen_data, gen_args)

    for gen_arg in gen_args:
        file_names = gen_data(gen_arg, render=False)

    return [file_names]


if __name__ == '__main__':
    # python gen_raw_data.py --config A2C/config/breakout.json

    print(__name__, 'begin')
    frame_skip = 10

    # TODO: This causes memory issues!
    train_frames = 5e4
    # train_frames = 1e2
    # train_frames = 1e2
    file_names = generate_raw_data(train_frames, 'train_5', frame_skip)
    print('\n'*2)

    valid_frames = 1e2
    valid_frames = 1e4
    # file_names = generate_raw_data(valid_frames, 'valid')

    if 1:
        print("Load test - Begin.")
        file_name = file_names[0]
        data_path = './data/' + file_name + '.h5'
        print(data_path)
        data = data_utils.load_h5_as_list(data_path)
        print('data', type(data))
        print('data[0]', type(data[0]))
        # print(data[0])
        print("Load test - Success!")






