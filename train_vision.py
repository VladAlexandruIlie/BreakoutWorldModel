import tensorflow as tf
import numpy as np
import time

import data_utils
from exp_parameters import ExpParam
from vision_module import ContinuousAutoEncoder, DiscreteAutoEncoder

np.random.seed(int(time.time()))
tf.set_random_seed(int(time.time())+1)

# TODO: Things to consider
# * Do experiments on MNIST
# * Try making KL_boost = 0. Companre + KL should perhaps be increasing, rather than decreasing?
# * Consider more powerful decoder

# * Add weight decay / monitor weights
# * save best validation score model

# * Don't do the one-hot encoding.

# * Use numpy for scalar tracking?
# * include a 'degree of determinism' measure in tensorboard.
# * Exponential smoothing should be towards a point, not just end abruptly.
#         Parameters: What should it annealt towards? and how many steps before it is 1% away from that?

# * Tensorboard measure: difference between two random prediction - pure prior check.
#   - This can be achieved by looking at q_y


def create_or_load_vae(model_path, exp_param, critical_load=False):
    graph = tf.Graph()
    with graph.as_default():  # Original formuation
        # graph.as_default()

        config = tf.ConfigProto(
            allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        # sess = tf.Session(config=config, graph=graph) # previous
        sess = tf.InteractiveSession(config=config, graph=graph)

        if "continuous" == exp_param.lat_type:
            print("Continuous")
            network = ContinuousAutoEncoder(exp_param)
        elif 'discrete' == exp_param.lat_type:
            print("Discrete")
            network = DiscreteAutoEncoder(exp_param)
        else:
            print("Undefined", model_path)
            raise NotImplementedError

        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(max_to_keep=1, name='ae_saver')
        try:
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
            print("Model restored from: {}".format(model_path))
        except:
            print("Could not restore saved model")
            if critical_load:
                raise Exception

        return sess, network, saver


def train_vae(exp_param, experiment_name=None):
    ### GENERAL SETUP
    if experiment_name is None:
        experiment_name = exp_param.toString()
    model_path = "saved_model/" + experiment_name + "/"
    model_name = model_path + experiment_name + '_model'

    print('experiment_name: ', experiment_name)
    print('model_path: ', model_path)
    # print('model_name: ', model_name)
    # exp_param.print()
    print()

    ################## SETTINGS #####################
    valid_inter = exp_param.valid_inter
    batch_size = exp_param.batch_size
    learning_rate = exp_param.learning_rate
    data_set = exp_param.dataset

    ### DATA
    train_iter, test_iter = data_utils.load_data(batch_size, data_set)
    # ball_col = data_utils.ball_col
    # rec_loss_multiplier = exp_param.rec_loss_multiplier

    ### NETWORK
    sess, network, saver = create_or_load_vae(model_path, exp_param=exp_param)

    # TODO: load or inferr gloabl step (don't start at zero!)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    epoch = -1

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(network.loss, global_step=global_step)
    tf.global_variables_initializer().run()

    writer = tf.summary.FileWriter('logdir/train_'+experiment_name)
    writer_test = tf.summary.FileWriter('logdir/valid_'+experiment_name)
    step = global_step.eval()

    print("\nBegin Training")
    try:
        while True:
            network.update_params(step*batch_size)

            if step % (valid_inter*10) == 0 and step > 0:
                ## PERFORM TEST SET EVALUATION
                _, _, data = next(test_iter)

                if exp_param.dataset == 'breakout':
                    # TODO: handle data for agent properly!
                    images = data[0]
                    masks = data[1]  # TODO: Test that this works! (currently 'exp_param.rec_loss_multiplier' is multiplied twice'
                else:
                    images = data
                    masks = np.zeros_like(images)

                # TODO: Test should use hard sample
                [summary, test_loss] = sess.run([network.merged, network.loss], feed_dict={
                    network.raw_input: images,
                    network.mask_in: masks,
                    network.is_training: False
                })
                test_loss = np.mean(test_loss)
                writer_test.add_summary(summary, step*batch_size)

                print("Epoch {:4}, obs {:10}: Te. loss {:9.6f}".format(
                    epoch, step*batch_size, test_loss), end=' ### ')
                network.print_summary()
                print()

            ## GET TRAIN BATCH
            epoch, e_step, data = next(train_iter)
            if exp_param.dataset == 'breakout':
                # TODO: handle data for agent properly!
                images = data[0]
                masks = data[1]
                # if np.random.randint(0,2) == 0:
                #     images = data[0]
                #     masks = data[1]
                # else:
                #     images = data[0][:,::-1,:,:]
                #     masks = data[1][:,::-1,:,:]
            else:
                images = data
                masks = np.zeros_like(images)

            ## COMPUTE TRAIN SET SUMMARY
            if step % valid_inter == 0 and step > 0:
                print("Epoch {:4}, obs {:10}: Tr. loss {:9.6f}".format(
                    epoch, step*batch_size, loss_value), end=' ### ')
                network.print_summary()

                [summary] = sess.run([network.merged], feed_dict={
                    network.raw_input: images,
                    network.mask_in: masks,
                    network.is_training: True
                })
                writer.add_summary(summary, step*batch_size)
                try:
                    save_path = saver.save(sess, model_name, global_step=global_step)
                except KeyboardInterrupt:
                    break
                except:
                    print("\nFAILED TO SAVE MODEL!\n")

            ## PERFORM TRAINING STEP
            _, loss_value = sess.run([train_op, network.loss], feed_dict={
                network.raw_input: images,
                network.mask_in: masks,
                network.is_training: True
                })
            loss_value = np.mean(loss_value)

            if np.any(np.isnan(loss_value)):
                raise ValueError('loss_value is NaN')

            if epoch >= exp_param.max_epoch and 0 < exp_param.max_epoch:
                print("Max epoch reached!")
                break

            if step*batch_size >= exp_param.max_example and 0 < exp_param.max_example:
                print("Max example reached!")
                break

            step += 1

    except (KeyboardInterrupt, SystemExit):
        print("Manual Interrupt" + '\n'*5)

    # except Exception as e:
    #     print("Exception: {}".format(e))
    sess.close()


if __name__ == '__main__':
    raw_dim = (210, 160, 3)
    net_dim = (32*4, 32*3, 3)

    ## DISCRETE
    rec_loss_multipliers = [1, 2, 4, 8, 16, 32, 64, 128, 256, 0]
    g_std = [0,1,2]

    for std in g_std:
        for x in rec_loss_multipliers:
            exp_param = ExpParam(
                lat_type="discrete",
                dataset='breakout',
                name_prefix='_BLM'+str(x)+'_STD'+str(std),
                latent=[[32*128, 2]],
                raw_type=tf.uint8,
                raw_dim=raw_dim,
                net_dim=net_dim,  # very close to org aspect ration
                g_std=std,
                learning_rate=0.001,
                rec_loss_multiplier=x,
               batch_size=2,  # for testing
            )
            train_vae(exp_param)

            exp_param = ExpParam(
                lat_type="continuous",
                dataset='breakout',
                name_prefix='_BLM'+str(x)+'_STD'+str(std),
                latent=[128],
                raw_type=tf.uint8,
                raw_dim=raw_dim,
                net_dim=net_dim,  # very close to org aspect ration
                learning_rate=0.001,
                g_std=std,
                rec_loss_multiplier=x,
               batch_size=2,  # for testing
            )
            train_vae(exp_param)




    '''
    ############### MNIST ###############
    ## CONTINUOUS
    raw_dim = (28, 28, 1)
    net_dim = (28, 28, 1)

    exp_param = ExpParam(
        lat_type="continuous",
        dataset='mnist',
        latent=[8],
        raw_type=tf.float32,
        raw_dim=raw_dim,
        net_dim=net_dim,
        learning_rate=0.001,
        # batch_size=2,  # for testing
    )
    train_vae(exp_param)
    
    ## DISCRETE
    
    exp_param = ExpParam(
        lat_type="discrete",
        dataset='mnist',
        latent=[[8*32, 2]],
        raw_type=tf.float32,
        raw_dim=raw_dim,
        net_dim=net_dim,
        learning_rate=0.001,
        # batch_size=2,  # for testing
    )
    train_vae(exp_param)
    '''

    print('Done')