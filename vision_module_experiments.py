import tensorflow as tf

from exp_parameters import ExpParam
from train_vision import train_vae

############### MNIST ###############
## CONTINUOUS
raw_dim = (28, 28, 1)
net_dim = (28, 28, 1)

exp_param = ExpParam(
    lat_type="continuous",
    dataset='mnist',
    latent=[4],
    raw_type=tf.float32,
    raw_dim=raw_dim,
    net_dim=net_dim,
    # batch_size=16,
    max_example=1e5,
)
train_vae(exp_param)

## DISCRETE
exp_param = ExpParam(
    lat_type="discrete",
    dataset='mnist',
    latent=[[2 * 32, 2]],
    raw_type=tf.float32,
    raw_dim=raw_dim,
    net_dim=net_dim,
    learning_rate=0.001,
    # batch_size=16,
    max_example=1e5,
)
train_vae(exp_param)

print('Done')