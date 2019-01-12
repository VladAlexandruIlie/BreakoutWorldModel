import tensorflow as tf
from tensorboard import main as tb
tf.flags.FLAGS.logdir = "logdir"
tb.main()
