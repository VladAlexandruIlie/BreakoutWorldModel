import tensorflow as tf

from A2C.ActorCritic import A2C
from A2C.utils.utils import create_experiment_dirs
from A2C.utils.utils import parse_args


def main():
    # Parse the JSON arguments
    # config_dir = ".\config\breakout.json"
    config_args = parse_args()

    tf.reset_default_graph()

    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=config_args.num_envs,
                            inter_op_parallelism_threads=config_args.num_envs)

    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Prepare Directories
    config_args.experiment_dir, config_args.summary_dir, config_args.checkpoint_dir, config_args.output_dir, config_args.test_dir = \
        create_experiment_dirs(config_args.experiment_dir)

    use_VAE = True

    a2c = A2C(sess, config_args, use_VAE)

    if config_args.to_train:
        a2c.train()
    if config_args.to_test:
        a2c.test(total_timesteps=10000000)


if __name__ == '__main__':
    main()
