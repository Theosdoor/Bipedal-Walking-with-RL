

normal_ hyperparams = {
    # env/general params
    "max_timesteps": MAX_TIMESTEPS, # per episode [DONT CHANGE]
    "max_episodes": 350,
    "target_score": 300, # stop training when average score over r_list > target_score
    "len_r_list": 100, # length of reward list to average over for target score (stop training when avg > target_score)
    "hardcore": False, # fixed in wandb sweep
    "init_rand_steps": 10000, # number of steps to take with random actions before training (helps exploration)

    # Agent hyperparams (from https://github.com/ArijusLengvenis/bipedal-walker-dreamer/blob/main/dreamer_bipedal_walker_code.ipynb)
    "n_mini_critics": 5, # each mini-critic is a single mlp, which combine to make one mega-critic
    "n_quantiles": 20, # quantiles per mini critic
    "top_quantiles_to_drop_per_net": 'auto', # per mini critic (auto based on n_quantiles)
    "actor_hidden_dims": [512, 512],
    "mini_critic_hidden_dims": [256, 256], # * n_mini_critics
    "batch_size": 256,
    "discount": 0.99, # gamma
    "tau": 0.005,
    "actor_lr": 3.67e-4, # empirically chosen
    "critic_lr": 3.89e-4, # empirically chosen
    "alpha_lr": 3.34e-4, # empirically chosen

    # ERE buffer (see paper for their choices)
    "buffer_size": 100000, # smaller size improves learning early on but is outperformed later on
    "eta0": 0.9975, # 0.994 - 0.999 is good (according to paper)
    "annealing_steps": 'auto', # number of steps to anneal eta over (after which sampling is uniform) - None = auto-set to max estimated steps in training
    "cmin": 5000, # min number of samples to sample from

    # dreamer hyperparams (from https://github.com/ArijusLengvenis/bipedal-walker-dreamer/blob/main/dreamer_bipedal_walker_code.ipynb)
    # info on hyperparam choice: https://arxiv.org/abs/1912.01603
    "use_dreamer": False,
    "intrinsic_reward_scale": 0.0, # scale factor for dreamer intrinsic reward
    "batch_size_dreamer": 512,
    "hidden_dim": 256,
    "num_layers": 16,
    "num_heads": 4,
    "dreamer_lr": 3e-4,
    "dropout_prob": 0.1,
    "window_size": 40,               # transformer context window size
    "step_size": 1,                  # how many timesteps to skip between each context window
    "train_split": 0.8,              # train/validation split
    "loss_threshold": 0.8,          # use dreamer if loss < loss_threshold
    "imagination_horizon": 15,       # how many timesteps to run the dreamer model for (H in Dreamer paper) - need to empirically test for best
    "dreamer_train_epochs": 15,      # how many epochs to train the dreamer model for
    "dreamer_train_frequency": 10,   # how often to train the dreamer model
    "episode_threshold": 50,         # how many episodes to run before training the dreamer model
    "max_size": int(5e4),            # maximum size of the training set for the dreamer model
}

hardcore_hyperparams = {
    # env/general params
    "max_timesteps": MAX_TIMESTEPS, # per episode [DONT CHANGE]
    "max_episodes": 1500,
    "target_score": 300, # stop training when average score over r_list > target_score
    "len_r_list": 100, # length of reward list to average over for target score (stop training when avg > target_score)
    "hardcore": True, # fixed in wandb sweep
    "init_rand_steps": 1000, # number of steps to take with random actions before training (helps exploration)

    # Agent hyperparams (from https://github.com/ArijusLengvenis/bipedal-walker-dreamer/blob/main/dreamer_bipedal_walker_code.ipynb)
    "n_mini_critics": 5, # each mini-critic is a single mlp, which combine to make one mega-critic
    "n_quantiles": 25, # quantiles per mini critic
    "top_quantiles_to_drop_per_net": 'auto', # per mini critic (auto based on n_quantiles)
    "actor_hidden_dims": [512, 512],
    "mini_critic_hidden_dims": [256, 256], # * n_mini_critics
    "batch_size": 256,
    "discount": 0.98, # gamma
    "tau": 0.005,
    "actor_lr": 3.67e-4, # empirically chosen
    "critic_lr": 3.89e-4, # empirically chosen
    "alpha_lr": 3.34e-4, # empirically chosen

    # ERE buffer (see paper for their choices)
    "buffer_size": 500000, # smaller size improves learning early on but is outperformed later on
    "eta0": 0.994, # 0.994 - 0.999 is good (according to paper)
    "annealing_steps": 'auto', # number of steps to anneal eta over (after which sampling is uniform) - None = auto-set to max estimated steps in training
    "cmin": 5000, # min number of samples to sample from

    # dreamer hyperparams (from https://github.com/ArijusLengvenis/bipedal-walker-dreamer/blob/main/dreamer_bipedal_walker_code.ipynb)
    # info on hyperparam choice: https://arxiv.org/abs/1912.01603
    "use_dreamer": True,
    "intrinsic_reward_scale": 0.0, # scale factor for dreamer intrinsic reward
    "batch_size_dreamer": 512,
    "hidden_dim": 256,
    "num_layers": 16,
    "num_heads": 4,
    "dreamer_lr": 3e-4,
    "dropout_prob": 0.1,
    "window_size": 40,               # transformer context window size
    "step_size": 1,                  # how many timesteps to skip between each context window
    "train_split": 0.8,              # train/validation split
    "loss_threshold": 0.8,          # use dreamer if loss < loss_threshold
    "imagination_horizon": 15,       # how many timesteps to run the dreamer model for (H in Dreamer paper) - need to empirically test for best
    "dreamer_train_epochs": 15,      # how many epochs to train the dreamer model for
    "dreamer_train_frequency": 10,   # how often to train the dreamer model
    "episode_threshold": 50,         # how many episodes to run before training the dreamer model
    "max_size": int(5e4),            # maximum size of the training set for the dreamer model
}