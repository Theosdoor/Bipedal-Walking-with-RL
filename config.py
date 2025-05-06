

# hardcore params
hyperparams = {
    # env/general params
    "max_timesteps": MAX_TIMESTEPS, # per episode [DONT CHANGE]
    "max_episodes": 1000,
    "target_score": 300, # stop training when average score over r_list > target_score
    "len_r_list": 100, # length of reward list to average over for target score (stop training when avg > target_score)
    "hardcore": True, # fixed in wandb sweep
    "init_rand_steps": 10000, # number of steps to take with random actions before training (helps exploration)

    # Agent hyperparams 
    "n_mini_critics": 5, # each mini-critic is a single mlp, which combine to make one mega-critic
    "n_quantiles": 20, # quantiles per mini critic
    "top_quantiles_to_drop_per_net": 'auto', # per mini critic (auto based on n_quantiles)
    "actor_hidden_dims": [512, 512],
    "mini_critic_hidden_dims": [256, 256], # * n_mini_critics
    "batch_size": 256,
    "discount": 0.98, # gamma
    "tau": 0.005,
    "actor_lr": 3.29e-4, # empirically chosen
    "critic_lr": 3.5e-4, # empirically chosen
    "alpha_lr": 3.24e-4, # empirically chosen

    # ERE buffer (see paper for their choices)
    "use_per": True, # use PER sampling as well (DO NOT USE ON MAC IT BREAKS AHHH)
    "buffer_size": 1000000, # smaller size improves learning early on but is outperformed later on
    "eta0": 0.996, # 0.994 - 0.999 is good (according to paper)
    "annealing_steps": 'auto', # number of steps to anneal eta over (after which sampling is uniform) - None = auto-set to max estimated steps in training
    "cmin": 5000, # min number of samples to sample from
    "recency_scale": 1, # scale factor for recency
}