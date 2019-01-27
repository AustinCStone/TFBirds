
birds_config = {
    # the number of nearest neighbors to include in the cohesion loss
    'cohesion_top_k': 6,
    # the number of nearest neighbors to include in the alignment loss
    'alignment_top_k': 6,
    # the number of nearest neighbors to include in the separation loss
    'separation_top_k': 3,
    # The desired distance a bird has from its nearest neighbor
    'separation_goal_dist': .2,
    # birds further away than this recieve a boundary loss
    'max_distance_from_origin': 4.,
    # birds moving faster than this receive a boundary loss
    'max_velocity': .4,
    # birds moving slower than this receive a boundary loss
    'min_velocity': .1,
    # the number of birds in the simulation
    'num_birds': 100,
    # used for numerical stability
    'epsilon': 1e-4,
    #### These weights are the knobs you turn at the exploratorium! #####
    # the weight of the "separation loss"
    'separation_weight': 5.,
    # the weight of the "boundary loss"
    'boundary_weight': 1.,
    # the weight of the "velocity alignment loss"
    'alignment_weight': 4.,
    # the weight of the spacial cohesion loss
    'cohesion_weight': 5.,
    # this controls how large of velocity updates birds
    # receive each step
    'learning_rate': .08,
    # how many time steps to run through
    'num_loops': 1000,
    # some noise is added to the bird's velocities to break up singular flocks
    'noise_std': .05,
    # the noise is resampled every noise_update iterations
    'noise_update': 3,
}
