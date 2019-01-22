
birds_config = {
    # birds further away than this recieve a boundary loss
    'max_distance_from_origin': 5.,
    # birds moving faster than this receive a boundary loss
    'max_velocity': .4,
    'min_velocity': .2,
    # the number of birds in the simulation
    'num_birds': 50,
    # used for numerical stability
    'epsilon': 1e-4,
    # the weight of the "separation loss"
    'separation_weight': .005,
    # the weight of the "boundary loss"
    'boundary_weight': 1.,
    # the weight of the "velocity alignment loss"
    'alignment_weight': 10.,
    # the weight of the spacial cohesion loss
    'cohesion_weight': 1e-3,
    # the radius around a bird to include in the alignment loss
    'alignment_radius': .4,
    # the radius around a bird to include in the separation loss
    'separation_radius': .2,
    # this controls how large of velocity updates birds
    # receive each step
    'learning_rate': .4,
    # how many time steps to run through
    'num_loops': 1000,
    'noise_std': .2,
}
