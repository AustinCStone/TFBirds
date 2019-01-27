""" Bird flocking simulator in tensorflow.

See https://en.wikipedia.org/wiki/Flocking_(behavior)
In particular:
    Separation - avoid crowding neighbours (short range repulsion)
    Alignment - steer towards average heading of neighbours
    Cohesion - steer towards average position of neighbours (long range attraction)
"""

from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from config import birds_config as config


def get_dist_mat(xyz_k3):
    """ Return a distance matrix where entry i,j is the difference
    between element i and element j in xyz_k3. """
    xyz_k13 = tf.reshape(xyz_k3, (tf.shape(xyz_k3)[0], 1, 3))
    return tf.sqrt(tf.einsum('ijk,ijk->ij', xyz_k3 - xyz_k13,
                             xyz_k3 - xyz_k13) + config['epsilon'])


def top_k_euclidean_difference(pos_dist_mat_kk, xyz_k3, k, goal_dist=None):
    """ Compute the mean euclidean distance for the k nearest neighbors
    in xyz_k3. As an optimization, the precomputed distance matrix,
    pos_dist_mat_kk is accepted as an input argument. """
    top_k_inds_kk = tf.nn.top_k(-pos_dist_mat_kk, k).indices
    top_k_inds_kk_split = tf.unstack(top_k_inds_kk, axis=0)
    loss = 0.
    for bird_num, top_k_inds in enumerate(top_k_inds_kk_split):
        neighbors_xyz_k3 = tf.gather(xyz_k3, top_k_inds)
        dists_k = tf.sqrt(
            tf.reduce_sum((xyz_k3[bird_num] - neighbors_xyz_k3) ** 2., axis=1) + config['epsilon'])
        if goal_dist is not None:
            dists_k = tf.abs(dists_k - goal_dist)
        loss += tf.reduce_mean(dists_k)
    return loss / config['num_birds']


def get_bounding_loss(pos_xyz_k3, vel_xyz_k3):
    """ Return a loss which penalizes birds for going out of bounds
    or exceeding the maximum velocity. """
    out_of_bounds_xyz_k3 = tf.maximum(
        tf.abs(pos_xyz_k3) - config['max_distance_from_origin'], 0.)
    position_loss = tf.reduce_sum(out_of_bounds_xyz_k3) ** 2.
    over_of_bounds_vel_xyz_k3 = tf.maximum(
        tf.abs(vel_xyz_k3) - config['max_velocity'], 0.)
    under_of_bounds_vel_xyz_k3 = tf.maximum(
        config['min_velocity'] - tf.abs(vel_xyz_k3), 0.)
    velocity_loss = (tf.reduce_sum(over_of_bounds_vel_xyz_k3) + \
        tf.reduce_sum(under_of_bounds_vel_xyz_k3)) ** 2.
    return (position_loss + velocity_loss) / config['num_birds'] * config['boundary_weight']


def get_alignment_loss(pos_dist_mat_kk, vel_xyz_k3):
    """ Return a loss which encourages birds to fly in the same direction as their neighbors. """
    loss = top_k_euclidean_difference(pos_dist_mat_kk, vel_xyz_k3, config['alignment_top_k'])
    return loss * config['alignment_weight']


def get_cohesion_loss(pos_dist_mat_kk, pos_xyz_k3):
    """ Return a loss which encourages birds to fly toward their neighbors' average position. """
    loss = top_k_euclidean_difference(pos_dist_mat_kk, pos_xyz_k3, config['cohesion_top_k'])
    return loss * config['cohesion_weight']


def get_separation_loss(pos_dist_mat_kk, pos_xyz_k3):
    """ Return a loss which encourages birds to separate from their neighbors"""
    loss = top_k_euclidean_difference(pos_dist_mat_kk, pos_xyz_k3, config['separation_top_k'],
                                      goal_dist=config['separation_goal_dist'])
    return loss * config['separation_weight']


def total_loss(pos_xyz_k3, vel_xyz_k3):
    """ Return the total loss for optimization. The loss will be computed with respect
    to the velocities, which are trainable variables. We add the velocities to the positions
    because some losses are computed with respect to the positions; these losses will produce
    velocity gradients which encourage lower position based losses. """
    pos_xyz_k3 += vel_xyz_k3
    pos_dist_mat_kk = get_dist_mat(pos_xyz_k3)
    bounding_loss = get_bounding_loss(pos_xyz_k3, vel_xyz_k3)
    bounding_loss = tf.Print(bounding_loss, [bounding_loss], 'bounding loss: ')
    separation_loss = get_separation_loss(pos_dist_mat_kk, pos_xyz_k3)
    separation_loss = tf.Print(separation_loss, [separation_loss], 'separation loss: ')
    alignment_loss = get_alignment_loss(pos_dist_mat_kk, vel_xyz_k3)
    alignment_loss = tf.Print(alignment_loss, [alignment_loss], 'alignment loss: ')
    cohesion_loss = get_cohesion_loss(pos_dist_mat_kk, pos_xyz_k3)
    cohesion_loss = tf.Print(cohesion_loss, [cohesion_loss], 'cohesion_loss loss: ')
    loss = bounding_loss + alignment_loss + cohesion_loss + separation_loss
    loss = tf.Print(loss, [loss], 'loss is: ')
    return loss


def main():
    pos_xyz_k3 = np.random.randn(config['num_birds'], 3)
    pos_xyz_ph_k3 = tf.placeholder(tf.float32, [config['num_birds'], 3])
    vel_xyz_k3 = tf.Variable(np.random.randn(config['num_birds'], 3).astype('float32') * .01)

    loss = total_loss(pos_xyz_ph_k3, vel_xyz_k3)
    train_op = tf.train.GradientDescentOptimizer(config['learning_rate']).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    noise = np.random.randn(config['num_birds'], 3)

    for iteration in range(config['num_loops']):
        print('On iteration {}'.format(iteration))
        try:
            if iteration % config['noise_update'] == 0:
                noise = np.random.randn(config['num_birds'], 3) * config['noise_std']

            _, l, new_vel_k3 = sess.run([train_op, loss, vel_xyz_k3],
                                        feed_dict={pos_xyz_ph_k3: pos_xyz_k3})
            pos_xyz_k3 += new_vel_k3 + noise
            li, = ax.plot(pos_xyz_k3[:, 0],
                          pos_xyz_k3[:, 1],
                          pos_xyz_k3[:, 2], 'ro')
            ax.relim() 
            ax.autoscale_view(True, True, True)
            plt.show(block=False)
            fig.canvas.draw()
            plt.pause(.01)
        except KeyboardInterrupt:
            break

if __name__ == '__main__':
    main()