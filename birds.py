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


def get_bounding_loss(pos_xyz_k3, vel_xyz_k3):
    """ Return a loss which penalizes birds for going out of bounds
    or exceeding the maximum velocity. """
    out_of_bounds_xyz_k3 = tf.maximum(
        tf.abs(pos_xyz_k3) - config['max_distance_from_origin'], 0.)
    position_loss = tf.reduce_sum(out_of_bounds_xyz_k3)
    over_of_bounds_vel_xyz_k3 = tf.maximum(
        tf.abs(vel_xyz_k3) - config['max_velocity'], 0.)
    under_of_bounds_vel_xyz_k3 = tf.maximum(
        config['min_velocity'] - tf.abs(vel_xyz_k3), 0.)
    velocity_loss = tf.reduce_sum(over_of_bounds_vel_xyz_k3) + \
        tf.reduce_sum(under_of_bounds_vel_xyz_k3)
    return (position_loss + velocity_loss) * config['boundary_weight']


def get_separation_loss(pos_dist_mat_kk):
    """ Return a loss which encourages birds to separate from each other
    spatially within a certain radius. We return the negative mean
    of the distances of all pairwise birds within the radius. """
    dists_k = tf.gather_nd(
        pos_dist_mat_kk, tf.where(pos_dist_mat_kk < config['separation_radius']))
    return -tf.reduce_sum(dists_k / 2.) * config['separation_weight']


def get_alignment_loss(pos_dist_mat_kk, vel_dist_mat_kk):
    """ Return a loss which encourages birds within a certain radius
    to fly in a similar direction. """
    vel_dists_k = tf.gather_nd(
        vel_dist_mat_kk, tf.where(pos_dist_mat_kk < config['alignment_radius']))
    return tf.reduce_sum(vel_dists_k / 2.) * config['alignment_weight']


def get_cohesion_loss(pos_dist_mat_kk):
    """ Return a loss which encourages birds to fly together generally. """
    return tf.reduce_sum(pos_dist_mat_kk / 2.) * config['cohesion_weight']


def total_loss(pos_xyz_k3, vel_xyz_k3):
    """ Return the total loss for optimization. The loss will be computed with respect
    to the velocities, which are trainable variables. We add the velocities to the positions
    because some losses are computed with respect to the positions; these losses will produce
    velocity gradients which encourage lower position based losses. """

    pos_xyz_k3 += vel_xyz_k3
    # add random noise to prevent falling into stable minimum
    pos_xyz_k3 += tf.random_normal(shape=tf.shape(pos_xyz_k3), mean=0.0,
                                   stddev=config['noise_std'], dtype=tf.float32)
    pos_dist_mat_kk = get_dist_mat(pos_xyz_k3)
    vel_dist_mat_kk = get_dist_mat(vel_xyz_k3)
    bounding_loss = get_bounding_loss(pos_xyz_k3, vel_xyz_k3)
    bounding_loss = tf.Print(bounding_loss, [bounding_loss], 'bounding loss: ')
    separation_loss = get_separation_loss(pos_dist_mat_kk)
    separation_loss = tf.Print(separation_loss, [separation_loss], 'separation loss: ')
    alignment_loss = get_alignment_loss(pos_dist_mat_kk, vel_dist_mat_kk)
    alignment_loss = tf.Print(alignment_loss, [alignment_loss], 'alignment loss: ')
    cohesion_loss = get_cohesion_loss(pos_dist_mat_kk)
    cohesion_loss = tf.Print(cohesion_loss, [cohesion_loss], 'cohesion_loss loss: ')
    
    return bounding_loss + separation_loss + alignment_loss + cohesion_loss


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
    li, = ax.plot(pos_xyz_k3[:, 0],
                  pos_xyz_k3[:, 1],
                  pos_xyz_k3[:, 2], 'ro')

    ax.relim() 
    ax.autoscale_view(True, True, True)
    fig.canvas.draw()
    plt.show(block=False)

    for _ in range(config['num_loops']):
        try:
            _, l, new_vel_k3 = sess.run([train_op, loss, vel_xyz_k3],
                                       feed_dict={pos_xyz_ph_k3: pos_xyz_k3})
            pos_xyz_k3 += new_vel_k3
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