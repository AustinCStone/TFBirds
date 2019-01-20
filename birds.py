from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

max_dist = 1
max_vel = .3

NUM_BIRDS = 100
EPS = 1e-4
separation_weight = .01
bound_weight = .01
alignment_weight = 1.
cohesion_weight = 1.
alignment_radius = .3
cohesion_radius = .3
learning_rate = 5.


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# some X and Y data
x = np.random.randn(NUM_BIRDS)
y = np.random.randn(NUM_BIRDS)
z = np.random.randn(NUM_BIRDS)

x_ph = tf.placeholder(tf.float32, [NUM_BIRDS])
y_ph = tf.placeholder(tf.float32, [NUM_BIRDS])
z_ph = tf.placeholder(tf.float32, [NUM_BIRDS])

vx_var = tf.Variable(np.random.randn((NUM_BIRDS)).astype('float32') * .01, name='vx_var')
vy_var = tf.Variable(np.random.randn((NUM_BIRDS)).astype('float32') * .01, name='vy_var')
vz_var = tf.Variable(np.random.randn((NUM_BIRDS)).astype('float32') * .01, name='vz_var')

li, = ax.plot(x, y, z, 'ro')

ax.relim() 
ax.autoscale_view(True, True, True)
fig.canvas.draw()
plt.show(block=False)


def bounding_loss(x, y, z):
    points_N3 = tf.stack([x, y, z], axis=1)
    points_N3 = tf.abs(points_N3)
    points_N3 = tf.maximum(points_N3 - max_dist, 0.)
    loss = tf.reduce_sum(points_N3)
    vel_N3 = tf.stack([vx_var, vy_var, vz_var], axis=1)
    vel_N3 = tf.maximum(tf.abs(vel_N3) - max_vel, 0.)
    return loss + tf.reduce_sum(vel_N3)


def separation_loss(x, y, z):
    points_N3 = tf.stack([x, y, z], axis=1)
    points_N13 = tf.reshape(points_N3, (NUM_BIRDS, 1, 3))
    dist_mat = tf.sqrt(tf.einsum('ijk,ijk->ij', points_N3-points_N13,
                                 points_N3-points_N13) + EPS)
    return -tf.reduce_mean(dist_mat / 2.), dist_mat


def alignment_loss(pos_dist_mat):
    _, vel_dist_mat = separation_loss(vx_var, vy_var, vz_var)
    vel_dists_K = tf.gather_nd(
        vel_dist_mat, tf.where(pos_dist_mat < alignment_radius))
    return tf.reduce_mean(vel_dists_K)


def cohesion_loss(pos_dist_mat, x, y, z):
    pos_dists_K = tf.gather_nd(
        pos_dist_mat, tf.where(pos_dist_mat < cohesion_radius))
    return tf.reduce_mean(pos_dists_K)


def total_loss():
    next_x = x_ph + vx_var
    next_y = y_ph + vy_var
    next_z = z_ph + vz_var
    sep_loss, pos_dist_mat = separation_loss(next_x, next_y, next_z)
    sep_loss *= separation_weight
    align_loss = alignment_weight * alignment_loss(pos_dist_mat)
    c_loss = cohesion_weight * cohesion_loss(pos_dist_mat, next_x, next_y, next_z)
    bound_loss = bound_weight * bounding_loss(next_x, next_y, next_z)
    sep_loss = tf.Print(sep_loss, [sep_loss], "SEP LOSS: ")
    align_loss = tf.Print(align_loss, [align_loss], "align LOSS: ")
    c_loss = tf.Print(c_loss, [c_loss], "cohesion LOSS: ")
    boung_loss = tf.Print(bound_loss, [bound_loss], "bound LOSS: ")
    return sep_loss + align_loss + c_loss + bound_loss


x[:] = np.random.randn(NUM_BIRDS)
y[:] = np.random.randn(NUM_BIRDS)
z[:] = np.random.randn(NUM_BIRDS)

# The Gradient Descent Optimizer does the heavy lifting
loss = total_loss()
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, var_list=tf.trainable_variables())


sess = tf.Session()
sess.run(tf.global_variables_initializer())

# loop to update the data
for _ in range(1000):
    try:
        _, l, vx, vy, vz = sess.run([train_op, loss, vx_var, vy_var, vz_var],
                                    feed_dict={x_ph: x, y_ph: y, z_ph: z})
        x += vx
        y += vy
        z += vz

        print('loss is {}'.format(l))

        li, = ax.plot(x, y, z, 'ro')

        ax.relim() 
        ax.autoscale_view(True, True, True)
        plt.show(block=False)

        fig.canvas.draw()

        plt.pause(.1)
    except KeyboardInterrupt:
        break