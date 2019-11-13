import gzip
import os
import time

import numpy as np
import six
import tensorflow as tf
import zhusuan as zs
from six.moves import cPickle as pickle


def to_one_hot(x, depth):
    ret = np.zeros((x.shape[0], depth))
    ret[np.arange(x.shape[0]), x] = 1
    return ret


def load_mnist(path, one_hot=True):
    with gzip.open(path, 'rb') as f:
        if six.PY2:
            train_set, valid_set, test_set = pickle.load(f)
        else:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    x_train, t_train = train_set[0], train_set[1]
    x_valid, t_valid = valid_set[0], valid_set[1]
    x_test, t_test = test_set[0], test_set[1]
    n_y = t_train.max() + 1
    t_transform = (lambda x: to_one_hot(x, n_y)) if one_hot else (lambda x: x)
    return x_train, t_transform(t_train), x_valid, t_transform(t_valid), x_test, t_transform(t_test)


def makedirs(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))


def save_image_collections(x, filename, shape=(10, 10), scale_each=False,
                           transpose=False):
    from skimage import io, img_as_ubyte
    from skimage.exposure import rescale_intensity
    makedirs(filename)
    n = x.shape[0]
    if transpose:
        x = x.transpose(0, 2, 3, 1)
    if scale_each is True:
        for i in range(n):
            x[i] = rescale_intensity(x[i], out_range=(0, 1))
    n_channels = x.shape[3]
    x = img_as_ubyte(x)
    r, c = shape
    if r * c < n:
        print('Shape too small to contain all images')
    h, w = x.shape[1:3]
    ret = np.zeros((h * r, w * c, n_channels), dtype='uint8')
    for i in range(r):
        for j in range(c):
            if i * c + j < n:
                ret[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = x[i * c + j]
    ret = ret.squeeze()
    io.imsave(filename, ret)


@zs.meta_bayesian_net(scope="gen", reuse_variables=True)
def build_gen(x_dim, z_dim, y, n, n_particles=1):
    bn = zs.BayesianNet()
    z_mean = tf.zeros([n, z_dim])
    z = bn.normal("z", z_mean, std=1., group_ndims=1, n_samples=n_particles)
    y = tf.reshape(y, [1, tf.shape(y)[0], 10])
    y = tf.tile(y, [n_particles, 1, 1])
    h = tf.layers.dense(tf.concat([y, z], axis=2), 500, activation=tf.nn.relu)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)
    x_logits = tf.layers.dense(h, x_dim)
    bn.deterministic("x_mean", tf.sigmoid(x_logits))
    bn.bernoulli("x", x_logits, group_ndims=1)
    return bn


@zs.reuse_variables(scope="q_net")
def build_q_net(x, z_dim, y, n_z_per_x):
    bn = zs.BayesianNet()
    h = tf.layers.dense(tf.concat([tf.cast(x, tf.float32), y], axis=-1), 500, activation=tf.nn.relu)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)
    z_mean = tf.layers.dense(h, z_dim)
    z_logstd = tf.layers.dense(h, z_dim)
    bn.normal("z", z_mean, logstd=z_logstd, group_ndims=1, n_samples=n_z_per_x)
    return bn


def main():
    data_path = "/home/zhikun/PycharmProjects/data/MNIST/mnist.pkl.gz"
    # the file comes from http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
    x_train, t_train, x_valid, t_valid, x_test, t_test = load_mnist(data_path)
    x_train = np.vstack([x_train, x_valid])
    y_train = np.vstack([t_train, t_valid])
    x_test = np.random.binomial(1, x_test, size=x_test.shape)
    x_dim = x_train.shape[1]
    y_dim = y_train.shape[1]
    # Define model parameters
    z_dim = 40
    n_particles = tf.placeholder(tf.int32, shape=[], name="n_particles")
    x_input = tf.placeholder(tf.float32, shape=[None, x_dim], name="x")
    y = tf.placeholder(tf.float32, shape=[None, y_dim], name="y")
    x = tf.cast(tf.less(tf.random_uniform(tf.shape(x_input)), x_input), tf.int32)
    n = tf.placeholder(tf.int32, shape=[], name="n")

    model = build_gen(x_dim, z_dim, y, n, n_particles)
    variational = build_q_net(x, z_dim, y, n_particles)

    lower_bound = zs.variational.elbo(
        model, {"x": x}, variational=variational, axis=0)
    cost = tf.reduce_mean(lower_bound.sgvb())
    lower_bound = tf.reduce_mean(lower_bound)

    # # Importance sampling estimates of marginal log likelihood
    is_log_likelihood = tf.reduce_mean(
        zs.is_loglikelihood(model, {"x": x}, proposal=variational, axis=0))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    infer_op = optimizer.minimize(cost)

    # Random generation
    y_observe = tf.placeholder(tf.float32, shape=[None, 10], name="y_obs")
    x_gen = tf.reshape(model.observe(y=y_observe)["x_mean"], [-1, 28, 28, 1])

    # Define training/evaluation parameters
    epochs = 3000
    batch_size = 16
    iters = x_train.shape[0] // batch_size
    save_freq = 10
    test_freq = 10
    test_batch_size = 20
    test_iters = x_test.shape[0] // test_batch_size
    result_path = "results/c-vae"

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, epochs + 1):
            time_epoch = -time.time()
            np.random.shuffle(x_train)
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                y_batch = y_train[t * batch_size:(t + 1) * batch_size]
                _, lb = sess.run([infer_op, lower_bound],
                                 feed_dict={x_input: x_batch,
                                            y: y_batch,
                                            n_particles: 1,
                                            n: batch_size})
                lbs.append(lb)
            time_epoch += time.time()
            print("Epoch {} ({:.1f}s): Lower bound = {}".format(
                epoch, time_epoch, np.mean(lbs)))

            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lbs, test_lls = [], []
                for t in range(test_iters):
                    test_x_batch = x_test[t * test_batch_size:(t + 1) * test_batch_size]
                    test_y_batch = t_test[t * test_batch_size:(t + 1) * test_batch_size]
                    test_lb = sess.run(lower_bound,
                                       feed_dict={x: test_x_batch,
                                                  y: test_y_batch,
                                                  n_particles: 1,
                                                  n: test_batch_size})
                    test_ll = sess.run(is_log_likelihood,
                                       feed_dict={x: test_x_batch,
                                                  y: test_y_batch,
                                                  n_particles: 1000,
                                                  n: test_batch_size})
                    test_lbs.append(test_lb)
                    test_lls.append(test_ll)
                time_test += time.time()
                print(">>> TEST ({:.1f}s)".format(time_test))
                print(">> Test lower bound = {}".format(np.mean(test_lbs)))
                print('>> Test log likelihood (IS) = {}'.format(np.mean(test_lls)))

            if epoch % save_freq == 0:
                y_index = np.repeat(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 10)
                y_index = np.eye(10)[y_index.reshape(-1)]
                images = sess.run(x_gen, feed_dict={y_observe: y_index, y: y_index, n: 100, n_particles: 1})
                name = os.path.join(result_path, "vae.epoch.{}.png".format(epoch))
                save_image_collections(images, name)


if __name__ == "__main__":
    main()
