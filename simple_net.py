#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import input_data

def init_weights(shape, dev):
    return tf.Variable(tf.random_normal(shape, stddev=dev))


def model(X, w_h, w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions
    return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us

data = input_data.read_data_sets("datasets/");
trX, trY = data.train.features, data.train.labels;
tvX, tvY = data.validate.features, data.validate.labels;
teX, teY = data.test.features, data.test.labels;

X = tf.placeholder("float32", [None, 5])
Y = tf.placeholder("float32", [None, 1])

w_h = init_weights([5, 100], np.sqrt(2 / np.prod(X.get_shape().as_list()[1:]))) # create symbolic variables
w_o = init_weights([100, 1], np.sqrt(2 / np.prod(Y.get_shape().as_list()[1:])))

py_x = model(X, w_h, w_o)

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(py_x, Y)) # compute costs
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
predict_op = tf.nn.sigmoid(py_x);

cost_history = [];
accuracy_history = [];

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    for i in range(300):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]});

        cost_history.append(sess.run(cost, feed_dict={X: tvX, Y:tvY}));
        print(i, 'cost:', sess.run(cost, feed_dict={X: tvX, Y:tvY}));
        print(i, 'accuracy:', (np.round(sess.run(predict_op, feed_dict={X: tvX})) == tvY).mean())


    print('Final score:', (np.round(sess.run(predict_op, feed_dict={X: teX})) == teY).mean())
