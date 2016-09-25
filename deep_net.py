#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import input_data


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden): # this network is the same as the previous one except with an extra hidden layer + dropout
    X = tf.nn.dropout(X, p_keep_input)
    h = tf.nn.relu(tf.matmul(X, w_h))

    h = tf.nn.dropout(h, p_keep_hidden)
    h2 = tf.nn.relu(tf.matmul(h, w_h2))

    h2 = tf.nn.dropout(h2, p_keep_hidden)

    return tf.matmul(h2, w_o)


data = input_data.read_data_sets("datasets/");
trX, trY = data.train.features, data.train.labels;
tvX, tvY = data.validate.features, data.validate.labels;
teX, teY = data.test.features, data.test.labels;

X = tf.placeholder("float32", [None, 5])
Y = tf.placeholder("float32", [None, 1])

w_h = init_weights([5, 100])
w_h2 = init_weights([100, 100])
w_o = init_weights([100, 1])

p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)
#
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.0001, 0.8).minimize(cost)
predict_op = tf.nn.sigmoid(py_x)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_input: 0.8, p_keep_hidden: 0.5})

        print(i, 'cost:', sess.run(cost, feed_dict={X: tvX, Y:tvY, p_keep_input: 1.0, p_keep_hidden: 1.0}));
        print(i, 'accuracy:', (np.round(sess.run(predict_op, feed_dict={X: tvX,
                                p_keep_input: 1.0, p_keep_hidden: 1.0})) == tvY).mean())


        # print(i, np.mean(np.argmax(teY, axis=1) ==
                        #  sess.run(predict_op, feed_dict={X: teX, Y: teY,
                                                        #  p_keep_input: 1.0,
                                                        #  p_keep_hidden: 1.0})))
    print('Final score:', (np.round(sess.run(predict_op, feed_dict={X: teX,
                                        p_keep_input: 1.0,
                                        p_keep_hidden: 1.0})) == teY).mean())
