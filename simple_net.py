#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import input_data
import matplotlib.pyplot as plt

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h, w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions
    return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us

data = input_data.read_data_sets("datasets/");
trX, trY = data.train.features, data.train.labels;
tvX, tvY = data.validate.features, data.validate.labels;
teX, teY = data.test.features, data.test.labels;

X = tf.placeholder("float32", [None, 5])
Y = tf.placeholder("float32", [None, 1])

w_h = init_weights([5, 100]) # create symbolic variables
w_o = init_weights([100, 1])

py_x = model(X, w_h, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y)) # compute costs
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
predict_op = tf.argmax(py_x, 1)

cost_history = [];
accuracy_history = [];

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    for i in range(300):
        for start, end in zip(range(0, len(trX), 128), range(10, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]});

        cost_history.append(sess.run(cost, feed_dict={X: tvX, Y:tvY}));
        print(i, 'cost', cost_history[len(cost_history) - 1]);
        accuracy_history.append(np.mean(np.argmax(tvY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: tvX})))
        print(i, 'acc', accuracy_history[len(accuracy_history)-1])


    print('Final score:', np.mean(np.argmax(teY, axis=1) ==
                     sess.run(predict_op, feed_dict={X: teX})))

plt.plot(cost_history, 'r--');
plt.text(120, .35, r'Cost')
plt.plot(accuracy_history, 'b--');
plt.text(120, .80, r'Accuracy')
plt.show();
