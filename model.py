import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from tf_utils import get_variable

class NN():
    def __init__(self, ):
        n_features = 14
        self.input_x = tf.placeholder(tf.float32, [None, n_features], name='input_x')  # value in the range of (0, 1)
        self.input_y = tf.placeholder(tf.float32, [None, 2], name='input_y')
        self.lr = tf.placeholder("float")
        self.w0 = get_variable('xavier', name='w0', shape=[n_features, 30])
        self.b0 = get_variable('zero', name='b0', shape=[30])
        self.w1 = get_variable('xavier', name='w1', shape=[30, 15])
        self.b1 = get_variable('zero', name='b1', shape=[15])
        self.w2 = get_variable('xavier', name='w2', shape=[15, 2])
        self.b2 = get_variable('zero', name='b2', shape=[2])

        h0 = tf.nn.tanh(tf.add(tf.matmul(self.input_x, self.w0), self.b0))
        h1 = tf.nn.tanh(tf.add(tf.matmul(h0, self.w1), self.b1))
        output = tf.add(tf.matmul(h1, self.w2), self.b2, name='output')
        #print(output)
        self.output_y = output
        alpha = 0.1
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('loss'):
                self.l2reg_loss = tf.nn.l2_loss(self.w0) + tf.nn.l2_loss(self.w1) + tf.nn.l2_loss(self.w2)
                self.clas_loss = tf.losses.softmax_cross_entropy(onehot_labels=self.input_y, logits=self.output_y)
                self.loss = self.clas_loss + alpha * self.l2reg_loss
                #self.loss = self.clas_loss
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=1e-8).minimize(self.loss)


if __name__ == "__main__":
    NN()
