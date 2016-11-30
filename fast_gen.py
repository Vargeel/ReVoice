import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from layers import (_causal_linear, _output_linear, conv1d,
                    dilated_conv1d)

def average_pooling(x,ds_rate=160):
    return tf.nn.avg_pool(x, ksize=[1, 1, ds_rate, 1],
                        strides=[1, 1, ds_rate, 1], padding='SAME')
def full_weight(name, shape):
    return tf.get_variable(name,shape = shape, initializer=tf.contrib.layers.xavier_initializer(uniform = False))


def bias_variable(name, shape):
    return tf.get_variable(name, shape=shape, initializer=tf.constant_initializer(0.01))



class Model(object):
    def __init__(self,
                 num_time_samples,
                 num_channels=1,
                 num_classes_value=256,
                 num_blocks=2,
                 num_layers=14,
                 num_hidden=128,
                 num_classes = 109):

        self.num_time_samples = num_time_samples
        self.num_channels = num_channels
        self.num_classes_value = num_classes_value
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_hidden = num_hidden

        inputs = tf.placeholder(tf.float32,
                                shape=(None, num_time_samples, num_channels))
        targets_pred = tf.placeholder(tf.int32, shape=(None, num_time_samples))
        targets_class = tf.placeholder(tf.float32, shape=(None, 109))
        b_size = tf.shape(inputs)[0]
        class_num = self.num_classes
        h = inputs
        hs = []
        for b in range(num_blocks):
            for i in range(num_layers):
                rate = 2 ** i
                name = 'block{}-layer{}'.format(b, i)
                h = dilated_conv1d(h, num_hidden, rate=rate, name=name)
                hs.append(h)

        with tf.variable_scope('prediction_layer'):
            outputs_pred = conv1d(h,
                             num_classes_value,
                             filter_width=1,
                             gain=1.0,
                             activation=None,
                             bias=True)

        cost_pred = tf.nn.sparse_softmax_cross_entropy_with_logits(outputs_pred, targets_pred)

        with tf.variable_scope('average_pooling'):
            h_reshape = tf.expand_dims(h, 1)
            output_ap_res = average_pooling(h_reshape,ds_rate = 20)
            shape = output_ap_res.get_shape().as_list()
            output_ap = tf.reshape(output_ap_res,[b_size,shape[2],shape[3]])
        with tf.variable_scope('non_causal_convolution'):
            with tf.variable_scope('conv_1'):
                fc_1 = conv1d(output_ap, num_hidden,
                                         filter_width=1,
                                         gain=1.0,
                                         activation=None,
                                         bias=True )
            with tf.variable_scope('conv_2'):
                fc_2 = conv1d(fc_1, num_hidden,
                                         filter_width=1,
                                         gain=1.0,
                                         activation=None,
                                         bias=True )
            with tf.variable_scope('conv_3'):
                outputs = conv1d(fc_2, num_classes,
                                         filter_width=1,
                                         gain=1.0,
                                         activation=None,
                                         bias=True )

            with tf.variable_scope('classification_layer'):
                size_final_layer = outputs.get_shape().as_list()[1] * outputs.get_shape().as_list()[2]
                output_class_resh = tf.reshape(outputs, [b_size, size_final_layer])

                class_w = full_weight('w_cl', [size_final_layer,class_num])
                class_b = bias_variable('b_fc1', [class_num])
                output_class = tf.matmul(output_class_resh, class_w) + class_b


        cost_class = tf.nn.softmax_cross_entropy_with_logits(output_class, targets_class)

        alpha = 0.9
        beta = 0.1

        cost = alpha * tf.reduce_mean(cost_pred) + beta * tf.reduce_mean(cost_class)
        # cost =  tf.reduce_mean(cost_pred)

        train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)


        sess = tf.Session(config=tf.ConfigProto())
        sess.run(tf.initialize_all_variables())

        self.inputs = inputs
        self.targets_pred = targets_pred
        self.targets_class = targets_class
        self.outputs_pred = outputs_pred
        self.hs = hs
        self.cost_pred = cost_pred
        self.cost_class = cost_class
        self.cost = cost
        self.train_step = train_step
        self.sess = sess


    def train(self, inputs, targets_pred, targets_class,print_b = False):
        feed_dict = {self.inputs: inputs, self.targets_pred: targets_pred, self.targets_class: targets_class}
        cost, _ = self.sess.run(
            [self.cost, self.train_step],
            feed_dict=feed_dict)
        if np.isnan(cost): import ipdb; ipdb.set_trace()
        if print_b:
            print 'cost value : {}'.format(cost)

