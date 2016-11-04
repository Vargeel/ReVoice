import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from layers import (_causal_linear, _output_linear, conv1d,
                    dilated_conv1d)

def average_pooling(x,ds_rate):
    return tf.nn.avg_pool(x, ksize=[1, 1, 160, 1],
                        strides=[1, 1, 160, 1], padding='SAME')
def full_weight(name, shape):
    return tf.get_variable(name,shape = shape, initializer=tf.contrib.layers.xavier_initializer(uniform = False))


def bias_variable(name, shape):
    return tf.get_variable(name, shape=shape, initializer=tf.constant_initializer(0.01))



class Model(object):
    def __init__(self,
                 num_time_samples,
                 num_channels=1,
                 num_classes_value=256,
                 num_blocks=4,
                 num_layers=12,
                 num_hidden=128,
                 gpu_fraction=6.1,
                 num_classes = 109):

        self.num_time_samples = num_time_samples
        self.num_channels = num_channels
        self.num_classes_value = num_classes_value
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.gpu_fraction = gpu_fraction

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

        with tf.variable_scope('average_pooling'):
            h_reshape = tf.expand_dims(h, 1)
            output_ap_res = average_pooling(h_reshape,ds_rate = 160)
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
                class_fc = tf.matmul(output_class_resh, class_w) + class_b
                output_class = tf.nn.softmax(class_fc)
            with tf.variable_scope('prediction_layer'):
                outputs_pred = conv1d(h,
                                 num_classes_value,
                                 filter_width=1,
                                 gain=1.0,
                                 activation=None,
                                 bias=True)

        cost_pred = tf.nn.sparse_softmax_cross_entropy_with_logits(outputs_pred, targets_pred)

        cost_class = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(targets_class, output_class))

        alpha = 0.7
        beta = 0.3

        cost = alpha * tf.reduce_mean(cost_pred) + beta * tf.reduce_mean(cost_class)

        train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.initialize_all_variables())

        self.inputs = inputs
        self.targets_pred = targets_pred
        self.targets_class = targets_class
        self.outputs = outputs
        self.hs = hs
        self.cost_pred = cost_pred
        self.cost_class = cost_class
        self.cost = cost
        self.train_step = train_step
        self.sess = sess

    def _train(self, inputs, targets_pred, targets_class):
        feed_dict = {self.inputs: inputs, self.targets_pred: targets_pred, self.targets_class: targets_class}
        cost, _ = self.sess.run(
            [self.cost, self.train_step],
            feed_dict=feed_dict)
        return cost

    def train(self, inputs, targets_pred, target_class):
        losses = []
        terminal = False
        i = 0
        while not terminal:
            i += 1
            cost = self._train(inputs, targets_pred, target_class)
            print 'cost value : {}'.format(cost)
            if cost < 1e-1:
                terminal = True
            losses.append(cost)
            if i % 50 == 0:
                plt.plot(losses)
                plt.show()


class Generator(object):
    def __init__(self, model, batch_size=1, input_size=1):
        self.model = model
        self.bins = np.linspace(-1, 1, self.model.num_classes)

        inputs = tf.placeholder(tf.float32, [batch_size, input_size],
                                name='inputs')

        print('Make Generator.')

        count = 0
        h = inputs

        init_ops = []
        push_ops = []
        for b in range(self.model.num_blocks):
            for i in range(self.model.num_layers):
                rate = 2 ** i
                name = 'b{}-l{}'.format(b, i)
                if count == 0:
                    state_size = 1
                else:
                    state_size = self.model.num_hidden

                q = tf.FIFOQueue(rate,
                                 dtypes=tf.float32,
                                 shapes=(batch_size, state_size))
                init = q.enqueue_many(tf.zeros((rate, batch_size, state_size)))

                state_ = q.dequeue()
                push = q.enqueue([h])
                init_ops.append(init)
                push_ops.append(push)

                h = _causal_linear(h, state_, name=name, activation=tf.nn.relu)
                count += 1

        outputs = _output_linear(h)

        out_ops = [tf.nn.softmax(outputs)]
        out_ops.extend(push_ops)

        self.inputs = inputs
        self.init_ops = init_ops
        self.out_ops = out_ops

        # Initialize queues.
        self.model.sess.run(self.init_ops)

    def run(self, input, num_samples):
        predictions = []
        for step in range(num_samples):

            feed_dict = {self.inputs: input}
            output = self.model.sess.run(self.out_ops, feed_dict=feed_dict)[0]  # ignore push ops
            value = np.argmax(output[0, :])

            input = np.array(self.bins[value])[None, None]
            predictions.append(input)

            if step % 1000 == 0:
                predictions_ = np.concatenate(predictions, axis=1)
                plt.plot(predictions_[0, :], label='pred')
                plt.legend()
                plt.xlabel('samples from start')
                plt.ylabel('signal')
                plt.show()

        predictions_ = np.concatenate(predictions, axis=1)


        return predictions_