import tensorflow as tf
from tensorflow.contrib.layers import fully_connected,convolution2d,flatten,batch_norm,max_pool2d,dropout
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.ops import init_ops
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt


def conv_a_trous(l_in, num_outputs, k_size, scope, stride = 1):
    l_norm = batch_norm(l_in)
    l_relu = tf.nn.relu(l_norm)
    return convolution2d(l_relu, num_outputs=num_outputs, rate=2, kernel_size=k_size, padding='SAME',activation_fn=None,scope=scope, stride=stride,weights_initializer=initializers.xavier_initializer(),biases_initializer=init_ops.constant_initializer(0.1))
#model.add(AtrousConvolution2D(64, 3, 3, atrous_rate=(2,2), border_mode='valid', input_shape=(3, 256, 256)))  #### now the actual kernel size is dilated from 3x3 to 5x5 (3+(3-1)*(2-1)=5)

def conv_2d(l_in, num_outputs, k_size, scope, stride = 1):
    l_norm = batch_norm(l_in)
    l_relu = tf.nn.relu(l_norm)
    return convolution2d(l_relu, num_outputs=num_outputs, kernel_size=k_size, padding='SAME',activation_fn=None,scope=scope, stride=stride,weights_initializer=initializers.xavier_initializer(),biases_initializer=init_ops.constant_initializer(0.1))


def time_to_batch(inputs, rate):
    '''If necessary zero-pads inputs and reshape by rate.

    Used to perform 1D dilated convolution.

    Args:
      inputs: (tensor)
      rate: (int)
    Outputs:
      outputs: (tensor)
      pad_left: (int)
    '''
    batch_size = tf.shape(inputs)[0]
    width = tf.shape(inputs)[1]
    _, _, channels = inputs.get_shape().as_list()

    # Also add rate to width to make convolutional causal.
    width_pad = tf.to_int32(rate * (tf.ceil(tf.to_float(width) / rate) + 1))
    pad_left = width_pad - width

    zeros = tf.zeros(shape=(batch_size, pad_left, channels))
    padded = tf.concat(1, (zeros, inputs))
    padded_reshape = tf.reshape(padded, (batch_size * (width_pad / rate), rate,
                                         channels))
    outputs = tf.transpose(padded_reshape, perm=(1, 0, 2))
    return outputs, pad_left - rate


def batch_to_time(inputs, crop_left, rate):
    ''' Reshape to 1d signal, and remove excess zero-padding.

    Used to perform 1D dilated convolution.

    Args:
      inputs: (tensor)
      crop_left: (int)
      rate: (int)
    Ouputs:
      outputs: (tensor)
    '''
    batch_size = tf.shape(inputs)[0] / rate
    width = tf.shape(inputs)[1]
    _, _, channels = inputs.get_shape().as_list()
    out_width = tf.to_int32(width * rate)

    inputs_transposed = tf.transpose(inputs, perm=(1, 0, 2))
    inputs_reshaped = tf.reshape(inputs_transposed,
                                 (batch_size, out_width, channels))
    outputs = tf.slice(inputs_reshaped, [0, crop_left, 0], [-1, -1, -1])
    return outputs



def conv1d(inputs,
           out_channels,
           filter_width=2,
           stride=1,
           padding='VALID',
           data_format='NHWC',
           gain=np.sqrt(2),
           activation=tf.nn.relu,
           bias=False):
    '''One dimension convolution helper function.

    Sets variables with good defaults.

    Args:
      inputs:
      out_channels:
      filter_width:
      stride:
      paddding:
      data_format:
      gain:
      activation:
      bias:

    Outputs:
      outputs:
    '''
    in_channels = inputs.get_shape().as_list()[-1]

    stddev = gain / np.sqrt(filter_width ** 2 * in_channels)
    w_init = tf.random_normal_initializer(stddev=stddev)

    w = tf.get_variable(name='w',
                        shape=(filter_width, in_channels, out_channels),
                        initializer=w_init)

    outputs = tf.nn.conv1d(inputs,
                           w,
                           stride=stride,
                           padding=padding,
                           data_format=data_format)

    if bias:
        b_init = tf.constant_initializer(0.0)
        b = tf.get_variable(name='b',
                            shape=(out_channels,),
                            initializer=b_init)

        outputs = outputs + tf.expand_dims(tf.expand_dims(b, 0), 0)

    if activation:
        outputs = activation(outputs)

    return outputs


def dilated_conv1d(inputs,
                   out_channels,
                   filter_width=2,
                   rate=1,
                   padding='VALID',
                   name=None,
                   gain=np.sqrt(2),
                   activation=tf.nn.relu):
    '''

    Args:
      inputs: (tensor)
      output_channels:
      filter_width:
      rate:
      padding:
      name:
      gain:
      activation:

    Outputs:
      outputs: (tensor)
    '''
    assert name
    with tf.variable_scope(name):
        inputs_, pad_left = time_to_batch(inputs, rate=rate)
        outputs_ = conv1d(inputs_,
                          out_channels=out_channels,
                          filter_width=filter_width,
                          padding=padding,
                          gain=gain,
                          activation=activation)

        outputs = batch_to_time(outputs_, pad_left, rate=rate)

        # Add additional shape information.
        outputs.set_shape(tf.TensorShape([tf.Dimension(None), tf.Dimension(
            None), tf.Dimension(out_channels)]))

    return outputs

def _causal_linear(inputs, state, name=None, activation=None):
    assert name
    '''
    '''
    with tf.variable_scope(name, reuse=True) as scope:
        w = tf.get_variable('w')
        w_r = w[0, :, :]
        w_e = w[1, :, :]

        output = tf.matmul(inputs, w_e) + tf.matmul(state, w_r)

        if activation:
            output = activation(output)
    return output

def max_pool_2d(l_in, scope, k_size = (3,3)):
    return max_pool2d(l_in,kernel_size=k_size,scope=scope)

def normalize(data):
    temp = np.float32(data) - np.min(data)
    out = (temp / np.max(temp) - 0.5) * 2
    return out


def make_batch(path):
    data = wavfile.read(path)[1][:, 0]

    data_ = normalize(data)
    # data_f = np.sign(data_) * (np.log(1 + 255*np.abs(data_)) / np.log(1 + 255))

    bins = np.linspace(-1, 1, 256)
    # Quantize inputs.
    inputs = np.digitize(data_[0:-1], bins, right=False) - 1
    inputs = bins[inputs][None, :, None]

    # Encode targets as ints.
    targets = (np.digitize(data_[1::], bins, right=False) - 1)[None, :]
    return inputs, targets



def _output_linear(h, name=''):
    with tf.variable_scope(name, reuse=True):
        w = tf.get_variable('w')[0, :, :]
        b = tf.get_variable('b')

        output = tf.matmul(h, w) + tf.expand_dims(b, 0)
    return output


class Queue(object):
    def __init__(self, batch_size, state_size, buffer_size, name=None):
        assert name
        self.batch_size = batch_size
        self.state_size = state_size
        self.buffer_size = buffer_size

        with tf.variable_scope(name):
            self.state_buffer = tf.get_variable(
                'state_buffer',
                dtype=tf.float32,
                shape=[buffer_size, batch_size, state_size],
                initializer=tf.constant_initializer(0.0))

            self.pointer = tf.get_variable('pointer',
                                           initializer=tf.constant(0))

    def pop(self):
        state = tf.slice(self.state_buffer, [self.pointer, 0, 0],
                         [1, -1, -1])[0, :, :]
        return state

    def push(self, item):
        update_op = tf.scatter_update(self.state_buffer, self.pointer, item)
        with tf.control_dependencies([update_op]):
            push_op = tf.assign(self.pointer, tf.mod(self.pointer + 1,
                                                     self.buffer_size))
        return push_op


class Model(object):
    def __init__(self, num_time_samples, num_channels, gpu_fraction):
        inputs = tf.placeholder(tf.float32,
                                shape=(None, num_time_samples, num_channels))
        targets = tf.placeholder(tf.int32, shape=(None, num_time_samples))

        h = inputs
        for b in range(2):
            for i in range(14):
                rate = 2**i
                name = 'b{}-l{}'.format(b, i)
                h = dilated_conv1d(h, 128, rate=rate, name=name)

        outputs = conv1d(h,
                         256,
                         filter_width=1,
                         gain=1.0,
                         activation=None,
                         bias=True)

        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            outputs, targets))

        train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.initialize_all_variables())

        self.inputs = inputs
        self.targets = targets
        self.outputs = outputs
        self.cost = cost
        self.train_step = train_step
        self.sess = sess

    def _train(self, inputs, targets):
        feed_dict = {self.inputs: inputs, self.targets: targets}
        cost, _ = self.sess.run(
            [self.cost, self.train_step],
            feed_dict=feed_dict)
        return cost

    def train(self, inputs, targets):
        losses = []
        terminal = False
        i = 0
        while not terminal:
            i += 1
            cost = self._train(inputs, targets)
            if cost < 1e-1:
                terminal = True
            losses.append(cost)
            if i % 50 == 0:
                plt.plot(losses)
                plt.show()


class Generator(object):
    def __init__(self, model, batch_size=1, input_size=1):
        self.model = model
        self.bins = np.linspace(-1, 1, 256)

        inputs = tf.placeholder(tf.float32, [batch_size, input_size],
                                name='inputs')

        print 'Make Generator.'

        count = 0
        h = inputs

        push_ops = []
        for b in range(2):
            for i in range(14):
                rate = 2**i
                name = 'b{}-l{}'.format(b, i)
                if count == 0:
                    state_size = 1
                else:
                    state_size = 128

                q = Queue(batch_size=batch_size,
                          state_size=state_size,
                          buffer_size=rate,
                          name=name)

                state_ = q.pop()
                push = q.push(h)
                push_ops.append(push)
                h = _causal_linear(h, state_, name=name, activation=tf.nn.relu)
                count += 1

        outputs = _output_linear(h)

        out_ops = [tf.argmax(tf.nn.softmax(outputs), 1)]
        out_ops.extend(push_ops)

        # Initialize new variables
        new_vars = [var for var in tf.trainable_variables()
                    if 'pointer' in var.name or 'state_buffer' in var.name]
        self.model.sess.run(tf.initialize_variables(new_vars))

        self.inputs = inputs
        self.out_ops = out_ops

    def run(self, input):

        predictions = []
        for step in range(32000):

            feed_dict = {self.inputs: input}
            outputs = self.model.sess.run(self.out_ops, feed_dict=feed_dict)
            output = outputs[0]  # ignore push ops

            input = self.bins[output][:, None]
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
