import numpy as np
import os
from scipy.io import wavfile
import fast_gen as fg
import scipy.signal as sc
from tensorflow.python.framework import ops
from Generator_model import Generator


def normalize(data):
    temp = np.float32(data) - np.min(data)
    out = (temp / np.max(temp) - 0.5) * 2
    return out


def make_batch(path,class_num):
    data = wavfile.read(path)[1]

    data_ = normalize(data)

    if data.shape[0]>10000 and data.shape[0]<200000 :
        data_ = np.lib.pad(data_, (0, 200000 - data_.shape[0]), 'constant', constant_values=(0, 0))
        # data_f = np.sign(data_) * (np.log(1 + 255*np.abs(data_)) / np.log(1 + 255))
        data_ = sc.resample(data_, 20001)
        bins = np.linspace(-1, 1, 255)
        # Quantize inputs.
        inputs = np.digitize(data_[0:-1], bins, right=False) - 1
        inputs = bins[inputs][None, :, None]

        # Encode targets as ints.
        targets_pred = (np.digitize(data_[1::], bins, right=False))[None, :]

        target_class = np.zeros(109)
        target_class[int(class_num)] = 1
        # inputs = np.lib.pad(inputs, ((0,0), (0, 200000 - inputs.shape[1]), (0,0)), 'constant',
        #                          constant_values=(0, 0))
        # targets_pred = np.lib.pad(targets_pred, ((0,0), (0, 200000 - targets_pred.shape[1])), 'constant',
        #                          constant_values=(0, 0))
        #
        # inputs = sc.resample(inputs.reshape(inputs.shape[1]), 20000)
        # targets_pred = sc.resample(targets_pred.reshape(targets_pred.shape[1]), 20000)
        inputs = inputs.reshape([1,20000,1])
        targets_pred = targets_pred.reshape([1,20000])
        return inputs, targets_pred, target_class
    else :
        return [], [], []

def generate_batches(root_path = 'data/wav48/', indexes = range(0,10)):
    inputs = []
    target_class = []
    target_pred =[]
    ns = 0
    for path, subdirs, files in os.walk(root_path):
        for name in files:
            if name[-3:] == 'wav':
                if ns in indexes:
                    path_to_file = os.path.join(path, name)
                    class_num = path[-3:]
                    inputs_loc, targets_pred_loc, targets_class_loc = make_batch(path_to_file,class_num)
                    if len(inputs_loc)!=0:
                        inputs.append(inputs_loc)
                        target_pred.append(targets_pred_loc)
                        target_class.append(targets_class_loc)
                ns += 1

    if len(inputs) == 0:
        return generate_batches(root_path = 'data/wav48/', indexes = indexes +len(indexes) )
    else:
        inputs_st = np.vstack(inputs)
        target_pred_st = np.vstack(target_pred)
        target_class_st = np.vstack(target_class)

        return inputs_st,target_pred_st,target_class_st

def train_model(b_size = 1):
    num_files = 10000
    batch_size = b_size
    num_time_samples = 20000
    num_channels = 1

    model = fg.Model(num_time_samples=num_time_samples,
                     num_channels=num_channels)


    for epoch in range(100000):
        rnd = np.random.permutation(num_files)

        for iteration in range(num_files/batch_size):

            indices = rnd[iteration*batch_size:(iteration +1)*batch_size]
            inputs, targets_pred, targets_class  = generate_batches(indexes = indices)

            if inputs.shape[1] != num_time_samples :
                import ipdb
                ipdb.set_trace()
            if iteration%100 == 0:
                print epoch * num_files + iteration

                print_b = True
            else :
                print_b = False

            model.train(inputs= inputs, targets_pred=targets_pred, targets_class = targets_class, print_b=print_b)

    generator = Generator(model)
    inputs, targets_pred, targets_class = generate_batches(indexes=indices)
    input_ = inputs[:, 0:1, 0]


    predictions = generator.run(input_, 32000)
    wavfile.write(filename='predictions.wav', rate=4410, data=predictions)

if __name__ == '__main__':
    ops.reset_default_graph()
    train_model()