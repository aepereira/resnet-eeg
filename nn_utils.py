from __future__ import division
import numpy as np
import pandas as pd
from math import floor
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.callbacks import Callback
import h5py


def img_scale(array, use_cutout=False):
    """Takes a 4-dimensional channels-last
    NumPy array (to match Keras/TF image tensor 
    format), bins each sample's data points from
    0 to 255, and then rescales from 0 to 1. 
    
    Arguments:
        array {NDArray} -- Data.

    Keyword Arguments:
        use_cutout {bool} -- Whether to use cutout, default: False.
    
    Returns:
        NDArray -- Transformed data.
    """

    n_array = np.empty_like(array)
    for idx, d in enumerate(array):
        dat = np.copy(d)
        d_min = np.amin(dat)
        dat -= d_min
        d_max = np.amax(dat)
        s = 255 / d_max
        dat *= s
        dat = np.rint(dat)
        dat /= 255
        if use_cutout:
            cutout(dat)
        n_array[idx] = dat
    return n_array

def cutout(array):
    """Takes a 3-dimensional channels-last
    NumPy array (a single sample) (already normalized from [0-1]) 
    and cuts out a random area of size Ds/2 in place, where Ds
    is the smaller of the first two dimensions of the sample.

    Arguments:
        array {NDArray} -- Data.
    """

    grey = 0.5
    x_size = array.shape[0]
    y_size = array.shape[1]
    cut_size = np.min([x_size, y_size]) // 2
    # Per Cutout paper, can partially cover at the edges.
    x_idxs = np.linspace(1- cut_size, x_size - 1, x_size + cut_size - 1, dtype=int)
    y_idxs = np.linspace(1 - cut_size, y_size - 1, y_size + cut_size - 1, dtype=int)
    cut_l = np.random.choice(x_idxs) # Left edge of cutout box
    cut_r = cut_l + cut_size # Right edge of cutout box
    cut_t = np.random.choice(y_idxs) # Top edge of cutout box
    cut_b = cut_t + cut_size # Bottom edge of cutout box
    
    # Find edges of cutout box within image
    l_idx = cut_l if cut_l >= 0 else 0 # Left edge
    r_idx = cut_r if cut_r <= x_size else x_size # Right edge
    t_idx = cut_t if cut_t >= 0 else 0 # Top edge
    b_idx = cut_b if cut_b <= y_size else y_size # Bottom edge

    # Make the cutout
    array[t_idx: b_idx, l_idx: r_idx, :] = grey


def ml_metrics(model, x_test, y_test, verbose=False, cuts=1):
    """Runs test set through Keras model,
    makes predictions, and prints out
    metrics.
    
    Arguments:
        model {keras.model} -- Keras model which outputs a
        single probability prediction from 0 to 1.
        x_test {NDArray} -- Test data in 4-dimensional
        Keras/TF channels_last format.
        y_test {array-like} -- True labels.

    Returns:
        dict -- Dictionary of metrics.
    """
    if len(y_test.shape) == 2:
        """
        Get class label from one-hot vectors.
        """
        predictions = []
        for n in range(cuts):
            x_test_cp = np.copy(x_test)
            if n > 0:
                for x in x_test_cp:
                    cutout(x)
            y_pred_n = model.predict(x_test_cp)
            predictions.append(y_pred_n)
        y_pred_probs = np.mean(predictions, axis=0)
        y_pred_c = [np.argmax(y) for y in y_pred_probs]

        y_test_c = [np.argmax(y) for y in y_test]
        # P300 is at index 0 and NP300 at index 1
        y_pred_c = [(y - 1) * -1 for y in y_pred_c] # Flip so P300 label is 1
        y_test_c = [(y - 1) * -1 for y in y_test_c] # and NP300 is 0.

    elif len(y_test.shape) == 1:
        predictions = []
        for n in range(cuts):
            x_test_cp = np.copy(x_test)
            if n > 0:
                for x in x_test_cp:
                    cutout(x)
            y_pred_n = model.predict(x_test_cp)
            predictions.append(y_pred_n)
        y_pred_probs = np.mean(predictions, axis=0)
        y_pred_c = [int(y > 0.5) for y in y_pred_probs]
        y_test_c = y_test

    else:
        print("Labels are not the right shape.")
        print("Skipping performance metrics.")
        return

    cm = confusion_matrix(y_test_c, y_pred_c)
    sums_real = cm.sum(axis=1)
    pos_real_count = sums_real[1]
    neg_real_count = sums_real[0]
    sums_pred = cm.sum(axis=0)
    pos_pred_count = sums_pred[1]
    neg_pred_count = sums_pred[0]
    precision = cm[1][1] / pos_pred_count
    recall = cm[1][1] / pos_real_count
    bal_acc = 0.5 * precision + 0.5 * (cm[0][0] / neg_pred_count)
    unbal_acc = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
    false_alarm = cm[0][1] / neg_real_count
    false_omission = cm[1][0] / neg_pred_count
    if verbose:
        print("Precision: {:.4f}%".format(precision * 100))
        print("Recall: {:.4f}%".format(recall * 100))
        print("Unbalanced accuracy: {:.4f}%".format(unbal_acc * 100))
        print("Balanced accuracy: {:.4f}%".format(bal_acc * 100))
        print("False alarm: {:.4f}%".format(false_alarm * 100))
        print("False omission: {:.4f}%".format(false_omission * 100))

    met_dict = {'precision': precision,
                'recall': recall,
                'bal_acc': bal_acc,
                'unbal_acc': unbal_acc,
                'false_alarm': false_alarm,
                'false_omission': false_omission}

    return met_dict

def boot_ml_metrics(model, x_test, y_test, n=5, split=0.6, cuts=1):
    """Bootstrapped metrics. See ml_metrics.
    """

    boot_met = {'precision' : [],
                'recall': [],
                'bal_acc': [],
                'unbal_acc': [],
                'false_alarm': [],
                'false_omission': []}

    n_samples = x_test.shape[0]
    all_idxs = np.linspace(0, n_samples - 1, 
                           n_samples, dtype = int)
    split_int = floor(split * n_samples)

    for _ in range(n):
        idxs = np.random.choice(all_idxs, 
                                size=split_int,
                                replace=False)
        met = ml_metrics(model, x_test[idxs], y_test[idxs], verbose=False, cuts=cuts)
        for key, val in met.items():
            boot_met[key].append(val)

    return boot_met


def load_data(file, mode, block_mat_len=18, use_cutout=False):
    """Import data created by make_nn_data
    as 4D Numpy arrays that can be turned
    into Keras/TF channels_last image tensors.
    
    Arguments:
        file {csv file} -- Data file to enter.

        mode {str} -- 'int' or 'one-hot'.
    
    Keyword Arguments:
        block_mat_len {int} -- Edge dimension of each
        of the block matrices that form the expanded
        covariance matrix features. (default: {18})

        use_cutout {bool} -- Whether to use cutout, default: False.
    
    Returns:
        tuple -- x (data in 4D channels_last format),
        y (labels), imbal_factor (ratio of NP300s to P300s)
    """

    if not (mode == 'one-hot' or mode == 'int'):
        raise ValueError("Unknown mode: {}".format(mode))

    print("Loading and preprocessing data. This can take a while.")
    ext = file.split('.')[-1]
    if ext == 'zip':
        df = pd.read_csv(file, compression='zip')
    elif ext == 'gzip':
        df = pd.read_csv(file, compression='gzip')
    else: 
        df = pd.read_csv(file)
    print("Dataframe loaded.")
    
    print("Converting to NDArray.")
    x = df.iloc[:, :-2].values
    p300_idxs = np.where(df["P300"].values == 1)[0]

    if mode == 'one-hot':
        y = df.iloc[:, -2:].values # Last two columns are one-hot [P300, NP300]
    elif mode == 'int':
        y = np.zeros(x.shape[0], dtype=int)
        y[p300_idxs] = 1
    del df

    print("Reshaping as 4D image tensor.")
    # Reshape each sample into 2D
    x = x.reshape([x.shape[0], -1, block_mat_len])

    # Keras convolutional layers expect a 4D input.
    # Wrap values with single (grayscale) channel.
    x = np.expand_dims(x, axis=-1)

    print("Binning data into pixel values and normalizing.")
    # Bin into 0-255 and then scale to 0-1.
    x = img_scale(x, use_cutout=use_cutout)

    num_p300s = p300_idxs.shape[0]
    imbal_factor = (y.shape[0] - num_p300s) / num_p300s

    print("Data loaded.")

    return x, y, imbal_factor

def int_to_one_hot(labels):
    # [P300, NP300] for consistency with csv files
    return np.array([[i, int(not i)] for i in labels])


class HDF5DataHandler:
    """Class to manage generating batches from
    large (larger than tens of GB) HDF5 files
    of upsampled EEG data.
    """

    def __init__(self, file, block_mat_len=18, imbal_factor=4, batch_size=32):
        """Constructor.
        
        Arguments:
            file {HDF5 file} -- Expects two datasets of matching
            length: 'data' and 'labels'. Each label should be a
            single int: 1 for P300, 0 for NP300.

            block_mat_len {int} -- Edge dimension of each
            of the block matrices that form the expanded
            covariance matrix features. (default: {18})

            imbal_factor {int} -- Ratio of NP300s to P300s in
            the data. (default: {4})

            batch_size {int} -- Number of samples to generate
            per minibatch. (default: {32})
        """

        self.f = h5py.File(file, 'r')
        self.block_mat_len = block_mat_len
        self.imbal_factor = imbal_factor
        self.batch_size = batch_size
        self.data = self.f['data']
        self.labels = self.f['labels']
        self.data_len = self.data.len()
        self.step_max = self.data_len // self.batch_size
        self.idxs = list(range(self.f['data'].len()))
        self.step_count = 0

    @staticmethod
    def joint_shuffle(x, y):
        """Shuffle data and corresponding labels
        in place, preserving correspondence."""
        rand_state = np.random.get_state()
        np.random.shuffle(x)
        np.random.set_state(rand_state)
        np.random.shuffle(y)

    @staticmethod
    def process_hdf5_data(x, y, block_mat_len, use_cutout=False):
        """Slice the data before passing it
        to this function.

        Usage for minibatches:
        x = f['data'][samples]
        y = f['data'][samples]

        Usage for validation data (must be
        relatively small to fit in RAM...a few GB
        at most):
        x = f['data'][:]
        y = f['data'][:]
        """

        # Shuffle the data in place
        HDF5DataHandler.joint_shuffle(x, y)

        # Reshape each sample into 2D
        x = x.reshape([x.shape[0], -1, block_mat_len])

        # Keras convolutional layers expect a 4D input.
        # Wrap values with single (grayscale) channel.
        x = np.expand_dims(x, axis=-1)

        # Scale to 0-1.
        x = img_scale(x, use_cutout=use_cutout)

        # Make labels one-hot
        y = int_to_one_hot(y)

        return x, y

    def shuffle_idxs(self):
        print("Shuffling data indices...")
        np.random.shuffle(self.idxs)
        print("Indices shuffled.")

    def datagen(self, use_cutout=False):
        """Generator. Fetches a minibatch from the HDF5
        file and preprocesses the batch."""

        self.shuffle_idxs() # First time
        batch_size = self.batch_size

        while True:
            """step_count is reset to 0 at start of every epoch
            and incremented by one at start of every batch."""
            step_count = self.step_count
            last_batch = False
            if step_count < self.step_max: # Full-size batches
                samps = self.idxs[batch_size * step_count:
                                  batch_size * (step_count + 1)]
            else: # Last batch may not be full-size.
                samps = self.idxs[batch_size * step_count:]
                last_batch = True
            
            # Fetch data from HDF5 file (indices must be sorted).
            samps.sort()
            x = self.data[samps]
            y = self.labels[samps]

            # Preprocess data, shuffle, and format labels.
            x, y = self.process_hdf5_data(x, y, self.block_mat_len, use_cutout=use_cutout)

            # Increment step count
            self.step_count += 1
            
            if last_batch: # Shuffle idxs and reset count.
                self.shuffle_idxs()
                self.step_count = 0

            yield (x, y)
