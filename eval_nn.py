import os
import numpy as np
import keras
import nn_utils as nnut
import h5py
import matplotlib.pyplot as plt

block_mat_len = 30  # Edge size of each of the square submatrices
data_folder = '/home/ubuntu/Work/keyboard_nn/data'
model_dir = '/home/ubuntu/Work/keyboard_nn/saved_models'
# subj_suffix = 'S63' # To identify data used to train
boot_n = 5 # Number of times to split the data
boot_split = 0.8 # Fraction of the data to use
cuts = 5 # Number of cuts of each image to average for prediction

# Test files
test_f_1seq = 'z_1seq_Through_Subject_63_validation.hdf5'
test_f_2seq = None

#for seqs in ['1seq', '2seq']:
for seqs in ['1seq']:
    if seqs == '1seq':
        test_f_n = os.path.join(data_folder, test_f_1seq)
    elif seqs == '2seq':
        test_f_n = os.path.join(data_folder, test_f_2seq)
    f_test = h5py.File(test_f_n, 'r')

    # Prepare test data
    print("Loading test data...")
    x_test = f_test['data'][:]
    y_test = f_test['labels'][:]
    print("Preparing test data...")
    x_test, y_test = nnut.HDF5DataHandler.process_hdf5_data(x_test,
                                                            y_test,
                                                            block_mat_len=block_mat_len)

    # Test old model on new test data
    for model_type in ['ZRN-16-1-D', 'ZRN-16-8-D', 'XRN-28-8-D']:
        for use_cutout in ['C']:
            
            # Load the model
            model_name = 'model_' + model_type + '-' + use_cutout + '.hdf5'
            m = os.path.join(model_dir, model_name)
            model = keras.models.load_model(m)
            print("Loaded model {} to evaluate on test data.".format(model_name))

            # Evaluate model on loaded test data.
            print("Evaluating model by bootstrapping {} times with a {} split.".format(boot_n, boot_split))
            met = nnut.boot_ml_metrics(model, x_test, y_test, n=boot_n, split=boot_split, cuts=cuts)

            print("Metrics for {} | {} | {}:".format(seqs, model_type, use_cutout))
            for k, v in met.items():
                print("Summary for metric {}".format(k))
                median = np.median(v)
                mean = np.mean(v)
                std_dev = np.std(v)
                maxi = np.max(v)
                mini = np.min(v)
                p5 = np.percentile(v, 5)
                p25 = np.percentile(v, 25)
                p75 = np.percentile(v, 75)
                p95 = np.percentile(v, 95)
                print("Median: {}".format(median))
                print("Mean: {}".format(mean))
                print("Std. Dev.: {}".format(std_dev))
                print("Max: {}".format(maxi))
                print("Min: {}".format(mini))
                print("5th percentile: {}".format(p5))
                print("25th percentile: {}".format(p25))
                print("75th percentile: {}".format(p75))
                print("95th percentile: {}".format(p95))
                print("\n\n")

"""
Plot metrics.
"""
"""
labels = ['P300 Precision', 'P300 Recall', 'Balanced Accuracy']
data = [mets['precision'], mets['recall'], mets['bal_acc']]
plt.style.use('ggplot')
bp = plt.boxplot(data, whis=[5, 95],
                 labels=labels,
                 showmeans=True,
                 meanline=True)

#Change linewidths
for box in bp['boxes']:
    box.set(linewidth=3)
for cap in bp['caps']:
    cap.set(linewidth=3)
for whisker in bp['whiskers']:
    whisker.set(linewidth=3)
for mean in bp['means']:
    mean.set(linewidth=3)
for median in bp['medians']:
    median.set(linewidth=3)

plt.tick_params(labelsize=36)
plt.title('CNN Metrics, Two-Sequence ERP Data', fontsize=48)
plt.show()
"""
