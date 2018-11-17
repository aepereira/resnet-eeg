from __future__ import division
import os
import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, Dropout, GlobalAveragePooling2D
from keras.layers import SpatialDropout2D
from keras.layers import Input, Add
from time import strftime
import nn_utils as nnut
import h5py

"""
Config
"""
# Data files
data_folder = '/home/ubuntu/Work/keyboard_nn/data'
training_f_n = os.path.join(data_folder, 'z_1seq_Through_Subject_63_training.hdf5')
val_f_n = os.path.join(data_folder, 'z_1seq_Through_Subject_63_validation.hdf5')

# h5py object for validation
f_val = h5py.File(val_f_n, 'r')

# Depend on the data.
block_mat_len = 30 # Number of columns per sample. 30 for decimated and padded raw features.
imbal_factor = 4 # Ratio of NP300s to P300s

# Keras model training config
batch_size = 128  # Have enough P300s, power of 2
epochs = 200
use_conv_dropout = True # Whether to use dropout in between convolutions
drop_frac = 0.3 # Fraction to drop
k = 1 # Width parameter for WideResNet
d = 28 # Depth of network. Must be 6n + 4.
use_cutout = True # Whether to use cutout in the generator

# Name architecture using experiment convention
arch = 'XRN-{}-{}-'.format(d, k)
if use_conv_dropout:
    arch += 'D-'
else:
    arch += 'ND-'
if use_cutout:
    arch += 'C'
else:
    arch += 'NC'

# Save directory
save_dir = os.path.join(os.getcwd(), 'saved_models')

# Instantiate HDF5DataHandler object with training file.
data_handler = nnut.HDF5DataHandler(training_f_n,
                                    block_mat_len=block_mat_len,
                                    imbal_factor=4,
                                    batch_size=batch_size)
datagen = data_handler.datagen # Training data generator
training_samples = data_handler.data_len

"""
# Prepare training data. Use above if not enough RAM.
f_train = h5py.File(training_f_n)
print("Loading training data...")
training_samples = f_train['data'].len()
x_train = f_train['data'][:]
y_train = f_train['labels'][:]
print("Preparing training data...")
x_train, y_train = nnut.HDF5DataHandler.process_hdf5_data(x_train,
                                                          y_train,
                                                          block_mat_len=block_mat_len)
print("Training data ready.")
"""

# Prepare validation data
print("Loading validation data...")
x_validation = f_val['data'][:]
y_validation = f_val['labels'][:]
print("Preparing validation data...")
x_validation, y_validation = nnut.HDF5DataHandler.process_hdf5_data(x_validation,
                                                                    y_validation,
                                                                    block_mat_len=block_mat_len)
print("Validation data ready.")


"""
Wide ResNet WRN-16-k architecture
"""
n = (d - 4) // 6 # Add error-checking later

x_in = Input(shape=x_validation.shape[1:], name='Input')

x = Conv2D(16, (3, 3), padding='same', use_bias=False, name='Conv_Base')(x_in)

# Residual block A
# First activation before splitting, per "Identity Mappings" paper by He et al.
x = BatchNormalization(name='BN_A1')(x)
x = Activation('relu', name='ReLU_A1')(x)
# Skip with convolution to match feature map dims
x_ = Conv2D(16 * k, (1, 1), padding='same', use_bias=False, name='SkipConv_A1')(x)
x = Conv2D(16 * k, (3, 3), padding='same', use_bias=False, name='Conv_A1')(x)
x = BatchNormalization(name='BN_A2')(x)
x = Activation('relu', name='ReLU_A2')(x)
if use_conv_dropout:
    x = SpatialDropout2D(drop_frac, name='Dropout_A1')(x)
x = Conv2D(16 * k, (3, 3), padding='same', use_bias=False, name='Conv_A2')(x)
x = Add(name='Add_Skip_A1')([x, x_])
for i in range(n - 1):
    x_ = x # Skip connection
    x = BatchNormalization(name='BN_A3_{}'.format(i))(x)
    x = Activation('relu', name='ReLU_A3_{}'.format(i))(x)
    if use_conv_dropout:
        x = SpatialDropout2D(drop_frac, name='Dropout_A2_{}'.format(i))(x)
    x = Conv2D(16 * k, (3, 3), padding='same', use_bias=False, name='Conv_A3_{}'.format(i))(x)
    x = BatchNormalization(name='BN_A4_{}'.format(i))(x)
    x = Activation('relu', name='ReLU_A4_{}'.format(i))(x)
    if use_conv_dropout:
        x = SpatialDropout2D(drop_frac, name='Dropout_A3_{}'.format(i))(x)
    x = Conv2D(16 * k, (3, 3), padding='same', use_bias=False, name='Conv_A5_{}'.format(i))(x)
    x = Add(name='Add_Skip_A2_{}'.format(i))([x, x_])

# Residual block B
x = BatchNormalization(name='BN_B1')(x)
x = Activation('relu', name='ReLU_B1')(x)
# 2x2 stride skip convolution to match feature map dims
x_ = Conv2D(32 * k, (1, 1), strides=(2, 2), padding='same',
            use_bias=False, name='SkipConv_B1')(x)
# 2x2 stride to downsample
x = Conv2D(32 * k, (3, 3), strides=(2, 2), padding='same',
           use_bias=False, name='Conv_B1')(x)
x = BatchNormalization(name='BN_B2')(x)
x = Activation('relu', name='ReLU_B2')(x)
if use_conv_dropout:
    x = SpatialDropout2D(drop_frac, name='Dropout_B1')(x)
x = Conv2D(32 * k, (3, 3), padding='same', use_bias=False, name='Conv_B2')(x)
x = Add(name='Add_Skip_B1')([x, x_])
for i in range(n - 1):
    x_ = x  # Skip connection
    x = BatchNormalization(name='BN_B3_{}'.format(i))(x)
    x = Activation('relu', name='ReLU_B3_{}'.format(i))(x)
    if use_conv_dropout:
        x = SpatialDropout2D(drop_frac, name='Dropout_B2_{}'.format(i))(x)
    x = Conv2D(32 * k, (3, 3), padding='same', use_bias=False, name='Conv_B3_{}'.format(i))(x)
    x = BatchNormalization(name='BN_B4_{}'.format(i))(x)
    x = Activation('relu', name='ReLU_B4_{}'.format(i))(x)
    if use_conv_dropout:
        x = SpatialDropout2D(drop_frac, name='Dropout_B3_{}'.format(i))(x)
    x = Conv2D(32 * k, (3, 3), padding='same', use_bias=False, name='Conv_B5_{}'.format(i))(x)
    x = Add(name='Add_Skip_B2_{}'.format(i))([x, x_])

# Residual block C
x = BatchNormalization(name='BN_C1')(x)
x = Activation('relu', name='ReLU_C1')(x)
# 2x2 stride skip convolution to match feature map dims
x_ = Conv2D(64 * k, (1, 1), strides=(2, 2), padding='same',
            use_bias=False, name='SkipConv_C1')(x)
# 2x2 stride to downsample
x = Conv2D(64 * k, (3, 3), strides=(2, 2), padding='same',
           use_bias=False, name='Conv_C1')(x)
x = BatchNormalization(name='BN_C2')(x)
x = Activation('relu', name='ReLU_C2')(x)
if use_conv_dropout:
    x = SpatialDropout2D(drop_frac, name='Dropout_C1')(x)
x = Conv2D(64 * k, (3, 3), padding='same', use_bias=False, name='Conv_C2')(x)
x = Add(name='Add_Skip_C1')([x, x_])
for i in range(n - 1):
    x_ = x  # Skip connection
    x = BatchNormalization(name='BN_C3_{}'.format(i))(x)
    x = Activation('relu', name='ReLU_C3_{}'.format(i))(x)
    if use_conv_dropout:
        x = SpatialDropout2D(drop_frac, name='Dropout_C2_{}'.format(i))(x)
    x = Conv2D(64 * k, (3, 3), padding='same', use_bias=False, name='Conv_C3_{}'.format(i))(x)
    x = BatchNormalization(name='BN_C4_{}'.format(i))(x)
    x = Activation('relu', name='ReLU_C4_{}'.format(i))(x)
    if use_conv_dropout:
        x = SpatialDropout2D(drop_frac, name='Dropout_C3_{}'.format(i))(x)
    x = Conv2D(64 * k, (3, 3), padding='same', use_bias=False, name='Conv_C5_{}'.format(i))(x)
    x = Add(name='Add_Skip_C2_{}'.format(i))([x, x_])

# Top of network
x = BatchNormalization(name='BN_Top')(x)
x = Activation('relu', name='ReLU_Top')(x)
x = GlobalAveragePooling2D(name='Global_Average')(x)
x = Dense(2, name='Dense_Top')(x)
y = Activation('softmax', name='Softmax_Top')(x)

model = Model(inputs=x_in, outputs=y)

# Add weight decay per WideResNet implementation
weight_decay = 0.0005
for layer in model.layers:
    if hasattr(layer, 'kernel_regularizer'):
        layer.kernel_regularizer = keras.regularizers.l2(weight_decay)

# Learning rate
optimizer = keras.optimizers.SGD(
    lr=0.1, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# Learning rate schedule callback
def scheduler(epoch_idx, lr):
    """Same schedule (for 200 epochs) as
    WideResNet paper implementation.

    But 2 cycles.
    """
    new_lr = lr
    if (epoch_idx == 60 or epoch_idx == 120 or epoch_idx == 160
        or epoch_idx == 260 or epoch_idx == 320 or epoch_idx == 360):
        new_lr *= 0.2
    """
    if epoch_idx == 200:
        new_lr = 0.1
    """
    return new_lr

lr_sched = keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

# Prepare directory for model checkpoint.
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

# Checkpoint callback
#chk_fname = os.path.join(save_dir, "model_tmp.hdf5")
#checkpointer = keras.callbacks.ModelCheckpoint(
#    filepath=chk_fname, verbose=1, save_best_only=True, save_weights_only=False)

# Prepare log directory
log_dir = os.path.join(os.getcwd(), 'logs')
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

# Tensorboard callback
# l_time = strftime("%Y-%m-%d_%H-%M") # Time of run.
tb = keras.callbacks.TensorBoard(log_dir=log_dir + "/{}".format(arch))

# Model summary
print("Model summary:")
model.summary()
print("TRAINING MODEL: {}".format(arch))

# If using generator
steps_per_epoch = data_handler.data_len / batch_size
model.fit_generator(datagen(use_cutout=use_cutout),
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    validation_data=(x_validation, y_validation),
                    shuffle=False, # Our generator handles shuffling internally
                    class_weight={0: imbal_factor, # 0 is P300 index in one-hot mode
                                  1: 1},
                    callbacks=[tb, lr_sched]) # Not checkpointing

"""
# If loading training data into RAM
model.fit(x_train, y_train,
          batch_size==batch_size,
          epochs=epochs,
          validation_data=(x_validation, y_validation),
          shuffle=True,
          class_weight={0: imbal_factor,  # 0 is P300 index in one-hot mode
                        1: 1},
          callbacks=[checkpointer, tb])
"""

# Evaluate final model.
scores = model.evaluate(x_validation, y_validation, verbose=1)
print("Evaluating final model...")
print('Validation loss:', scores[0])
print('Validation accuracy:', scores[1])
nnut.ml_metrics(model, x_validation, y_validation, verbose=True)

#Save final model
# f_time = l_time  # Time of file is time of run, to match logs.
final_fname = os.path.join(save_dir, "model_" + arch + "_final.hdf5")
print("Saving final model from run...")
model.save(final_fname)


# Load and evaluate best model from run.
#model = keras.models.load_model(chk_fname)
#scores = model.evaluate(x_validation, y_validation, verbose=1)
#print("Evaluating best model from run...")
#print('Validation loss:', scores[0])
#print('Validation accuracy:', scores[1])
#nnut.ml_metrics(model, x_validation, y_validation, verbose=True)

# Save best model and graph of its structure.
#best_fname = os.path.join(save_dir, "model_" + f_time + "_best.hdf5")
#print("Saving best model from run...")
#os.rename(chk_fname, best_fname)
#print("Model saved to {}.".format(final_fname))
graph_fname = os.path.join(save_dir, "model_" + arch + ".png")
keras.utils.plot_model(model, to_file=graph_fname, show_shapes=True)
print("Graph of model saved to {}.".format(graph_fname))
