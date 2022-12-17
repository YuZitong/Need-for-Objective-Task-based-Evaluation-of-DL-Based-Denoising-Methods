###############################################################################
#                            Import Libraries
###############################################################################
import numpy as np
from PIL import Image
import os
import sys

import tensorflow as tf

from keras import backend as K

K.set_image_data_format('channels_last')
K.tensorflow_backend._get_available_gpus()

from keras.models import Model,load_model
from keras.layers import Input, Conv3D, Conv3DTranspose, MaxPooling3D, Dropout, BatchNormalization, concatenate, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import Constant
from keras.models import load_model
from keras.utils import multi_gpu_model
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau
# from custom_classes_02 import DataGenerator

import h5py

from sklearn.model_selection import KFold

config = tf.ConfigProto( device_count = {'GPU': 2} )
sess = tf.Session(config=config)
K.tensorflow_backend.set_session(sess)

###############################################################################
#                            Network Architecture
###############################################################################

#  Define loss function
def loss_fn(y_true, y_pred):
    data_fidelity = tf.reshape(y_true, shape=[-1, 128*128*128]) - tf.reshape(y_pred, shape=[-1, 128*128*128])
    data_fidelity = tf.reduce_mean(tf.square(data_fidelity))
    return data_fidelity

# Repeating layers throughout the network 
def add_common_layers(filters, kernelsize,std, layer, bias_ct=0.03, leaky_alpha=0.01, drop_prob=0.1):
    if std == 2:
        pad = 'valid'
    else:
        pad = 'same'
    layer = Conv3D(filters, kernel_size=kernelsize, # num. of filters and kernel size 
                   strides=std,
                   padding=pad,
                   use_bias=True,
                   kernel_initializer='glorot_normal', # Xavier init
                   bias_initializer=Constant(value=bias_ct),
                   kernel_regularizer=l2(0.1),
                   bias_regularizer=l2(0.1))(layer)
    layer = LeakyReLU(alpha=leaky_alpha)(layer) # activation func. 
    layer = Dropout(drop_prob)(layer) 
    return layer


def get_cnn(num_layers):
    # num_layers - number of slices
    # This model has skip connections in place, here we use element-wise addition.
    # Define Convolutional Neural Network
    # Input shape 
    input = Input(shape=(128,128,num_layers,1))
    # Conv1
    conv1 = add_common_layers(16,(3, 3, 3),1,layer=input)
    x = add_common_layers(16,(2, 2, 2),2,conv1)
    # Conv2
    conv2 = add_common_layers(32,(3, 3, 3),1, x)
    x = add_common_layers(32,(2, 2, 2),2,conv2)
    # Conv3
    conv3 = add_common_layers(64,(3, 3, 3),1, x)
    x = add_common_layers(64,(2, 2, 2),2,conv3)
    # Transposed Convolution (upsampling)
    x = Conv3DTranspose(64, (3,3,3), strides=(2,2,2), padding='same', use_bias=True, kernel_initializer='glorot_normal', bias_initializer=Constant(value=0.03),kernel_regularizer=l2(0.1),bias_regularizer=l2(0.1))(x)
    x = LeakyReLU(alpha=0.01)(x)
    # Conv-3
    x = Add()([x, conv3])
    x = add_common_layers(64,(3, 3, 3),1,x)
    # Transposed Convolution (upsampling)
    x = Conv3DTranspose(32, (2,2,2), strides=(2,2,2), padding='same', use_bias=True, kernel_initializer='glorot_normal', bias_initializer=Constant(value=0.03),kernel_regularizer=l2(0.1),bias_regularizer=l2(0.1))(x)
    x = LeakyReLU(alpha=0.01)(x)
    # Conv-2
    x = Add()([x, conv2])
    x = add_common_layers(32,(3, 3, 3), 1, x)
    # Transposed Convolution (upsampling)
    x = Conv3DTranspose(16, (2,2,2), strides=(2,2,2), padding='same', use_bias=True, kernel_initializer='glorot_normal', bias_initializer=Constant(value=0.03),kernel_regularizer=l2(0.1),bias_regularizer=l2(0.1))(x)
    x = LeakyReLU(alpha=0.01)(x)
    # Conv-1
    x = Add()([x, conv1])
    x = add_common_layers(16,(3, 3, 3),1, x)
    # Conv_out
    x = Conv3D(1, (3,3,3), strides=1, padding='same', use_bias=True, kernel_initializer='glorot_normal', bias_initializer=Constant(value=0.03),kernel_regularizer=l2(0.1),bias_regularizer=l2(0.1))(x)
    x = LeakyReLU(alpha=0.00)(x)#negative to zero
    output = x
    model = Model(inputs=[input], outputs=[output])
    return model


###############################################################################
#                            	    Training
###############################################################################
params = {'dim': (128,128,128),
          'batch_size': 32,
          'n_channels': 1,
          'shuffle': True}

print('Loading the training data...')
num_train = #800
dose_lvl = sys.argv[1] #low_02
num_epochs = int(sys.argv[2])
print('Current dose level: ', dose_lvl)

low_dose_name = []
normal_dose_name = []
rec_loc = #'./npy_data/train/'

defect_folder_list = #['1_0.5','2_0.25','4_0.5','5_0.25']

for defect_type in defect_folder_list:
    for patient_idx in np.arange(1,101):
        # low_dose_name.append(rec_loc+'/defect/'+dose_lvl+'/'+defect_type+'_'+str(patient_idx)+'.npy')
        # low_dose_name.append(rec_loc+'/healthy/'+dose_lvl+'/'+defect_type+'_'+str(patient_idx)+'.npy')
        # normal_dose_name.append(rec_loc+'/defect/1/'+defect_type+'_'+str(patient_idx)+'.npy')
        # normal_dose_name.append(rec_loc+'/healthy/1/'+defect_type+'_'+str(patient_idx)+'.npy')

X = np.zeros((num_train, *params['dim'], params['n_channels']))
Y = np.zeros((num_train, *params['dim'], params['n_channels']))

for i in np.arange(1,num_train+1):
    X[i-1,:,:,7:121,0] = np.load(low_dose_name[i-1])*1000
    Y[i-1,:,:,7:121,0] = np.load(normal_dose_name[i-1])*1000

model = get_cnn(128)
model.summary()
parallel_model = multi_gpu_model(model, 2)
parallel_model.compile(loss=loss_fn, optimizer='adam')

print('Training...')
history = parallel_model.fit(x=X,
                            y=Y,
                            batch_size=params['batch_size'],
                            verbose=1,
                            epochs=num_epochs,
                            shuffle=True)
train_loss = history.history['loss']
np.save('train_loss_epochs_'+str(num_epochs)+'_'+dose_lvl+'.npy',train_loss)

model.save('epochs_'+str(num_epochs)+'_'+dose_lvl+'.hdf5')