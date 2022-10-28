# load required libraries

from __future__ import print_function
import keras
import tensorflow
from tensorflow.keras.layers import Input
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, precision_score, recall_score,\
    f1_score, cohen_kappa_score,accuracy_score,hamming_loss
from sklearn.preprocessing import LabelBinarizer
import tensorflow_addons as tfa
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gc
import scipy.io as sio
import math
import bootstrap_sampling
import CNN_2models_cosine_lr
import sys


gc.collect()
path = os.getcwd()

# Define the row and column size of input image
handles_repertoire_unit_size_seconds = 200
handles_frame_shift_ms = 5e-04
NbPatternFrames=handles_repertoire_unit_size_seconds/handles_frame_shift_ms*1e-3  # ms divided by frame_shift
NbChannels = 64
IMAGE_DIMS = (NbChannels, math.floor(NbPatternFrames+1), 1)
height, width, depth = NbChannels, math.floor(NbPatternFrames+1), 1 # MNIST images are 28x28 and greyscale
num_classes = 12

# load data
path1='/home/rabbasi/data/preparing_data_clustering_code/undersampled_data_implementations/' \
      'undersampled_classification/5_gt_20_120/original'
mat_contents_gt=sio.loadmat(path1+'/original_gt_data_with_h.mat')
data_gt=mat_contents_gt['id']
data_main=data_gt

#load labels
man_lab= np.load(path1+'/original_final_labels_with_h.npz')
man_labf=man_lab['arr_0'][()]

# load train, validatioon, and test ids
mat_contents_kfold_original_train_ids=sio.loadmat(path1+'/kfold_original_train_ids.mat')
kfold_original_train_ids=mat_contents_kfold_original_train_ids['id']
mat_contents_kfold_test_ids=sio.loadmat(path1+'/kfold_test_ids.mat')
kfold_test_ids=mat_contents_kfold_test_ids['id']
mat_contents_kfold_val_ids=sio.loadmat(path1+'/kfold_val_ids.mat')
kfold_val_ids=mat_contents_kfold_val_ids['id']


data_main= data_main.astype('float32')
data_main = data_main.reshape(data_main.shape[0], height, width, depth)

# preparing output for classifier
lb = LabelBinarizer()
labels = lb.fit_transform(man_labf)
inp = Input(shape=(height, width, depth)) # N.B. TensorFlow back-end expects channel dimension last

# If the following function changes, the following two lines must be uncommented
#importlib.reload(sys.modules['bootstrap_sampling2'])
#importlib.reload(sys.modules['CNN_2models_cosine_lr'])

# training the classifier for number of bootstrap * number of folds
for kk in range(8):
    l2_lambda = 0.0001
    # train
    for jj in range(10):
        print(kk)
        print(jj)
        X_train = data_main[kfold_original_train_ids[0][kk][0],]
        Y_train_122 = labels[kfold_original_train_ids[0][kk][0],]
        X_train_122 = X_train.astype('float32')
        #X_train_122.shape
        indices_bootstraped=bootstrap_sampling.bootstrapped_sampled(Y_train_122,lb)
        Y_train_12=Y_train_122[indices_bootstraped,]
        X_train_12=X_train_122[indices_bootstraped,]
        # validation
        X_val = data_main[kfold_val_ids[kk],]
        Y_val_12 = labels[kfold_val_ids[kk],]
        X_val_12 = X_val.astype('float32')
        # prepare data for model
        x_train=X_train_12
        y_train=Y_train_12
        x_val=X_val_12
        y_val=Y_val_12
        num_classes = 12
        input_shape = x_train.shape[1:]
        ########
        #  define weights for network
        y_integers = np.argmax(y_train,axis=1)
        class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
        d_class_weights = dict(enumerate(class_weights))
        # network parameteres
        batch_size = 32
        epochs = 200
        data_augmentation = True
        model = CNN_2models_cosine_lr.CNN_model2(inp,num_classes)
        sgd = optimizers.SGD( momentum=0.9, nesterov=False)
        model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=[tfa.metrics.F1Score(num_classes=num_classes,average='macro')]) #['accuracy']
        #model.summary()
        # Prepare callbacks for model saving and for learning rate adjustment.
        n_cycles = epochs / 40
        ca = CNN_2models_cosine_lr.SnapshotEnsemble(epochs, n_cycles, 0.01,kk,jj)
        # Run training, with or without data augmentation.
        if not data_augmentation:
            print('Not using data augmentation.')
            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(x_val, y_val),
                      shuffle=True,
                      callbacks=callbacks)
        else:
            print('Using real-time data augmentation.')
            # This will do pre-processing and realtime data augmentation:
            datagen = ImageDataGenerator(
                # set input mean to 0 over the dataset
                featurewise_center=False,
                # set each sample mean to 0
                samplewise_center=False,
                # divide inputs by std of dataset
                featurewise_std_normalization=False,
                # divide each input by its std
                samplewise_std_normalization=False,
                # apply ZCA whitening
                zca_whitening=False,
                # epsilon for ZCA whitening
                zca_epsilon=1e-06,
                # randomly rotate images in the range (deg 0 to 180)
                rotation_range=0,
                # randomly shift images horizontally
                width_shift_range=0.1,
                # randomly shift images vertically
                height_shift_range=0.1,
                # set range for random shear
                shear_range=0.,
                # set range for random zoom
                zoom_range=0, # it was zero, changed by RA to 1
                # set range for random channel shifts
                channel_shift_range=0.,
                # set mode for filling points outside the input boundaries
                fill_mode='nearest',
                # value used for fill_mode = "constant"
                cval=0.,
                # randomly flip images
                horizontal_flip=False, # it was True,changed by RA to False
                # randomly flip images
                vertical_flip=False,
                # set rescaling factor (applied before any other transformation)
                rescale=None,
                # set function that will be applied on each input
                preprocessing_function=None,
                # image data format, either "channels_first" or "channels_last"
                data_format=None,
                # fraction of images reserved for validation (strictly between 0 and 1)
                validation_split=0.0)
            # Compute quantities required for featurewise normalization
            #  (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)
        model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                  validation_data=(x_val, y_val),
                  epochs=epochs, verbose=2, workers=2,
                  callbacks=[ca],steps_per_epoch=x_train.shape[0]/batch_size
                  ,class_weight=d_class_weights) #,shuffle=True



# plot model history
plt.clf()
plt.plot(model.history.history['acc'])
plt.plot(model.history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'validation'], loc='upper left')
plt.tight_layout()
plt.savefig("accuracy_val_train.pdf")

# Plot training & validation loss values
plt.clf()
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'validation'], loc='upper left')
plt.tight_layout()
plt.savefig("loss_val_train.pdf")
