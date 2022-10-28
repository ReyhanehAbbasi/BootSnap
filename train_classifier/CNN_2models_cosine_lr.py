from math import pi, cos, floor
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Dense, Flatten, Convolution2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model

# snapshot ensemble with custom learning rate schedule
class SnapshotEnsemble(Callback):
    # constructor
    def __init__(self, n_epochs, n_cycles, lrate_max,kk,jj, verbose=0):
        self.epochs = n_epochs
        self.cycles = n_cycles
        self.lr_max = lrate_max
        self.fold_n = kk
        self.rep=jj
        self.lrates = list()
    # calculate learning rate for epoch
    def cosine_annealing(self, epoch, n_epochs, n_cycles, lrate_max):
        epochs_per_cycle = floor(n_epochs/n_cycles)
        cos_inner = (pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
        return lrate_max/2 * (cos(cos_inner) + 1)
    # calculate and set learning rate at the start of the epoch
    def on_epoch_begin(self, epoch, logs={}):
        # calculate learning rate
        lr = self.cosine_annealing(epoch, self.epochs, self.cycles, self.lr_max)
        # set learning rate
        backend.set_value(self.model.optimizer.lr, lr)
        # log value
        self.lrates.append(lr)
    # save models at the end of each cycle
    def on_epoch_end(self, epoch, logs={}):
        # check if we can save model
        epochs_per_cycle = floor(self.epochs / self.cycles)
        if epoch != 0 and (epoch + 1) % epochs_per_cycle == 0:
            # save model to file
            filename = "ELU_CNN_original_fold"+str(self.fold_n)+'_rep'+str(self.rep)+"_snapshot_model_%d.h5" % int((epoch + 1) / epochs_per_cycle)
            self.model.save(filename)
            print('>saved snapshot %s, epoch %d' % (filename, epoch))





def CNN_model2(inp,num_classes):
    outs = [] # the list of ensemble outputs
    kernel_size=(3,18)
    strides=1
    l2_lambda = 0.0001
    pool_size=2
    conv_depth = 16
    hidden_size = 128 
    drop_prob_2 = 0.5
    conv_1 = Conv2D(2*conv_depth,
                           kernel_size=kernel_size,
                           padding='same',
                           strides=2,
                           kernel_initializer='he_normal',
                           kernel_regularizer=l2(l2_lambda)
                           )(inp)
    bn1 = BatchNormalization()(conv_1)    
    conv_1=Activation('elu')(bn1)
    pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1)
    conv_2 = Conv2D(2*conv_depth, (3,3),
                           padding='same',
                           kernel_initializer='he_uniform',
                           kernel_regularizer=l2(l2_lambda)
                           )(pool_1)
    bn2 = BatchNormalization()(conv_2)    
    conv_2=Activation('elu')(bn2)
    pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
    # adding conv layer
    conv_3 = Conv2D(2*conv_depth, (3,3),
                           padding='same',
                           kernel_initializer='he_uniform',
                           kernel_regularizer=l2(l2_lambda)
                           )(pool_2)
    bn3 = BatchNormalization()(conv_3)    
    conv_3=Activation('elu')(bn3)
    pool_3 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_3)
    conv_4 = Conv2D(3*conv_depth, (3,3),
                           padding='same',
                           kernel_initializer='he_uniform',
                           kernel_regularizer=l2(l2_lambda)
                           )(pool_3)
    bn4 = BatchNormalization()(conv_4)    
    conv_4=Activation('elu')(bn4)
    pool_4 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
    conv_5 = Conv2D(3*conv_depth, (3,3),
                           padding='same',
                           kernel_initializer='he_uniform',
                           kernel_regularizer=l2(l2_lambda)
                           )(pool_4)
    bn5 = BatchNormalization()(conv_5)
    conv_5=Activation('elu')(bn5)
    pool_5 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_5)
    # end of  conv layers
    flat = Flatten()(pool_5)
    hidden = Dense(int(hidden_size),
                   kernel_initializer='he_uniform',
                   kernel_regularizer=l2(l2_lambda))(flat) #
    hidden = BatchNormalization()(hidden)
    hidden=Activation('elu')(hidden)
    hidden2 = Dense(int(hidden_size),
                   kernel_initializer='he_uniform',
                   kernel_regularizer=l2(l2_lambda))(hidden) # Hidden ReLU layer # for relu
    #activation=act)(flat) # Hidden ReLU layer
    hidden2 = BatchNormalization()(hidden2)
    hidden2=Activation('elu')(hidden2)
    drop2 = Dropout(drop_prob_2)(hidden2)
    outs.append(Dense(num_classes,
                      kernel_initializer='glorot_uniform',
                      kernel_regularizer=l2(l2_lambda),
                      activation='softmax')(drop2)) # Output softmax layer
    model = Model(inputs=inp, outputs=outs) # To define a model, just specify its input and output layers
    return model

