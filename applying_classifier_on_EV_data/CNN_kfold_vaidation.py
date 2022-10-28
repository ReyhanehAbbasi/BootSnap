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

from tensorflow_addons.metrics import F1Score
from tensorflow.keras.models import load_model

gc.collect() # equal to clear all
path = os.getcwd() # current working directory
path_model='/home/rabbasi/data/preparing_data_clustering_code/undersampled_data_implementations/undersampled_classification/5_gt_20_120/original/CNN/bootstraped/Best_model'
# initial parmeteres for data
handles_repertoire_unit_size_seconds = 200
handles_frame_shift_ms = 5e-04
NbPatternFrames=handles_repertoire_unit_size_seconds/handles_frame_shift_ms*1e-3  # ms divided by frame_shift
NbChannels = 64  
NbChannels1 = 251 


INIT_LR = 1e-3
IMAGE_DIMS = (NbChannels, math.floor(NbPatternFrames+1), 1) # the same in the net, should change that # 64*401

# load data

mat_contents_gt=sio.loadmat(path+'/input_clustering_data_3_indi.mat')
data_gt=mat_contents_gt['id']
data_main=data_gt.transpose()

# labels
man_lab= np.load(path+'/man_lab_f_3_indi.npz')
type_man=man_lab['arr_0'][()]

height, width, depth = NbChannels, math.floor(NbPatternFrames+1), 1 
num_classes = 12 # 


data_main= data_main.astype('float32')
data_main = data_main.reshape(data_main.shape[0], height, width, depth)


# preparing output

predict1={}
predict2={}
predict3={}
predict4={}
predict5={}
predict6={}
predict_s1={}
predict_s2={}
predict_s3={}
predict_s4={}
predict_s5={}
predict_s6={}

inp = Input(shape=(height, width, depth)) # N.B. TensorFlow back-end expects channel dimension
ll=0
import bootstrap_sampling
import CNN_2models_cosine_lr
import sys, importlib
importlib.reload(sys.modules['CNN_2models_cosine_lr'])


kk=0
for kk in range(0,8):
    l2_lambda = 0.0001
    # test
    X_test = data_main
    Y_test_12 = type_man
    X_test_12 = X_test.astype('float32')
    # validation
    x_test=X_test_12
    y_test=Y_test_12
    num_classes = 12
    input_shape = x_test.shape[1:]
    ########
    model = CNN_2models_cosine_lr.CNN_model2(inp, num_classes)
    ## load snapshot models
    for jj in range(10):
        snapshot_model1 = load_model(path_model+"/ELU_CNN_original_fold"+str(kk)+"_rep"+str(jj)+"_snapshot_model_1.h5",compile=False)
        snapshot_model2 = load_model(path_model+"/ELU_CNN_original_fold"+str(kk)+"_rep"+str(jj)+"_snapshot_model_2.h5",compile=False)
        snapshot_model3 = load_model(path_model+"/ELU_CNN_original_fold"+str(kk)+"_rep"+str(jj)+"_snapshot_model_3.h5",compile=False)
        snapshot_model4 = load_model(path_model+"/ELU_CNN_original_fold"+str(kk)+"_rep"+str(jj)+"_snapshot_model_4.h5",compile=False)
        snapshot_model5 = load_model(path_model+"/ELU_CNN_original_fold"+str(kk)+"_rep"+str(jj)+"_snapshot_model_5.h5",compile=False)
        predict_s1[kk*10+jj]=snapshot_model1.predict(x_test)
        predict_s2[kk*10+jj]=snapshot_model2.predict(x_test)
        predict_s3[kk*10+jj]=snapshot_model3.predict(x_test)
        predict_s4[kk*10+jj]=snapshot_model4.predict(x_test)
        predict_s5[kk*10+jj]=snapshot_model5.predict(x_test)



f1macro_tot_s=np.zeros((8,1)) 
f1_class_s=np.zeros((8,6))

f1macro_tot_s_mode=np.zeros((8,1)) 
f1_class_s_mode=np.zeros((8,6))



from sklearn.preprocessing import LabelBinarizer


#ll=0


type_man_updated=type_man.copy()
type_man_updated[type_man_updated=='BLANK']='fp'
type_man_updated[type_man_updated=='NOISE']='fp'
type_man_updated[type_man_updated=='c4']='c3'
type_man_updated[type_man_updated=='c5']='c3'
type_man_updated[type_man_updated=='uh']='simple'
type_man_updated[type_man_updated=='us']='simple'
type_man_updated[type_man_updated=='s']='simple'
type_man_updated[type_man_updated=='up']='simple'
type_man_updated[type_man_updated=='d']='simple'
type_man_updated[type_man_updated=='f']='simple'
type_man_updated[type_man_updated=='u']='simple'
lb = LabelBinarizer()
labels = lb.fit_transform(type_man_updated)


np.unique(type_man_updated,return_counts=True)
from scipy import stats

for kk in range(0,8):
    bb=range(10)
    num_p2s=np.mean([predict_s5[kk*10+x1] for x1 in bb], 0).argmax(1)+1
    type_our_updated=num_p2s.copy()
    type_our_updated[type_our_updated==4]=5
    type_our_updated[type_our_updated==5]=5
    type_our_updated[type_our_updated==6]=4
    type_our_updated[type_our_updated==7]=3
    type_our_updated[type_our_updated==8]=5
    type_our_updated[type_our_updated==9]=5
    type_our_updated[type_our_updated==10]=6
    type_our_updated[type_our_updated==11]=5
    type_our_updated[type_our_updated==12]=5
    # real
    num_t=labels.argmax(1)+1
    # fscore
    f1macro_tot_s[kk]=f1_score(num_t, type_our_updated,average='macro')
    f1_class_s[kk,:]=f1_score(num_t, type_our_updated,average=None)

type_our_updated=np.zeros((len(type_man_updated),8)) 
for kk in range(0,8):
    bb=range(10)
    num_p2s1=np.mean([predict_s5[kk*10+x1] for x1 in bb], 0).argmax(1)+1
    type_our_updated[:,kk]=num_p2s1.copy()
    type_our_updated[:,kk][type_our_updated[:,kk]==4]=5
    type_our_updated[:,kk][type_our_updated[:,kk]==5]=5
    type_our_updated[:,kk][type_our_updated[:,kk]==6]=4
    type_our_updated[:,kk][type_our_updated[:,kk]==7]=3
    type_our_updated[:,kk][type_our_updated[:,kk]==8]=5
    type_our_updated[:,kk][type_our_updated[:,kk]==9]=5
    type_our_updated[:,kk][type_our_updated[:,kk]==10]=6
    type_our_updated[:,kk][type_our_updated[:,kk]==11]=5
    type_our_updated[:,kk][type_our_updated[:,kk]==12]=5


type_our_updated2=np.zeros((len(type_man_updated),1)) 
for i in range(type_our_updated.shape[0]):
    counts = np.bincount(type_our_updated[i,:].astype(int))
    type_our_updated2[i]=np.argmax(counts)

f1_score(num_t, type_our_updated2,average='macro')
f1_score(num_t, type_our_updated2,average=None)
### extract errors on classification
c2i=np.where(num_t==2)[0]
c2_rise=c2i[np.where(type_our_updated[c2i]==5)[0]]
c3i=np.where(num_t==3)[0]
c3_c2=c3i[np.where(type_our_updated[c3i]==2)[0]]
uii=np.where(num_t==6)[0]
ui_rise=uii[np.where(type_our_updated[uii]==5)[0]]

plt.clf()

for i in range(4):
    plt.subplot(3,4,i+1)
    plt.imshow(X_test[c2_rise[40+i],:,:,0], vmin=0, vmax=X_test[c2_rise[40+i],:,:,0].max(), cmap='gray_r',aspect='auto',origin='lower')
    plt.tick_params(bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    ##
    plt.subplot(3,4,i+5)
    plt.imshow(X_test[c3_c2[50+i],:,:,0], vmin=0, vmax=X_test[c3_c2[50+i],:,:,0].max(), cmap='gray_r',aspect='auto',origin='lower')
    plt.tick_params(bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    ##
    plt.subplot(3,4,i+9)
    plt.imshow(X_test[ui_rise[20+i],:,:,0], vmin=0, vmax=X_test[ui_rise[20+i],:,:,0].max(), cmap='gray_r',aspect='auto',origin='lower')
    plt.tick_params(bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    


plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('a.png', bbox_inches='tight')

# original case

# snapshot case
f1macro_tot_mean_s=100*(f1macro_tot_s.mean())
f1macro_tot_std_s=100*(np.sqrt(f1macro_tot_s.var()))
f1_class_mean_s=100*(f1_class_s.mean(axis=0))
f1_class_std_s=100*(np.sqrt(f1_class_s.var(axis=0)))

######################
########################
### extract errors on clssfication
c2i=np.where(num_t==3)[0]
c3_c2=c2i[np.where(type_our_updated2[c2i]==2)[0]]
c3_c3=c2i[np.where(type_our_updated2[c2i]==3)[0]]
c3_rise=c2i[np.where(type_our_updated2[c2i]==5)[0]]
c3_ui=c2i[np.where(type_our_updated2[c2i]==6)[0]]

'''
c3i=np.where(num_t==3)[0]
c3_c2=c3i[np.where(type_our_updated[c3i]==2)[0]]
uii=np.where(num_t==6)[0]
ui_rise=uii[np.where(type_our_updated[uii]==5)[0]]
'''

plt.clf()

for i in range(4):
    plt.subplot(4,4,i+1)
    plt.imshow(X_test[c3_c3[20+i],:,:,0], vmin=0, vmax=X_test[c3_c3[20+i],:,:,0].max(), cmap='gray_r',aspect='auto',origin='lower')
    plt.tick_params(bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    ##
    plt.subplot(4,4,i+5)
    plt.imshow(X_test[c3_ui[2+i],:,:,0], vmin=0, vmax=X_test[c3_ui[2+i],:,:,0].max(), cmap='gray_r',aspect='auto',origin='lower')
    plt.tick_params(bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    ##
    plt.subplot(4,4,i+9)
    plt.imshow(X_test[c3_c2[20+i],:,:,0], vmin=0, vmax=X_test[c3_c2[20+i],:,:,0].max(), cmap='gray_r',aspect='auto',origin='lower')
    plt.tick_params(bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plt.subplot(4,4,i+13)
    plt.imshow(X_test[c3_rise[2+i],:,:,0], vmin=0, vmax=X_test[c3_rise[2+i],:,:,0].max(), cmap='gray_r',aspect='auto',origin='lower')
    plt.tick_params(bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    


plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('c3_our_klivv.png', bbox_inches='tight')

#################
###################c4 and c5:
c4_id=np.where(type_man=='c4')[0]
c5_id=np.where(type_man=='c5')[0]
np.unique(type_our_updated2[c4_id],return_counts=True)
c4_c2=c4_id[np.where(type_our_updated2[c4_id]==2)[0]]
c4_c3=c4_id[np.where(type_our_updated2[c4_id]==3)[0]]
np.unique(type_our_updated2[c5_id],return_counts=True)
c5_c2=c5_id[np.where(type_our_updated2[c5_id]==2)[0]]
c5_c3=c5_id[np.where(type_our_updated2[c5_id]==3)[0]]

'''
c3i=np.where(num_t==3)[0]
c3_c2=c3i[np.where(type_our_updated[c3i]==2)[0]]
uii=np.where(num_t==6)[0]
ui_rise=uii[np.where(type_our_updated[uii]==5)[0]]
'''

plt.clf()

for i in range(3):
    plt.subplot(4,3,i+1)
    plt.imshow(X_test[c4_c2[10+i],:,:,0], vmin=0, vmax=X_test[c4_c2[10+i],:,:,0].max(), cmap='gray_r',aspect='auto',origin='lower')
    plt.tick_params(bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    ##
    plt.subplot(4,3,i+4)
    plt.imshow(X_test[c4_c3[10+i],:,:,0], vmin=0, vmax=X_test[c4_c3[10+i],:,:,0].max(), cmap='gray_r',aspect='auto',origin='lower')
    plt.tick_params(bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    ##
    plt.subplot(4,3,i+7)
    plt.imshow(X_test[c5_c2[i],:,:,0], vmin=0, vmax=X_test[c5_c2[i],:,:,0].max(), cmap='gray_r',aspect='auto',origin='lower')
    plt.tick_params(bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plt.subplot(4,3,i+10)
    plt.imshow(X_test[c5_c3[i],:,:,0], vmin=0, vmax=X_test[c5_c3[i],:,:,0].max(), cmap='gray_r',aspect='auto',origin='lower')
    plt.tick_params(bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('c4_c5_our_klivv.png', bbox_inches='tight')



###################



