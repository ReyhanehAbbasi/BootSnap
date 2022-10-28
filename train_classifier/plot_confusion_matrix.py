import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.PuRd):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype(int) / cm.sum(axis=1)[ :,np.newaxis]
        #cm = cm.astype(int) / cm.sum(axis=1)[ :, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    #print(cm);
    plt.imshow(100*cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #pcl=plt.colorbar()
    #pcl.set_label('F1-score (%)', rotation=270)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=30,fontsize=25)
    plt.yticks(tick_marks, classes,fontsize=25)
    fmt = 'd' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format((100*cm[i, j]).astype(int), fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    #plt.annotate('%', xy=(.5, .5))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.rcParams["font.size"] = "18"

