import numpy as np

import pandas as pd

###### manual labeling 
df_org1= pd.read_csv('mM3.5individuality_run3_day2-2016-10-28_17-05-14_0020.WAV.csv')
type_man1= df_org1.Type


df_org2 = pd.read_csv('mM3.5individuality_run3_day2-2016-10-28_17-10-15_0021.WAV.csv')
type_man2= df_org2.Type


df_org3 = pd.read_csv('mM27.18-individuality_run1_day2-2016-10-14_15-16-00_0003.WAV.csv')
type_man3= df_org3.Type


type_man=type_man1.append(type_man2.append(type_man3))

##### DSQ labeling
df_dsq1= pd.read_csv('mM3.5individuality_run3_day2-2016-10-28_17-05-14_0020.WAV_denoised_classified_Stats.csv')
type_dsq1= df_dsq1.Label


df_dsq2 = pd.read_csv('mM3.5individuality_run3_day2-2016-10-28_17-10-15_0021.WAV_updated_denoised_classified_Stats.csv')
type_dsq2= df_dsq2.Label


df_dsq3 = pd.read_csv('mM27.18-individuality_run1_day2-2016-10-14_15-16-00_0003.WAV_updated_denoised_classified_Stats.csv')
type_dsq3= df_dsq3.Label


type_dsq=type_dsq1.append(type_dsq2.append(type_dsq3))

##### re-label DSQ results:
type_dsq_updated=type_dsq.copy()
type_dsq_updated[type_dsq_updated=='Split']='c3'
type_dsq_updated[type_dsq_updated=='Step']='c2'
type_dsq_updated[type_dsq_updated=='Noise']='fp'
type_dsq_updated[type_dsq_updated=='Wave1']='c'
type_dsq_updated[type_dsq_updated=='Rise']='simple'
type_dsq_updated[type_dsq_updated=='InvertedU1']='ui'

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
type_man_updated[type_man_updated=='uh']='simple'
type_man_updated[type_man_updated=='u']='simple'

#### evaluation
from sklearn.metrics import f1_score
f1_score(type_man_updated,type_dsq_updated, average=None)


#################
###################c4 and c5:
c4_id=np.where(type_man=='c4')[0]
c5_id=np.where(type_man=='c5')[0]
a=[]
for i in c4_id:
    a.append(type_dsq_updated[i])
[type_dsq_updated[i] for i in c4_id]
np.unique(type_dsq_updated.iloc()[c4_id],return_counts=True)
c4_c2=c4_id[np.where(type_dsq_updated.iloc()[c4_id]==2)[0]]
c4_c3=c4_id[np.where(type_dsq_updated[c4_id]==3)[0]]
np.unique(type_dsq_updated[c5_id],return_counts=True)
c5_c2=c5_id[np.where(type_dsq_updated[c5_id]==2)[0]]
c5_c3=c5_id[np.where(type_dsq_updated[c5_id]==3)[0]]



