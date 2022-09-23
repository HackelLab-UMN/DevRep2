import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns 
import pickle
from scipy.spatial.distance import cdist, pdist, squareform

seqs_per_model=[]
for mdl in ['DevRep','OneHot']:
	savename=mdl+'_Sim_Anneal'
	seqs_per_T=np.array(pickle.load(open('./sampling/'+savename+'seqs_per_T.pkl', "rb")))

	seqs=[]
	t=1
	while len(seqs)<100:
		seqs=seqs_per_T[-t,:,:]
		seqs=np.unique(seqs,axis=0)
		t=t+1

	df_a=pd.read_pickle('./datasets/assay_to_dot_training_data.pkl')
	df_b=pd.read_pickle('./datasets/seq_to_dot_test_data.pkl')

	df=pd.concat([df_a,df_b])

	tested_df=np.stack(df['Ordinal'].values)


	min_dist=[]
	for predict_seq in seqs:
		distances=cdist(np.expand_dims(predict_seq,0),tested_df,metric='hamming')
		min_dist.append(min(distances[0])*16)
	print(min(min_dist),max(min_dist))

	self_dist_list=[]
	self_dist=squareform(pdist(seqs,metric='hamming'))
	for i in range(len(self_dist)):
		a=self_dist[i][0:i]
		b=self_dist[i][i+1:]
		compares=np.concatenate([a,b])
		self_dist_list.append(min(compares)*16)

	self_dist_list=np.array(self_dist_list)


	seqs=seqs[np.argsort(self_dist_list)]
	seqs=seqs[::-1]

	seqs_to_order=seqs[0:100]
	seqs_per_model.append(seqs_to_order)