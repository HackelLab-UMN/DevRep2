import NS_driver
import pandas as pd 
import numpy as np
from _graph_vis import subset_df,get_graph_rep
import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist 



onehot_run = NS_driver.NS_run(10)
devrep_run = NS_driver.NS_run(8)
unirep_run = NS_driver.NS_run(9)
devrep_reducer = pickle.load(open('./umap_reducers/'+devrep_run.savename+'.pkl','rb'))
onehot_reducer = pickle.load(open('./umap_reducers/'+onehot_run.savename+'.pkl','rb'))
unirep_reducer = pickle.load(open('./umap_reducers/'+unirep_run.savename+'.pkl','rb'))


def create_final_df(df_a):
	df = df_a.copy()
	sequence = df.Sequence.to_numpy()

	devrep_dev=NS_driver.devrep(np.stack(df['Sequence']))
	onehot_dev=NS_driver.onehot(np.stack(df['Sequence']))
	unirep_dev=NS_driver.unirep_paratope(np.stack(df['Sequence']))

	df['Develop']=devrep_dev
	devrep_emb = get_graph_rep(devrep_run,df)['Embedding']
	devrep_xy=devrep_reducer.transform(np.stack(devrep_emb))

	df['Develop']=onehot_dev
	onehot_emb = get_graph_rep(onehot_run,df)['Embedding']
	onehot_xy=onehot_reducer.transform(np.stack(onehot_emb))

	df['Develop']=unirep_dev
	unirep_emb = get_graph_rep(unirep_run,df)['Embedding']
	unirep_xy = unirep_reducer.transform(np.stack(unirep_emb))

	run_df = pd.DataFrame({'Sequence':sequence,'DevRep Dev':devrep_dev,'OneHot Dev':onehot_dev, 'UniRep Dev':unirep_dev,
								'DevRep Embedding':devrep_emb, 'OneHot Embedding':onehot_emb, 'UniRep Embedding':unirep_emb,
								'DevRep UMAP 1':devrep_xy[:,0],'DevRep UMAP 2':devrep_xy[:,1],
								'OneHot UMAP 1':onehot_xy[:,0],'OneHot UMAP 2':onehot_xy[:,1],
								'UniRep UMAP 1':unirep_xy[:,0],'UniRep UMAP 2':unirep_xy[:,1]})

	return run_df

if __name__ =='__main__':
	devrep_ns_df = pd.read_pickle('./final_df/nested_sampling_df_DevRep.pkl')
	devrep_ns_df = create_final_df(devrep_ns_df)
	devrep_ns_df['Model']='DevRep'
	devrep_ns_df['Method']='Nested Sampling'

	devrep_sa_df = pd.read_pickle('./final_df/simulated_annealing_df_DevRep.pkl')
	devrep_sa_df = create_final_df(devrep_sa_df)
	devrep_sa_df['Model']='DevRep'
	devrep_sa_df['Method']='Simulated Annealing'

	onehot_ns_df = pd.read_pickle('./final_df/nested_sampling_df_OneHot.pkl')
	onehot_ns_df = create_final_df(onehot_ns_df)
	onehot_ns_df['Model']='OneHot'
	onehot_ns_df['Method']='Nested Sampling'

	onehot_sa_df = pd.read_pickle('./final_df/simulated_annealing_df_OneHot.pkl')
	onehot_sa_df = create_final_df(onehot_sa_df)
	onehot_sa_df['Model']='OneHot'
	onehot_sa_df['Method']='Simulated Annealing'

	unirep_ns_df = pd.read_pickle('./final_df/nested_sampling_df_UniRep.pkl')
	unirep_ns_df = create_final_df(unirep_ns_df)
	unirep_ns_df['Model']='UniRep'
	unirep_ns_df['Method']='Nested Sampling'

	unirep_sa_df = pd.read_pickle('./final_df/simulated_annealing_df_UniRep.pkl')
	unirep_sa_df = create_final_df(unirep_sa_df)
	unirep_sa_df['Model']='UniRep'
	unirep_sa_df['Method']='Simulated Annealing'

	random_df = pd.read_pickle('./final_df/random_df.pkl')
	random_df['Model'] = 'Random'
	random_df['Method'] = 'Random'


	df = pd.concat([devrep_ns_df,devrep_sa_df,onehot_ns_df,onehot_sa_df,unirep_ns_df,unirep_sa_df,random_df])

	top_devrep_ns = pd.read_pickle('./final_df/high_yield_nested_sampling_df_DevRep.pkl')
	top_devrep_ns = create_final_df(top_devrep_ns)
	top_devrep_ns['Model']='DevRep'
	top_devrep_ns['Method']='Top Nested Sampling'

	top_unirep_ns = pd.read_pickle('./final_df/high_yield_nested_sampling_df_UniRep.pkl')
	top_unirep_ns = create_final_df(top_unirep_ns)
	top_unirep_ns['Model']='UniRep'
	top_unirep_ns['Method']='Top Nested Sampling'

	df_all = pd.concat([devrep_ns_df,devrep_sa_df,onehot_ns_df,onehot_sa_df,unirep_ns_df,unirep_sa_df,random_df,top_devrep_ns,top_unirep_ns])
	df_all.to_pickle('./ordered_sequences_df.pkl')

	fig, axs = plt.subplots(2,3,figsize=[6,4],dpi=600)

	ax=axs[0,0]
	sns.violinplot(data=df,x='Model',y='DevRep Dev',hue='Method',ax=ax,inner='quartile',split=False,scale='width')
	ax.tick_params(labelsize=6,which='both',colors='black')
	ax.set_ylabel('DevRep Model Developability',fontsize=6)
	ax.set_xlabel(None)
	ax.legend(fontsize=6)
	ax.legend().set_visible(False)

	ax=axs[1,0]
	sns.scatterplot(data=df,x='DevRep UMAP 1',y='DevRep UMAP 2',style='Model',hue='Method',ax=ax,alpha=0.1)
	ax.tick_params(labelsize=6,which='both',colors='black')
	ax.set_ylabel('DevRep UMAP 1',fontsize=6)
	ax.set_xlabel('DevRep UMAP 2',fontsize=6)
	ax.legend(fontsize=6)
	ax.legend().set_visible(False)

	ax=axs[0,1]
	sns.violinplot(data=df,x='Model',y='OneHot Dev',hue='Method',ax=ax,inner='quartile',split=False,scale='width')
	ax.tick_params(labelsize=6,which='both',colors='black')
	ax.set_ylabel('OneHot Model Developability',fontsize=6)
	ax.set_xlabel(None)
	ax.legend(fontsize=6)
	ax.legend().set_visible(False)

	ax=axs[1,1]
	sns.scatterplot(data=df,x='OneHot UMAP 1',y='OneHot UMAP 2',style='Model',hue='Method',ax=ax,alpha=0.1)
	ax.tick_params(labelsize=6,which='both',colors='black')
	ax.set_ylabel('OneHot UMAP 1',fontsize=6)
	ax.set_xlabel('OneHot UMAP 2',fontsize=6)
	ax.legend(fontsize=6)
	ax.legend().set_visible(False)

	ax=axs[0,2]
	sns.violinplot(data=df,x='Model',y='UniRep Dev',hue='Method',ax=ax,inner='quartile',split=False,scale='width')
	ax.tick_params(labelsize=6,which='both',colors='black')
	ax.set_ylabel('UniRep Model Developability',fontsize=6)
	ax.set_xlabel(None)
	ax.legend(fontsize=6,framealpha=1)
	# ax.legend().set_visible(False)

	ax=axs[1,2]
	sns.scatterplot(data=df,x='UniRep UMAP 1',y='UniRep UMAP 2',style='Model',hue='Method',ax=ax,alpha=0.1)
	ax.tick_params(labelsize=6,which='both',colors='black')
	ax.set_ylabel('UniRep UMAP 1',fontsize=6)
	ax.set_xlabel('UniRep UMAP 2',fontsize=6)
	ax.legend(fontsize=6,framealpha=1)
	# ax.legend().set_visible(False)


	fig.tight_layout()
	fig.savefig('./final_df/final_comparison.png')
	plt.close()

	# devrep_seqs = pd.concat([devrep_ns_df,devrep_sa_df])
	# seqs = np.stack(devrep_seqs.Sequence)
	# _, unique_idx = np.unique(seqs,return_index=True, axis=0)
	# devrep_seqs=devrep_seqs.iloc[unique_idx].sort_index()

	# onehot_seqs = pd.concat([onehot_ns_df,onehot_sa_df])
	# seqs = np.stack(onehot_seqs.Sequence)
	# _, unique_idx = np.unique(seqs,return_index=True, axis=0)
	# onehot_seqs=onehot_seqs.iloc[unique_idx].sort_index()

	# unirep_seqs = pd.concat([unirep_ns_df,unirep_sa_df])
	# seqs = np.stack(unirep_seqs.Sequence)
	# _, unique_idx = np.unique(seqs,return_index=True, axis=0)
	# unirep_seqs=unirep_seqs.iloc[unique_idx].sort_index()


	# fig,axs= plt.subplots(1,3,figsize=[9,3],dpi=600,sharex=True,sharey=True)

	# ax = axs[0]
	# distance_list=[]
	# for i in range(len(devrep_seqs)):
	# 	seq = devrep_seqs.Sequence.iloc[i]
	# 	distances = cdist([seq],np.stack(onehot_seqs.Sequence),metric='hamming')
	# 	distance_list.append(np.min(distances)*16)

	# sns.histplot(x=distance_list,stat='probability',binrange=[-0.5,16.5],bins=17,ax=ax)
	# ax.tick_params(labelsize=6)
	# ax.set_xlabel('Distance to Closest Sequence',fontsize=6)
	# ax.set_ylabel('Fraction of Sequences',fontsize=6)
	# ax.set_title('DevRep Vs. OneHot',fontsize=6)

	# ax = axs[1]
	# distance_list=[]
	# for i in range(len(devrep_seqs)):
	# 	seq = devrep_seqs.Sequence.iloc[i]
	# 	distances = cdist([seq],np.stack(unirep_seqs.Sequence),metric='hamming')
	# 	distance_list.append(np.min(distances)*16)

	# sns.histplot(x=distance_list,stat='probability',binrange=[-0.5,16.5],bins=17,ax=ax)
	# ax.tick_params(labelsize=6)
	# ax.set_xlabel('Distance to Closest Sequence',fontsize=6)
	# ax.set_ylabel('Fraction of Sequences',fontsize=6)
	# ax.set_title('DevRep Vs. UniRep',fontsize=6)

	# ax = axs[2]
	# distance_list=[]
	# for i in range(len(onehot_seqs)):
	# 	seq = onehot_seqs.Sequence.iloc[i]
	# 	distances = cdist([seq],np.stack(unirep_seqs.Sequence),metric='hamming')
	# 	distance_list.append(np.min(distances)*16)

	# sns.histplot(x=distance_list,stat='probability',binrange=[-0.5,16.5],bins=17,ax=ax)
	# ax.tick_params(labelsize=6)
	# ax.set_xlabel('Distance to Closest Sequence',fontsize=6)
	# ax.set_ylabel('Fraction of Sequences',fontsize=6)
	# ax.set_title('OneHot Vs. UniRep',fontsize=6)

	# fig.tight_layout()
	# fig.savefig('./final_df/final_distance_plot.png')
	# plt.close()