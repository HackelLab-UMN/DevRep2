import NS_driver
import pandas as pd 
import numpy as np
from _graph_vis import subset_df,get_graph_rep
import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def get_unique_df(threshold_df):
	seqs = np.stack(threshold_df['Sequence'])
	_, unique_idx = np.unique(seqs,return_index=True, axis=0)
	threshold_df=threshold_df.iloc[unique_idx].sort_index()
	return threshold_df

onehot_run = NS_driver.NS_run(10)
devrep_run = NS_driver.NS_run(8)
unirep_run = NS_driver.NS_run(9)
devrep_reducer = pickle.load(open('./umap_reducers/'+devrep_run.savename+'.pkl','rb'))
onehot_reducer = pickle.load(open('./umap_reducers/'+onehot_run.savename+'.pkl','rb'))
unirep_reducer = pickle.load(open('./umap_reducers/'+unirep_run.savename+'.pkl','rb'))


def create_final_df(df,subset=True):
	df=get_unique_df(df)
	if subset:
		df = df.iloc[-100:]
	sequence = df.Sequence.to_numpy()

	devrep_dev=NS_driver.devrep(np.stack(df['Sequence']))
	onehot_dev=NS_driver.onehot(np.stack(df['Sequence']))
	unirep_dev=NS_driver.unirep_paratope(np.stack(df['Sequence']))

	devrep_emb = get_graph_rep(devrep_run,df)['Embedding']
	devrep_xy=devrep_reducer.transform(np.stack(devrep_emb))

	onehot_emb = get_graph_rep(onehot_run,df)['Embedding']
	onehot_xy=onehot_reducer.transform(np.stack(onehot_emb))

	unirep_emb = get_graph_rep(unirep_run,df)['Embedding']
	unirep_xy = unirep_reducer.transform(np.stack(unirep_emb))

	run_df = pd.DataFrame({'Sequence':sequence,'DevRep Dev':devrep_dev,'OneHot Dev':onehot_dev, 'UniRep Dev':unirep_dev,
								'DevRep UMAP 1':devrep_xy[:,0],'DevRep UMAP 2':devrep_xy[:,1],
								'OneHot UMAP 1':onehot_xy[:,0],'OneHot UMAP 2':onehot_xy[:,1],
								'UniRep UMAP 1':unirep_xy[:,0],'UniRep UMAP 2':unirep_xy[:,1]})

	return run_df

if __name__ =='__main__':
	devrep_df = pd.read_pickle('./ns_threshold/'+devrep_run.savename+'.pkl')
	devrep_df = create_final_df(devrep_df)
	devrep_df['Model']='DevRep'

	onehot_df = pd.read_pickle('./ns_threshold/'+onehot_run.savename+'.pkl')
	onehot_df = create_final_df(onehot_df)
	onehot_df['Model']='OneHot'

	unirep_df = pd.read_pickle('./ns_threshold/'+unirep_run.savename+'.pkl')
	unirep_df = create_final_df(unirep_df)
	unirep_df['Model']='UniRep'

	df = pd.concat([devrep_df,onehot_df,unirep_df])

	fig, axs = plt.subplots(2,3,figsize=[9,6],dpi=600)

	ax=axs[0,0]
	sns.violinplot(data=df,x='Model',y='DevRep Dev',ax=ax,inner=None,scale='width')
	ax.tick_params(labelsize=6,which='both',colors='black')
	ax.set_ylabel('DevRep Model Developability',fontsize=6)
	ax.set_xlabel(None)
	ax.legend().set_visible(False)

	ax=axs[1,0]
	sns.scatterplot(data=df,x='DevRep UMAP 1',y='DevRep UMAP 2',hue='Model',ax=ax,alpha=0.5)
	ax.tick_params(labelsize=6,which='both',colors='black')
	ax.set_ylabel('DevRep UMAP 1',fontsize=6)
	ax.set_xlabel('DevRep UMAP 2',fontsize=6)
	ax.legend(fontsize=6)

	ax=axs[0,1]
	sns.violinplot(data=df,x='Model',y='OneHot Dev',ax=ax,inner=None,scale='width')
	ax.tick_params(labelsize=6,which='both',colors='black')
	ax.set_ylabel('OneHot Model Developability',fontsize=6)
	ax.set_xlabel(None)
	ax.legend().set_visible(False)

	ax=axs[1,1]
	sns.scatterplot(data=df,x='OneHot UMAP 1',y='OneHot UMAP 2',hue='Model',ax=ax,alpha=0.5)
	ax.tick_params(labelsize=6,which='both',colors='black')
	ax.set_ylabel('OneHot UMAP 1',fontsize=6)
	ax.set_xlabel('OneHot UMAP 2',fontsize=6)
	ax.legend(fontsize=6)

	ax=axs[0,2]
	sns.violinplot(data=df,x='Model',y='UniRep Dev',ax=ax,inner=None,scale='width')
	ax.tick_params(labelsize=6,which='both',colors='black')
	ax.set_ylabel('UniRep Model Developability',fontsize=6)
	ax.set_xlabel(None)
	ax.legend().set_visible(False)

	ax=axs[1,2]
	sns.scatterplot(data=df,x='UniRep UMAP 1',y='UniRep UMAP 2',hue='Model',ax=ax,alpha=0.5)
	ax.tick_params(labelsize=6,which='both',colors='black')
	ax.set_ylabel('UniRep UMAP 1',fontsize=6)
	ax.set_xlabel('UniRep UMAP 2',fontsize=6)
	ax.legend(fontsize=6)


	fig.tight_layout()
	fig.savefig('./top_results.png')
	plt.close()
