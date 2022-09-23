import pandas as pd 
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import hdbscan
from sklearn.decomposition import PCA
from matplotlib.colors import LinearSegmentedColormap
import NS_driver
from get_top_results import create_final_df

run_name = 'UniRep'

if run_name == 'DevRep':
	run = NS_driver.NS_run(8)
elif run_name =='OneHot':
	run = NS_driver.NS_run(10)
elif run_name == 'UniRep':
	run = NS_driver.NS_run(9)

df = pd.read_pickle('./ns_threshold/'+run.savename+'.pkl')
# devrep_df = devrep_df[devrep_df['Develop']>1]
df = create_final_df(df,subset=False)

df['Cluster'] = hdbscan.HDBSCAN(min_cluster_size=int(len(df)*0.01)).fit_predict(df[[run_name+' UMAP 1',run_name+' UMAP 2']].to_numpy())
# df['Cluster'] = hdbscan.HDBSCAN().fit_predict(df[['DevRep UMAP 1','DevRep UMAP 2']].to_numpy())

clust_list=np.unique(df['Cluster'].values)

#remove outliers
if -1 in clust_list:
	clust_list=np.delete(clust_list,np.where(clust_list==-1))

cluster_median=[]
for c in clust_list:
	med = np.median(df[df['Cluster']==c][run_name+' Dev'])
	cluster_median.append(med)

clust_order=np.argsort(cluster_median)

df_list =[]
for i,o in enumerate(clust_order):
	c = clust_list[o]
	df_t = df[df['Cluster']==c]
	df_t['Cluster']=i
	df_list.append(df_t)

df_final = pd.concat(df_list)

fig,axs= plt.subplots(1,2,figsize=[6,3],dpi=1200)
cmap=sns.color_palette('tab10',n_colors=len(clust_list))

ax=axs[0]
sns.scatterplot(data=df_final,x=run_name+' UMAP 1',y=run_name+' UMAP 2',hue='Cluster',ax=ax,legend=None,palette=cmap,marker='.',edgecolor=None,linewidth=0,alpha=1,s=6, hue_order=clust_list)
ax.tick_params(labelsize=6)
ax.set_xlabel('DevRep UMAP 1',fontsize=6)
ax.set_ylabel('DevRep UMAP 2',fontsize=6)

ax=axs[1]
sns.violinplot(data=df_final,x='Cluster',y=run_name+' Dev',palette=cmap, scale='width',inner='quartile',saturation=1,ax=ax, odrer=clust_list, hue_order=clust_list)
ax.tick_params(labelsize=6)
ax.set_xlabel('Cluster',fontsize=6)
ax.set_ylabel('Developability',fontsize=6)

fig.tight_layout()
fig.savefig('./Seq_clusters.png')


for c in range(len(clust_list)):
	df_clust=df_final[df_final['Cluster']==c]

	seqs=df_clust.loc[:,'Sequence'].values
	x_a=np.zeros([16,21])
	for seq in seqs:
		for i in range(len(seq)):
			aa=seq[i]
			x_a[i,aa]=x_a[i,aa]+1

	print(len(df_clust))

	frequency=x_a/len(seqs)

	frequency=frequency.reshape(16,21)
	frequency=pd.DataFrame(frequency)
	frequency.columns=list("ACDEFGHIKLMNPQRSTVWXY")
	frequency['Gap']=frequency['X']
	frequency=frequency[['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','Gap']]
	frequency.index=['7','8','9','9b','9c','10','11','12','34','35','36','36b','36c','37','38','39']
	for pos in ['7','8','9','10','11','12','34','35','36','37','38','39']:
		frequency['Gap'][pos]=np.nan
	
	fig,ax = plt.subplots(1,1,figsize=[6.5,3],dpi=1200)
	cmap_cluster = LinearSegmentedColormap.from_list('mycmap', ['white',cmap[c]])
	cmap_cluster.set_bad('gray')


	heat_map=sns.heatmap(frequency,square=True, vmin=0, vmax=.25 ,cmap=cmap_cluster,cbar_kws={"shrink": 0.6,"ticks":[0,0.1,.2]})
	heat_map.figure.axes[-1].set_ylabel('Frequency',size=6)
	heat_map.figure.axes[-1].tick_params(labelsize=6)
	ax.set_yticks([x+0.5 for x in list(range(16))])
	ax.set_yticklabels(['7','8','9','9b','9c','10','11','12','34','35','36','36b','36c','37','38','39'])
	ax.set_ylim([16.5,-0.5])

	ax.set_xticks([x+0.5 for x in list(range(21))])
	ax.set_xticklabels(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','-'])
	ax.set_xlim([-0.5,21.5])
	ax.tick_params(labelsize=6)

	plt.tight_layout()
	fig.savefig('./cluster_'+str(c)+'_heatmap.png')