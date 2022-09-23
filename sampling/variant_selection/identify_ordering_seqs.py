from NS_driver import NS_run
from SA_driver import SA_run
from _graph_vis import get_graph_rep

import pandas as pd 
import numpy as np
import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import cdist 


name='DevRep'


#devrep
if name == 'DevRep':
	ns_100 = NS_run(8)
	umap_reducer = pickle.load(open('./umap_reducers/'+ns_100.savename+'.pkl','rb'))
	sa_100 = SA_run(4)

#onehot
if name =='OneHot':
	ns_100 = NS_run(10)
	umap_reducer = pickle.load(open('./umap_reducers/'+ns_100.savename+'.pkl','rb'))
	sa_100 = SA_run(6)

#unirep
if name =='UniRep':
	ns_100 = NS_run(9)
	umap_reducer = pickle.load(open('./umap_reducers/'+ns_100.savename+'.pkl','rb'))
	sa_100 = SA_run(5)

def get_unique_and_emb(df,run,n_seq=1000):
	seqs = np.stack(df['Sequence'])
	_, unique_idx = np.unique(seqs,return_index=True, axis=0)
	df=df.iloc[unique_idx].sort_index()
	# df = df.iloc[-n_seq:]

	df = get_graph_rep(run,df)

	seq_xy = umap_reducer.transform(np.stack(df.Embedding))

	df['UMAP 1']=seq_xy[:,0]
	df['UMAP 2']=seq_xy[:,1]

	df['Rank']= max(df.index) - np.array(df.index) + 1


	return df


def top_ns_seqs(run):
	df = pd.read_pickle('./ns_threshold/'+run.savename+'.pkl')
	df = get_unique_and_emb(df,run)
	df['Method']= 'Nested Sampling'
	return df

def top_sa_seqs(run,n_seq=1000):
	T_list = run.T_list
	# i_list = list(range(0,500,50))+[499]
	i_list= [499]
	df_list=[]
	for T in T_list:
		for i in i_list:
			walkers,n_positions,_T,p_accept = pickle.load(open('./sa_walkers/'+run.savename+'_'+str(T)+'_'+str(i)+'.pkl', "rb"))
			seqs,devs = zip(*[(walk.sequence,walk.develop) for walk in walkers])
			df_loc = pd.DataFrame({'Sequence':seqs,'Develop':devs})

			seqs = np.stack(df_loc['Sequence'])
			_, unique_idx = np.unique(seqs,return_index=True, axis=0)
			df_loc=df_loc.iloc[unique_idx].sort_index()

			df_list.append(df_loc)
	df = pd.concat(df_list)

	df = df.sort_values(by=['Develop'],ignore_index=True)
	df = get_unique_and_emb(df,run)
	df['Method']= 'Simulated Annealing'
	return df 

ns_100_df = top_ns_seqs(ns_100)
sa_100_df = top_sa_seqs(sa_100)
df = pd.concat([ns_100_df,sa_100_df])

fig, axs= plt.subplots(1,2,figsize=[6,3],dpi=600)

ax = axs[0]
sns.scatterplot(data=df,x='Rank',y='Develop',hue='Method',ax=ax,legend="full",palette=['r','b'],alpha=0.1)
ax.set_ylabel('Developability',fontsize=6)
ax.set_xlabel('Rank',fontsize=6)
ax.set_xscale('log')
ax.tick_params(labelsize=6,which='both',colors='black')
ax.legend(fontsize=6)

ax = axs[1]
sns.scatterplot(data=df,x='UMAP 1',y='UMAP 2',hue='Method',ax=ax,linewidth=0,legend="full",palette=['r','b'],alpha=0.1)
ax.set_ylabel('UMAP 2',fontsize=6)
ax.set_xlabel('UMAP 1',fontsize=6)
ax.tick_params(labelsize=6,which='both',colors='black')
ax.legend(fontsize=6)

fig.tight_layout()
fig.savefig('./rank_plot.png')
plt.close()

########################################
def add_cluster(df,clust_frac=0.01):

	df['Cluster'] = hdbscan.HDBSCAN(min_cluster_size=int(len(df)*clust_frac)).fit_predict(df[['UMAP 1','UMAP 2']].to_numpy())
	clust_list=np.unique(df['Cluster'].values)

	#remove outliers
	if -1 in clust_list:
		clust_list=np.delete(clust_list,np.where(clust_list==-1))

	cluster_median=[]
	for c in clust_list:
		med = np.median(df[df['Cluster']==c]['Develop'])
		cluster_median.append(med)

	clust_order=np.argsort(cluster_median)

	df_list =[]
	for i,o in enumerate(clust_order):
		c = clust_list[o]
		df_t = df[df['Cluster']==c]
		df_t['Cluster']=i+1
		df_list.append(df_t)

	df_final = pd.concat(df_list)
	return df_final


def even_subsample(df):
	n_clusters=len(np.unique(df['Cluster']))
	n_seq_per_clust=int(np.ceil(100/n_clusters))

	df_list=[]
	for c in range(1,n_clusters+1):
		df_t = df[df['Cluster']==c]
		if len(df_t)>n_seq_per_clust:
			df_t = df_t.iloc[-n_seq_per_clust:]
		df_list.append(df_t)
	df = pd.concat(df_list)
	return df


def heatmap_per_cluster(method,cmap,df):
	for c in np.unique(df['Cluster']):
		df_clust=df[df['Cluster']==c]

		seqs=df_clust.loc[:,'Sequence'].values
		x_a=np.zeros([16,21])
		for seq in seqs:
			for i in range(len(seq)):
				aa=seq[i]
				x_a[i,aa]=x_a[i,aa]+1

		frequency=x_a/len(seqs)
		frequency=frequency.reshape(16,21)
		frequency=pd.DataFrame(frequency)
		frequency.columns=list("ACDEFGHIKLMNPQRSTVWXY")
		frequency['Gap']=frequency['X']
		frequency=frequency[['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','Gap']]
		frequency.index=['7','8','9','9b','9c','10','11','12','34','35','36','36b','36c','37','38','39']
		for pos in ['7','8','9','10','11','12','34','35','36','37','38','39']:
			frequency['Gap'][pos]=np.nan
		
		fig,ax = plt.subplots(1,1,figsize=[6.5,3],dpi=300)
		cmap_cluster = LinearSegmentedColormap.from_list('mycmap', ['white',cmap[c-1]],gamma=0.5)
		cmap_cluster.set_bad('gray')


		heat_map=sns.heatmap(frequency,square=True, vmin=0, vmax=1 ,cmap=cmap_cluster,cbar_kws={"shrink": 0.6,"ticks":[0,0.5,1]})
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
		fig.savefig('./'+method+'_'+str(c)+'_heatmap.png')
		plt.close()


#################################################
ns_100_df = add_cluster(ns_100_df)
sa_100_df = add_cluster(sa_100_df)

fig,axs= plt.subplots(2,4,figsize=[6.5,3],dpi=1200)

ax=axs[0,0]
cmap=sns.color_palette('hls',n_colors=len(np.unique(ns_100_df['Cluster'])))
sns.scatterplot(data=ns_100_df,x='UMAP 1',y='UMAP 2',hue='Cluster',ax=ax,legend=None,palette=cmap,marker='.',edgecolor=None,linewidth=0,alpha=.1)
ax.tick_params(labelsize=6)
ax.set_xlabel('UMAP 1',fontsize=6)
ax.set_ylabel('UMAP 2',fontsize=6)
# ax.set_title('Nested Sampling', fontsize=6)

ax = axs[0,1]
sns.violinplot(data=ns_100_df,x='Cluster',y='Develop', palette=cmap, scale='width',inner='quartile',saturation=1,ax=ax,linewidth=0)
ax.tick_params(labelsize=6)
ax.set_xlabel('Cluster',fontsize=6)
ax.set_ylabel('Developability',fontsize=6)
# ax.set_title('Nested Sampling',fontsize=6)

ax=axs[1,0]
sub_ns_100_df=even_subsample(ns_100_df)
sns.scatterplot(data=sub_ns_100_df,x='UMAP 1',y='UMAP 2',hue='Cluster',ax=ax,legend=None,palette=cmap,marker='.',edgecolor=None,linewidth=0,alpha=.1)
ax.tick_params(labelsize=6)
ax.set_xlabel('UMAP 1',fontsize=6)
ax.set_ylabel('UMAP 2',fontsize=6)

ax = axs[1,1]
sns.violinplot(data=sub_ns_100_df,x='Cluster',y='Develop', palette=cmap, scale='width',inner='quartile',saturation=1,ax=ax, linewidth=0)
ax.tick_params(labelsize=6)
ax.set_xlabel('Cluster',fontsize=6)
ax.set_ylabel('Developability',fontsize=6)

# heatmap_per_cluster('Nested Sampling',cmap,sub_ns_100_df)

ax=axs[0,2]
cmap=sns.color_palette('hls',n_colors=len(np.unique(sa_100_df['Cluster'])))
sns.scatterplot(data=sa_100_df,x='UMAP 1',y='UMAP 2',hue='Cluster',ax=ax,legend=None,palette=cmap,marker='.',edgecolor=None,linewidth=0,alpha=.1)
ax.tick_params(labelsize=6)
ax.set_xlabel('UMAP 1',fontsize=6)
ax.set_ylabel('UMAP 2',fontsize=6)
# ax.set_title('Simulated Annealing', fontsize=6)

ax = axs[0,3]
sns.violinplot(data=sa_100_df,x='Cluster',y='Develop', palette=cmap, scale='width',inner='quartile',saturation=1,ax=ax, linewidth=0)
ax.tick_params(labelsize=6)
ax.set_xlabel('Cluster',fontsize=6)
ax.set_ylabel('Developability',fontsize=6)
# ax.set_title('Simulated Annealing', fontsize=6)

ax=axs[1,2]
sub_sa_100_df=even_subsample(sa_100_df)
sns.scatterplot(data=sub_sa_100_df,x='UMAP 1',y='UMAP 2',hue='Cluster',ax=ax,legend=None,palette=cmap,marker='.',edgecolor=None,linewidth=0,alpha=.1)
ax.tick_params(labelsize=6)
ax.set_xlabel('UMAP 1',fontsize=6)
ax.set_ylabel('UMAP 2',fontsize=6)

ax = axs[1,3]
sns.violinplot(data=sub_sa_100_df,x='Cluster',y='Develop', palette=cmap, scale='width',inner='quartile',saturation=1,ax=ax, linewidth=0)
ax.tick_params(labelsize=6)
ax.set_xlabel('Cluster',fontsize=6)
ax.set_ylabel('Developability',fontsize=6)

# heatmap_per_cluster('Simulated Annealing',cmap,sub_sa_100_df)

xl,yl = np.inf,np.inf 
xh,yh = -np.inf,-np.inf
for ax_i in [(0,0),(1,0),(0,2),(1,2)]:
	ax = axs[ax_i[0],ax_i[1]]
	x_l,x_h = ax.get_xlim()
	if x_l < xl:
		xl = x_l
	if x_h > xh:
		xh = x_h
	y_l,y_h = ax.get_ylim()
	if y_l < yl:
		yl = y_l
	if y_h > yh:
		yh = y_h
for ax_i in [(0,0),(1,0),(0,2),(1,2)]:
	ax = axs[ax_i[0],ax_i[1]]
	ax.set_xlim(xl,xh)
	ax.set_ylim(yl,yh)

yl,yh = np.inf,-np.inf
for ax_i in [(0,1),(1,1),(0,3),(1,3)]:
	ax = axs[ax_i[0],ax_i[1]]
	y_l,y_h = ax.get_ylim()
	if y_l < yl:
		yl = y_l
	if y_h > yh:
		yh = y_h
for ax_i in [(0,1),(1,1),(0,3),(1,3)]:
	ax = axs[ax_i[0],ax_i[1]]
	ax.set_ylim(yl,yh)


fig.tight_layout()
fig.savefig('./cluster_plot.png')
plt.close()

#########################################################################
# fig, axs = plt.subplots(2,2,figsize=[6,6],dpi=600,sharey=True,sharex='col')

# intra_h_list,inter_h_list=[],[]
# intra_u_list,inter_u_list=[],[]

# for i in range(len(sub_ns_100_df)):
# 	seq = sub_ns_100_df.iloc[i]
# 	clust = seq.Cluster
# 	intra_clust_df = sub_ns_100_df[sub_ns_100_df['Cluster']==clust]
# 	intra_clust_df=intra_clust_df.drop(index=seq.name)
# 	h_distances = cdist([seq.Sequence],np.stack(intra_clust_df.Sequence),metric='hamming')
# 	intra_h_list.append(np.min(h_distances)*16)
# 	# u_distances = cdist([seq[['UMAP 1','UMAP 2']]], np.stack(intra_clust_df[['UMAP 1','UMAP 2']].values),metric='euclidean')
# 	u_distances = cdist([seq.Embedding],np.stack(intra_clust_df.Embedding),metric='euclidean')
# 	intra_u_list.append(np.min(u_distances))
# 	inter_clust_df = sub_ns_100_df[sub_ns_100_df['Cluster']!=clust]
# 	h_distances = cdist([seq.Sequence],np.stack(inter_clust_df.Sequence),metric='hamming')
# 	inter_h_list.append(np.min(h_distances)*16)
# 	# u_distances = cdist([seq[['UMAP 1','UMAP 2']]], np.stack(inter_clust_df[['UMAP 1','UMAP 2']].values),metric='euclidean')
# 	u_distances = cdist([seq.Embedding],np.stack(inter_clust_df.Embedding),metric='euclidean')
# 	inter_u_list.append(np.min(u_distances))

# intra_h_list = pd.DataFrame({'Distance':intra_h_list})
# intra_h_list['Pair Type']='Intra-Cluster'
# inter_h_list = pd.DataFrame({'Distance':inter_h_list})
# inter_h_list['Pair Type']='Inter-Cluster'
# h_list=pd.concat([intra_h_list,inter_h_list])

# ax = axs[0,0]
# ax=sns.histplot(data=h_list,x='Distance',hue='Pair Type',stat='probability',binrange=[-0.5,16.5],bins=17,ax=ax,legend=True)
# ax.tick_params(labelsize=6)
# ax.legend(['Inter-Cluster','Intra-Cluster'],fontsize=6)
# ax.set_xlabel('Hamming Distance to Closest Sequence',fontsize=6)
# ax.set_ylabel('Fraction of Sequences',fontsize=6)

# intra_u_list = pd.DataFrame({'Distance':intra_u_list})
# intra_u_list['Pair Type']='Intra-Cluster'
# inter_u_list = pd.DataFrame({'Distance':inter_u_list})
# inter_u_list['Pair Type']='Inter-Cluster'
# u_list=pd.concat([intra_u_list,inter_u_list])

# ax = axs[0,1]
# sns.histplot(data=u_list,x='Distance',hue='Pair Type',stat='probability',ax=ax,legend=True)
# ax.tick_params(labelsize=6)
# ax.legend(['Inter-Cluster','Intra-Cluster'],fontsize=6)
# ax.set_xlabel(name+' Distance to Closest Sequence',fontsize=6)
# ax.set_ylabel('Fraction of Sequences',fontsize=6)


# intra_h_list,inter_h_list=[],[]
# intra_u_list,inter_u_list=[],[]

# for i in range(len(sub_sa_100_df)):
# 	seq = sub_sa_100_df.iloc[i]
# 	clust = seq.Cluster
# 	intra_clust_df = sub_sa_100_df[sub_sa_100_df['Cluster']==clust]
# 	intra_clust_df=intra_clust_df.drop(index=seq.name)
# 	h_distances = cdist([seq.Sequence],np.stack(intra_clust_df.Sequence),metric='hamming')
# 	intra_h_list.append(np.min(h_distances)*16)
# 	# u_distances = cdist([seq[['UMAP 1','UMAP 2']]], np.stack(intra_clust_df[['UMAP 1','UMAP 2']].values),metric='euclidean')
# 	u_distances = cdist([seq.Embedding],np.stack(intra_clust_df.Embedding),metric='euclidean')
# 	intra_u_list.append(np.min(u_distances))
# 	inter_clust_df = sub_sa_100_df[sub_sa_100_df['Cluster']!=clust]
# 	h_distances = cdist([seq.Sequence],np.stack(inter_clust_df.Sequence),metric='hamming')
# 	inter_h_list.append(np.min(h_distances)*16)
# 	# u_distances = cdist([seq[['UMAP 1','UMAP 2']]], np.stack(inter_clust_df[['UMAP 1','UMAP 2']].values),metric='euclidean')
# 	u_distances = cdist([seq.Embedding],np.stack(inter_clust_df.Embedding),metric='euclidean')
# 	inter_u_list.append(np.min(u_distances))

# intra_h_list = pd.DataFrame({'Distance':intra_h_list})
# intra_h_list['Pair Type']='Intra-Cluster'
# inter_h_list = pd.DataFrame({'Distance':inter_h_list})
# inter_h_list['Pair Type']='Inter-Cluster'
# h_list=pd.concat([intra_h_list,inter_h_list])

# ax = axs[1,0]
# ax=sns.histplot(data=h_list,x='Distance',hue='Pair Type',stat='probability',binrange=[-0.5,16.5],bins=17,ax=ax,legend=True)
# ax.tick_params(labelsize=6)
# ax.legend(['Inter-Cluster','Intra-Cluster'],fontsize=6)
# ax.set_xlabel('Hamming Distance to Closest Sequence',fontsize=6)
# ax.set_ylabel('Fraction of Sequences',fontsize=6)

# intra_u_list = pd.DataFrame({'Distance':intra_u_list})
# intra_u_list['Pair Type']='Intra-Cluster'
# inter_u_list = pd.DataFrame({'Distance':inter_u_list})
# inter_u_list['Pair Type']='Inter-Cluster'
# u_list=pd.concat([intra_u_list,inter_u_list])

# ax = axs[1,1]
# sns.histplot(data=u_list,x='Distance',hue='Pair Type',stat='probability',ax=ax,legend=True)
# ax.tick_params(labelsize=6)
# ax.legend(['Inter-Cluster','Intra-Cluster'],fontsize=6)
# ax.set_xlabel(name+' Distance to Closest Sequence',fontsize=6)
# ax.set_ylabel('Fraction of Sequences',fontsize=6)


# fig.tight_layout()
# fig.savefig('./cluster_distance_plot.png')
# plt.close()





#########################################################################
# fig,axs= plt.subplots(2,3,figsize=[9,6],dpi=600,sharex='row',sharey=True)

# seen_df =pd.read_pickle('./final_df/previous_seen_df.pkl')
# umap_names=[name+' UMAP 1',name+' UMAP 2']

# h_list,u_list=[],[]
# for i in range(len(sub_sa_100_df)):
# 	seq = sub_sa_100_df.iloc[i]
# 	h_dists = cdist([seq.Sequence],np.stack(sub_ns_100_df.Sequence),metric='hamming')
# 	h_list.append(np.min(h_dists)*16)
# 	# u_dists = cdist([seq[['UMAP 1','UMAP 2']]],np.stack(sub_ns_100_df[['UMAP 1','UMAP 2']].values))
# 	u_dists = cdist([seq.Embedding],np.stack(sub_ns_100_df.Embedding),metric='euclidean')
# 	u_list.append(np.min(u_dists))

# ax = axs[0,0]
# sns.histplot(x=h_list,stat='probability',binrange=[-0.5,16.5],bins=17,ax=ax)
# ax.tick_params(labelsize=6)
# ax.set_xlabel('Hamming Distance to Closest Sequence',fontsize=6)
# ax.set_ylabel('Fraction of Sequences',fontsize=6)
# ax.set_title('Nested Sampling Vs. Simulated Annealing',fontsize=6)

# ax = axs[1,0]
# sns.histplot(x=u_list,stat='probability',ax=ax)
# ax.tick_params(labelsize=6)
# ax.set_xlabel(name+' Distance to Closest Sequence',fontsize=6)
# ax.set_ylabel('Fraction of Sequences',fontsize=6)
# ax.set_title('Nested Sampling Vs. Simulated Annealing',fontsize=6)

# h_list,u_list=[],[]
# for i in range(len(sub_ns_100_df)):
# 	seq = sub_ns_100_df.iloc[i]
# 	h_dists = cdist([seq.Sequence],np.stack(seen_df.Sequence),metric='hamming')
# 	h_list.append(np.min(h_dists)*16)
# 	# u_dists = cdist([seq[['UMAP 1','UMAP 2']]],np.stack(seen_df[umap_names].values))
# 	u_dists=cdist([seq.Embedding],np.stack(seen_df[name+' Embedding']),metric='euclidean')
# 	u_list.append(np.min(u_dists))

# ax = axs[0,1]
# sns.histplot(x=h_list,stat='probability',binrange=[-0.5,16.5],bins=17,ax=ax)
# ax.tick_params(labelsize=6)
# ax.set_xlabel('Hamming Distance to Closest Sequence',fontsize=6)
# ax.set_ylabel('Fraction of Sequences',fontsize=6)
# ax.set_title('Nested Sampling Vs. Seen Sequences',fontsize=6)

# ax = axs[1,1]
# sns.histplot(x=u_list,stat='probability',ax=ax)
# ax.tick_params(labelsize=6)
# ax.set_xlabel(name+' Distance to Closest Sequence',fontsize=6)
# ax.set_ylabel('Fraction of Sequences',fontsize=6)
# ax.set_title('Nested Sampling Vs. Seen Sequences',fontsize=6)

# h_list,u_list=[],[]
# for i in range(len(sub_sa_100_df)):
# 	seq = sub_sa_100_df.iloc[i]
# 	h_dists = cdist([seq.Sequence],np.stack(seen_df.Sequence),metric='hamming')
# 	h_list.append(np.min(h_dists)*16)
# 	# u_dists = cdist([seq[['UMAP 1','UMAP 2']]],np.stack(seen_df[umap_names].values))
# 	u_dists=cdist([seq.Embedding],np.stack(seen_df[name+' Embedding']),metric='euclidean')
# 	u_list.append(np.min(u_dists))

# ax = axs[0,2]
# sns.histplot(x=h_list,stat='probability',binrange=[-0.5,16.5],bins=17,ax=ax)
# ax.tick_params(labelsize=6)
# ax.set_xlabel('Hamming Distance to Closest Sequence',fontsize=6)
# ax.set_ylabel('Fraction of Sequences',fontsize=6)
# ax.set_title('Simulated Annealing Vs. Seen Sequences',fontsize=6)

# ax = axs[1,2]
# sns.histplot(x=u_list,stat='probability',ax=ax)
# ax.tick_params(labelsize=6)
# ax.set_xlabel(name+' Distance to Closest Sequence',fontsize=6)
# ax.set_ylabel('Fraction of Sequences',fontsize=6)
# ax.set_title('Simulated Annealing Vs. Seen Sequences',fontsize=6)

# fig.tight_layout()
# fig.savefig('./distance_plot.png')
# plt.close()

# sub_sa_100_df.to_pickle('./final_df/simulated_annealing_df_'+name+'.pkl')
# sub_ns_100_df.to_pickle('./final_df/nested_sampling_df_'+name+'.pkl')

#####################################################

# if name =='DevRep':
# 	high_cluster = np.max(np.unique(ns_100_df.Cluster))
# elif name=='UniRep':
# 	high_cluster = 2
# high_ns_df = ns_100_df[ns_100_df.Cluster == high_cluster]
# high_ns_df = add_cluster(high_ns_df,clust_frac=0.01)
# sub_high_ns_df = even_subsample(high_ns_df)

# fig,axs= plt.subplots(2,2,figsize=[6,6],dpi=1200,sharey='col',sharex='col')

# ax=axs[0,0]
# cmap=sns.color_palette('hls',n_colors=len(np.unique(high_ns_df['Cluster'])))
# sns.scatterplot(data=high_ns_df,x='UMAP 1',y='UMAP 2',hue='Cluster',ax=ax,legend=None,palette=cmap,marker='.',edgecolor=None,linewidth=0,alpha=1)
# ax.tick_params(labelsize=6)
# ax.set_xlabel('UMAP 1',fontsize=6)
# ax.set_ylabel('UMAP 2',fontsize=6)
# ax.set_title('Top Cluster From Nested Sampling', fontsize=6)

# ax = axs[0,1]
# sns.violinplot(data=high_ns_df,x='Cluster',y='Develop', palette=cmap, scale='width',inner='quartile',saturation=1,ax=ax)
# ax.tick_params(labelsize=6)
# ax.set_xlabel('Sub-Cluster',fontsize=6)
# ax.set_ylabel('Developability',fontsize=6)
# ax.set_title('Top Cluster From Nested Sampling',fontsize=6)

# ax=axs[1,0]
# sns.scatterplot(data=sub_high_ns_df,x='UMAP 1',y='UMAP 2',hue='Cluster',ax=ax,legend=None,palette=cmap,marker='.',edgecolor=None,linewidth=0,alpha=1)
# ax.tick_params(labelsize=6)
# ax.set_xlabel('UMAP 1',fontsize=6)
# ax.set_ylabel('UMAP 2',fontsize=6)

# ax = axs[1,1]
# sns.violinplot(data=sub_high_ns_df,x='Cluster',y='Develop', palette=cmap, scale='width',inner='quartile',saturation=1,ax=ax)
# ax.tick_params(labelsize=6)
# ax.set_xlabel('Sub-Cluster',fontsize=6)
# ax.set_ylabel('Developability',fontsize=6)

# heatmap_per_cluster('subcluster_nested_sampling',cmap,sub_high_ns_df)

# fig.tight_layout()
# fig.savefig('./high_yield_subcluster_plot.png')
# plt.close()

# sub_high_ns_df.to_pickle('./final_df/high_yield_nested_sampling_df_'+name+'.pkl')
