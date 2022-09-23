import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 
import networkx as nx 
from sklearn.neighbors import NearestNeighbors 
import pickle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import umap 
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from sequence_models import get_devrep_rep,get_unirep_rep,get_potts_rep,get_onehot_rep
from functools import partial 
import multiprocessing 
from sklearn.preprocessing import minmax_scale


def subset_df(threshold_df):
	'''
	Input: 
	threshold df 2col: a pandas df that is: df[:,0]=sequences and df[:,1]=threshold developabilities for the NS step
	
	Output: 
	threshold df (modified) 2col: df[:,0]=UNIQUE sequences and df[:,1]=dev thresholds corresponding to the unique sequences for the NS step
	'''
	seqs = np.stack(threshold_df['Sequence'])
	_, unique_idx = np.unique(seqs,return_index=True, axis=0)
	threshold_df=threshold_df.iloc[unique_idx].sort_index()
	# if len(threshold_df)>2000:
		# threshold_df_middle=threshold_df.iloc[np.linspace(0,len(threshold_df)-250,250,endpoint=True,dtype=int)]
		# threshold_df_end=threshold_df.iloc[-249:]
		# threshold_df= pd.concat([threshold_df_middle,threshold_df_end])
		# threshold_df=threshold_df.iloc[np.linspace(0,len(threshold_df)-1,2000,endpoint=True,dtype=int)]
	return threshold_df

def get_graph_rep(NS_run,threshold_df):
	'''
	Input: 
	NS_run = object that aggregates nested sampling step data; used here to get the model to embed sequences
	threshold df (modified) 2col = df[:,0]=UNIQUE sequences and df[:,1]=dev thresholds corresponding to the unique sequences for the NS step
	

	Output:
	threshold df (modified) 3col = df[:,0]=UNIQUE sequences; df[:,1]= corresponding dev thresholds; 
	df[:,2]=corresponding sequences' embeddings using the corresponding NS_run model
	'''
	if 'devrep' in NS_run.model_name:
		rep_fxn = get_devrep_rep
	elif 'unirep_paratope' in NS_run.model_name:
		rep_fxn = get_unirep_rep
	elif 'onehot' in NS_run.model_name:
		rep_fxn = get_onehot_rep
	elif 'potts' in NS_run.model_name:
		rep_fxn = partial(get_potts_rep,NS_run.q)
	reps=[list(x) for x in rep_fxn(np.stack(threshold_df['Sequence']))]
	sequence = threshold_df.Sequence.to_numpy()
	dev =threshold_df.Develop.to_numpy()
	threshold_df=pd.DataFrame({'Sequence':sequence,'Develop':dev,'Embedding':reps})
	return threshold_df

def get_neighbors(threshold_df,i):
	'''
	Inputs:
	threshold_df (modified) = contains unique threshold sequences df[0,:], threshold developabilities df[1,:], 
	and threshold embeddings df[2,:]
	These values correspond to the sequences for a given nested sampling step (along the 50 steps)
	that have the lowest corresponding developability in the step of all other sequences with
	desired embeddings of corresponding unique sequences

	i = the current nested sampling step number recorded in loops of 50 within /ns_walkers/

	Outputs: 
	edges = list of tuples of (current nested sampling step i , ); will have either 1 or 2 entries (neighbors);
	these edges will be used to eventually construct a graph of all of the NS runs that will be connected if: 
	1) developabilities are lower than some cutoff, and 
	2) the embeddings for the corresponding NS_run are the closest for 0-2 neighboring "node" NS_runs 
	'''
	if i%50==0:
		print(i,'/',len(threshold_df))

	n_neighbors=2
	A=[]

	cur_develop=threshold_df.iloc[i]['Develop']

	if cur_develop == min(threshold_df['Develop']):
		return []

	seqs_lower = np.stack(threshold_df[threshold_df['Develop']<cur_develop]['Embedding'])

	if len(seqs_lower)<n_neighbors:
			n_neighbors=1
	#find closest neighbor
	knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean',n_jobs=1,algorithm='brute').fit(seqs_lower)
	d, nn = knn.kneighbors(np.stack([threshold_df.iloc[i]['Embedding']]),return_distance=True)
	# print(d,nn)

	#connect to all sequences that distance away
	neigh = NearestNeighbors(radius=d[0][-1], metric='euclidean',n_jobs=1,algorithm='brute').fit(seqs_lower)
	A = neigh.radius_neighbors(np.stack([threshold_df.iloc[i]['Embedding']]),return_distance=False)[0]
	if len(A)<len(nn[0]):
		A = nn[0]

	edges = []
	for a in A:
		edges.append((i,a))

	return edges
		

def make_graph(NS_run):
	'''
	Input: 


	Output:

	'''
	print('making graph')
	threshold_df = pd.read_pickle('./ns_threshold/'+NS_run.savename+'.pkl')#load threshold df of 2 cols: sequences, threshold dev
	threshold_df = subset_df(threshold_df)#prune threshold df s.t. df={unique sequences, threshold dev}
	threshold_df = get_graph_rep(NS_run,threshold_df)#extend threshold df s.t. df={unique sequences, threshold dev, embedding}

	# dos_df=pd.read_pickle('./ns_dos/'+NS_run.savename+'.pkl')

	G= nx.Graph()
	G.add_nodes_from(list(range(len(threshold_df))))#initialize graph G as all NS steps within the NS_run for which there are unique sequences

	get_neighbors_filled=partial(get_neighbors,threshold_df.copy())#set the automatic input of get_neighbors to current 3col threshold_df(_, node i)
	pool=multiprocessing.Pool(processes=32)#setup 32 different threads
	i=list(range(len(threshold_df)))#create list of all NS steps we want to investigate within the NS_run: these are nodes we will investigate
	# edge_list=[]
	# for i in range(0,len(threshold_df)):
	# 	edge_list.append(get_neighbors_filled(i))
	(edge_list)=pool.map(get_neighbors_filled,i)#use the thread to find the neighbors of every query node 'i' and return all of them in a list
	pool.close()#close the thread: we don't need it anymore 

	#assign the edges for all of the nodes in the nx.Graph() object using the edges we just found (in parallel)
	for edges in edge_list:
		G.add_edges_from(edges)

	#save the graph object we just made within /graphs/
	with open('./graphs/'+NS_run.savename+'.pkl','wb') as f:
		pickle.dump(G.copy(),f)


def plot_graph_3d(NS_run):
	print('plotting 3d graph')
	threshold_df = pd.read_pickle('./ns_threshold/'+NS_run.savename+'.pkl')
	threshold_df = subset_df(threshold_df)
	threshold_df = get_graph_rep(NS_run,threshold_df)


	dos_df=pd.read_pickle('./ns_dos/'+NS_run.savename+'.pkl')

	with open('./graphs/'+NS_run.savename+'.pkl','rb') as f:
		G=pickle.load(f)
	cmap=plt.cm.gist_rainbow
	norm = mpl.colors.Normalize(vmin=min(threshold_df['Develop']),vmax=max(threshold_df['Develop']))
	node_color=np.array([cmap(norm(threshold_df.iloc[node]['Develop'])) for node in G.nodes])
	
	node_size =np.array([dos_df[dos_df['Develop']==threshold_df.iloc[node]['Develop']].Cum_DoS for node in G.nodes])
	node_size = np.log10(node_size)
	node_size = minmax_scale(node_size,feature_range=(0.1,10))
	print(max(node_size))
	print(min(node_size))

	# node_size = (node_size-min(node_size)+1)*2

	# node_size = (node_size-min(node_size)+`2)**(2)

	pos=nx.random_layout(G,dim=3,seed=42)
	if NS_run.q > 2:
		reducer=umap.UMAP(metric='euclidean',random_state=420)
		seq_xy=reducer.fit_transform(np.stack(threshold_df['Embedding']))
		pickle.dump(reducer,open('./umap_reducers/'+NS_run.savename+'.pkl','wb'))
	else:
		seq_xy=np.stack(threshold_df['Embedding'])
		seq_xy[:,1]=0
	seq_z=threshold_df['Develop'].to_numpy()
	for i in range(len(pos)):
		pos[i]=[seq_xy[i,0],seq_xy[i,1],seq_z[i]]

	threshold_df['x']=seq_xy[:,0]
	threshold_df['y']=seq_xy[:,1]
	threshold_df['z']=seq_z[:]
	threshold_df.to_pickle('./3d_graphs/'+NS_run.savename+'.pkl')
	# df = pd.read_pickle('./3d_graphs/'+a.savename+'.pkl')


	node_xyz = np.array([pos[v] for v in G.nodes()])
	edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

	fig = plt.figure(dpi=1200,figsize=[4,3])
	gs = GridSpec(3, 3, figure=fig)
	if 'potts' in NS_run.model_name:
		dev_label='Energy'
	else:
		dev_label='Developability'

	ax = fig.add_subplot(gs[:,0:2], projection="3d")

	ax.scatter(*node_xyz.T, s=node_size, c=node_color)
	# for vizedge in edge_xyz:
	# 	ax.plot(*vizedge.T, color="tab:gray")

	ax.grid(False)
	# for dim in (ax.xaxis, ax.yaxis):
	# 	dim.set_ticks([])
	ax.set_xlabel("UMAP 1",fontsize=6)
	ax.set_ylabel("UMAP 2",fontsize=6)

	ax.set_zlabel(dev_label,fontsize=6)
	ax.tick_params(labelsize=6,which='both',colors='black')

	# ax = fig.add_subplot(gs[0,2])
	# ax.scatter(seq_xy[:,0],seq_z,s=node_size,c=node_color,alpha=0.1)
	# ax.grid(False)
	# ax.set_xlabel("UMAP 1",fontsize=6)
	# ax.set_box_aspect(1)
	# ax.set_ylabel(dev_label,fontsize=6)
	# ax.tick_params(labelsize=6,which='both',colors='black')

	# ax = fig.add_subplot(gs[1,2])
	# ax.scatter(seq_xy[:,1],seq_z,s=node_size,c=node_color,alpha=0.1)
	# ax.grid(False)
	# ax.set_xlabel("UMAP 2",fontsize=6)
	# ax.set_box_aspect(1)
	# ax.set_ylabel(dev_label,fontsize=6)
	# ax.tick_params(labelsize=6,which='both',colors='black')

	# ax = fig.add_subplot(gs[2,2])
	# ax.scatter(seq_xy[:,0],seq_xy[:,1],s=node_size,c=node_color,alpha=0.1)
	# ax.grid(False)
	# ax.set_xlabel("UMAP 1",fontsize=6)
	# ax.set_box_aspect(1)
	# ax.set_ylabel("UMAP 2",fontsize=6)
	# ax.tick_params(labelsize=6,which='both',colors='black')


	fig.tight_layout(w_pad=5)
	# plt.subplots_adjust()
	fig.savefig('./3d_graphs/'+NS_run.savename+'_3d_graph.png')
	plt.close()


class discont_plot():
	def __init__(self,NS_run):
		print('plotting disconnectivity plot')
		threshold_df = pd.read_pickle('./ns_threshold/'+NS_run.savename+'.pkl')
		self.threshold_df = subset_df(threshold_df)
		self.dos_df=pd.read_pickle('./ns_dos/'+NS_run.savename+'.pkl')

		with open('./graphs/'+NS_run.savename+'.pkl','rb') as f:
			G=pickle.load(f)
		point_list=[]
		point_list = self.get_points(G.copy(),0,1,1,point_list) #[[xi, Emax],[...],...] where Emax=threshold_dev for the NS-step within NS_run
		point_list = np.array(point_list)

		#first: sort the point list entries according to the xi from lhs to rhs 0-1 line, return np array of indices
		#then: remake point list to order the elts according to xi lhs to rhs
		point_list = point_list[np.argsort(point_list[:,0])]

		fig, ax = plt.subplots(1,1,figsize=[2, 2],dpi=1200)

		ax.plot(point_list[:,0],point_list[:,1],linewidth=0.5,color='black',marker='.',markersize=0.5)
		ax.tick_params(labelsize=6,which='both',colors='black')

		ax.set_ylim(bottom=-1.5,top=2.5)#trying to standardize the axes bound

		ax2=ax.secondary_yaxis('right',functions=(self.fwd,self.rev))
		ax2.tick_params(labelsize=6,which='both',colors='black')
		# ax2.set_yscale('log')
		ax2_tick_list=[]
		ax2_tick_lables=[]
		exp=0
		# while 10**exp > min(self.dos_df['Cum_DoS']):

		#assign the secondary axis values we'll interpolate to get dev and plot corresponding dev
		for exp in [0, -1, -5, -20]:#[0, -1, -20]:#[0, -1, -5, -20] #[0, -1, -2, -5, -10 , -20]
			ax2_tick_list.append(exp)
			ax2_tick_lables.append("$10^{" + str(exp) + "}$")
			exp=exp-1 #?? why bother modifying exp item in the list only to move on to the next elt???

		ax2.set_yticks(ax2_tick_list)
		ax2.set_yticklabels(ax2_tick_lables)

		ax.set_xticks([])
		ax.set_xlabel('Configuration Space',fontsize=6)
		if 'potts' in NS_run.model_name:
			ax.set_ylabel('Energy',fontsize=6)
			ax2.set_ylabel('Fraction of Space With Higher Energy',fontsize=6,color='black')
		else:
			ax.set_ylabel('Developability',fontsize=6)
			ax2.set_ylabel('Fraction of Space\nWith Higher Developability',fontsize=6,color='black')
		ax.spines['top'].set_visible(False)

		fig.tight_layout()
		image_name='./discont/'+NS_run.savename+'_discont.png'
		fig.savefig(image_name)

	def get_points(self,G,x_min,x_max,frac_of_seq,point_list):
		'''
		Inputs:
		G = the (hopefully 1 single fully connected graph) of the NS run from make_graph(): every node is a NS step;
		each node has 1-2 edges that connect to NS step nodes for which embeddings were closest
		and the corresponding developabilities were at or below the original node's treshold developability

		x_min and x_max = left to right location of the disconnectivity plot
		frac_of_seq = help for when the graph splits; this is set to 1 initially, then sets %config space for bounds for subgraphs
		point_list = empty list to start; then running list;  will populate with [[x, Emax],...] for every step in the ns_run in a 
		way amenable to plotting; 
		
		Note: Emax=(lowest)threshold dev for the ns_step within ns_run

		Output: 
		point_list (modified) = running list of [[x, Emax],...] for every step in the ns_run in a way amenable to plotting
		'''

		n_subgraphs=len(list(G.subgraph(c) for c in nx.connected_components(G))) 
		while n_subgraphs==1:
			E_max = self.threshold_df.iloc[min(G.nodes)]['Develop']#maximum energy=penalty=minimum developability
			width = sum(self.dos_df[self.dos_df['Develop']>=E_max]['DoS'])#width of config space depends on all dos that have energies worse than emax
			width_adj = (width*frac_of_seq)/2#this "helps for when the graph splits" using frac_of_seq=1 normally
			
			#find and record bounds in config space for the Emax (lowest developability)
			x_center = np.average([x_min,x_max])
			x_min = max(x_center-width_adj,x_min)#don't want to go below 0
			x_max = min(x_center+width_adj,x_max)#don't want to go above 1
			point_list.append([x_min, E_max])
			point_list.append([x_max, E_max])
			
			#we've recorded the lhs and rhs corresponding to emax, so now find, record, and 
			#remove all ns_run step nodes that have emax labels (the lowest threshold developability)
			nodes_to_remove = np.where(self.threshold_df['Develop']==E_max)[0]
			for n in nodes_to_remove:
				if n in G.nodes:
					G.remove_node(n)

			#now that we've removed nodes that we've recorded, check if we caused a split in the graph; record #subgraphs formed (if any)
			n_subgraphs=len(list(G.subgraph(c) for c in nx.connected_components(G))) 

			# if (x_max-x_min)<0.1 or len(G.nodes)<2:

			#if we only have 1 node left, we need to find/record this optimal point at the "peak" of the landscape as the center point 
			#along with the correspdonding developability which is determined here via max(remaining_nodes)
			if len(G.nodes)==1:
					point_list.append([x_center,self.threshold_df.iloc[max(G.nodes)]['Develop']])
					del(G)
					return point_list

		#now we've most likely broken 1 large graph into 2 or more subgraph (phases) and (of course) have more than 1 node total to examine
		if n_subgraphs>1:
			G_list=list(G.subgraph(c) for c in nx.connected_components(G))
			g_size_list = []
			
			#record number nodes in each of the subgraphs
			for g in G_list:
				g_size_list.append(len(g.nodes))


			g_frac_list = np.array(g_size_list)/sum(g_size_list)#find how much remaining room we should give to each subgraph based on the #nodes in each subgraph compared to total
			g_width = x_max-x_min#the remaining room to place nodes depends on the starting lhs/rhs of the first full graph

			g_idx = np.argsort(g_frac_list) #rank the relative graph #nodes' INDICES for each graph small to large
			x_min_g = x_min
			new_points_list=point_list
			
			#go over all graph fractions' INDICES, starting from small to large fractions
			for gidx in g_idx: 
				g = G_list[gidx] #find the current "smallest" graph out
				g_frac = g_frac_list[gidx] #find the corresponding smallest graph's fraction
				x_max_g = x_min_g + g_width*g_frac #find current available width of config space on rhs
				
				#if the rhs width of the subgroup's config space is beyond the rhs width of the original graph, set it to that of the original (full) graph
				if x_max_g > x_max: 
					x_max_g = x_max
				# if (x_max_g-x_min_g)>0.1 and len(g.nodes)>1:

				#now that we've resized bounds of the graph we'll draw, do a recursion on the subgraph we just redrew the bounds for:
				#now the frac_of_seq is "frac_of_seq (1 originally) * g_frac" in which g_frac is the fraction of config space the 
				#subgraph should take up (no long 1 since it's not the full graph)
				p=self.get_points(g.copy(),x_min_g,x_max_g,frac_of_seq*g_frac,[])

				#0 is false; 1 is true; checking if the list exists at all: are there any points associated with that subgraph we just examined?
				#record all points that we found in the get_points() call
				if p:
					for pt in p:
						new_points_list.append(pt)

				#this most likely shifts over the bounds for the next subgraph to be examined: 
				#the rhs of the current subgraph will be the lhs/lower bound on the proceeding subgraph
				x_min_g = x_max_g

			#update the current point list and delete the current list of graphs; we will consult the if statement above to recreate the 
			#remaining subgraphs and assign them to a new G_list
			point_list=new_points_list
			del(G_list)

		#most likely for safety: remove the current graph
		del(G)
		return point_list


	def fwd(self,x):
		return interp1d(self.dos_df['Develop'].values,np.log10(self.dos_df['Cum_DoS'].values),fill_value="extrapolate")(x)

	def rev(self,x):
		return interp1d(np.log10(self.dos_df['Cum_DoS'].values),self.dos_df['Develop'].values,fill_value="extrapolate")(x)