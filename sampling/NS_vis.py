from NS_driver import NS_run
import pickle

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 

# for ii in [8,9,10,1,2,3]:
for ii in [2]:
	run = NS_run(ii)
	df = pd.read_pickle('./ns_threshold/'+run.savename+'.pkl')
	ns_loop_max = max(df.index)
	ns_loop_list = list(range(0,ns_loop_max,50))+[ns_loop_max]

	df_list=[]

	for ns_loop in ns_loop_list:

		walkers,n_positions = pickle.load(open('./ns_walkers/'+run.savename+'_'+str(ns_loop)+'.pkl', "rb"))
		seq_list=np.stack([walk.sequence for walk in walkers])
		# if ns_loop>0:
		# 	seq_list=np.concatenate([seq_list,[df.loc[ns_loop].Sequence]]) #add back removed threshold sequence

		unique_seq, unique_idx = np.unique(seq_list,axis=0,return_index=True)
		dev_list=[]
		for idx in unique_idx:
			dev_list.append(walkers[idx].develop)
		
		if ns_loop>0:
			dev_list.append(df.loc[ns_loop].Develop)

		dev_mean = np.average(dev_list)
		dev_std = np.std(dev_list)

		df_loc = pd.DataFrame({'Average Yield':dev_mean,'Yield Stdev':dev_std,'Unique Sequences':len(dev_list),'Step Size':n_positions},index=[ns_loop])
		df_list.append(df_loc)


	df_final = pd.concat(df_list)

	fig, ax = plt.subplots(1,1,figsize=[5,2],dpi=600)
	fig.subplots_adjust(left=0.25,right=0.75,bottom=0.2)

	x=np.array(df_final.index)
	ax.plot(x,df_final['Average Yield'],color='black',ms=0,ls='-',zorder=100)

	ax.fill_between(x,df_final['Average Yield']-df_final['Yield Stdev'],df_final['Average Yield']+df_final['Yield Stdev'],color='black',alpha=0.5,lw=0,zorder=99)

	ax.set_xlabel("Nested Sampling Loop",fontsize=6)
	ax.set_ylabel("Yield $\pm$ Stdev",fontsize=6)
	ax.tick_params(labelsize=6,which='both',colors='black')
	ax.set_zorder(10)
	ax.patch.set_visible(False)


	ax_2=ax.twinx()
	ax_2.spines['right'].set_position(("axes", 1))
	ax_2.plot(df.index,df.Develop,ms=0,ls='--',color='red')
	ax_2.set_ylabel('Threshold Yield',fontsize=6,color='red')
	ax_2.tick_params(axis='y',labelsize=6,which='both',colors='red')

	ax_min = min(ax.get_ylim()[0],ax_2.get_ylim()[0])
	ax_max = max(ax.get_ylim()[1],ax_2.get_ylim()[1])

	ax.set_ylim([ax_min,ax_max])
	ax_2.set_ylim([ax_min,ax_max])

	ax_3=ax.twinx()
	ax_3.spines['right'].set_position(("axes", 1.25))
	ax_3.plot(x,df_final['Step Size'],ms=0,ls='-',color='blue')
	ax_3.set_ylabel('Step Size',fontsize=6,color='blue')
	ax_3.tick_params(axis='y',labelsize=6,which='both',colors='blue')

	ax_5=ax.twinx()
	ax_5.spines['right'].set_visible(False)
	ax_5.spines['left'].set_position(("axes", -0.25))
	ax_5.spines['left'].set_visible(True)
	ax_5.yaxis.set_label_position('left')
	ax_5.yaxis.set_ticks_position('left')
	ax_5.plot(x,df_final['Unique Sequences'],ms=0,ls='-',color='green')
	ax_5.set_ylabel('Unique Sequences',fontsize=6,color='green')
	ax_5.tick_params(axis='y',labelsize=6,which='both',colors='green')


	# fig.tight_layout()
	fig.savefig('./ns_plots/'+run.savename+'_ns_plot.png')
	plt.close()

