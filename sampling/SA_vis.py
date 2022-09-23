from SA_driver import SA_run
import pickle

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 

for ii in [1,2,3,4,5,6]:
	run = SA_run(ii)

	T_list = run.T_list
	i_list = list(range(0,500,50))+[499]

	df_list=[]

	running_i=0
	for T in T_list:
		for i in i_list:
			walkers,n_positions,_T,p_accept = pickle.load(open('./sa_walkers/'+run.savename+'_'+str(T)+'_'+str(i)+'.pkl', "rb"))
			seq_list=np.stack([walk.sequence for walk in walkers])
			unique_seq, unique_idx = np.unique(seq_list,axis=0,return_index=True)

			dev_list=[]
			for idx in unique_idx:
				dev_list.append(walkers[idx].develop)
			dev_mean = np.average(dev_list)
			dev_std = np.std(dev_list)

			df_loc = pd.DataFrame({'Average Yield':dev_mean,'Yield Stdev':dev_std,'Unique Sequences':len(unique_seq),'Step Size':n_positions,'Temp':T,'Acceptance Rate':p_accept},index=[i+running_i])
			df_list.append(df_loc)
			# print(np.average(dev_list), running_i+i)
		running_i = running_i + 500

	df_final = pd.concat(df_list)

	fig, ax = plt.subplots(1,1,figsize=[5,2],dpi=600)
	fig.subplots_adjust(left=0.25,right=0.55,bottom=0.2)

	x=np.array(df_final.index)
	ax.plot(x,df_final['Average Yield'],color='black',ms=0,ls='-',zorder=100)

	ax.fill_between(x,df_final['Average Yield']-df_final['Yield Stdev'],df_final['Average Yield']+df_final['Yield Stdev'],color='black',alpha=0.5,lw=0,zorder=99)

	ax.set_xlabel("Simulated Annealing Step",fontsize=6)
	ax.set_ylabel("Yield $\pm$ Stdev",fontsize=6)
	ax.tick_params(labelsize=6,which='both',colors='black')
	ax.set_zorder(10)
	ax.patch.set_visible(False)

	ax_2=ax.twinx()
	ax_2.plot(x,df_final['Temp'],ms=0,ls='-',color='red')
	ax_2.set_ylabel('Temperature',fontsize=6,color='red')
	ax_2.set_yscale('symlog',linthresh=0.001)
	ax_2.tick_params(axis='y',labelsize=6,which='both',colors='red')

	ax_3=ax.twinx()
	ax_3.spines['right'].set_position(("axes", 1.33))
	ax_3.plot(x,df_final['Step Size'],ms=0,ls='-',color='blue')
	ax_3.set_ylabel('Step Size',fontsize=6,color='blue')
	ax_3.tick_params(axis='y',labelsize=6,which='both',colors='blue')

	ax_4=ax.twinx()
	ax_4.spines['right'].set_position(("axes", 1.67))
	ax_4.plot(x,df_final['Acceptance Rate'],ms=0,ls='-',color='orange')
	ax_4.set_ylabel('Acceptance Rate',fontsize=6,color='orange')
	ax_4.tick_params(axis='y',labelsize=6,which='both',colors='orange')

	ax_5=ax.twinx()
	ax_5.spines['right'].set_visible(False)
	ax_5.spines['left'].set_position(("axes", -0.4))
	ax_5.spines['left'].set_visible(True)
	ax_5.yaxis.set_label_position('left')
	ax_5.yaxis.set_ticks_position('left')
	ax_5.plot(x,df_final['Unique Sequences'],ms=0,ls='-',color='green')
	ax_5.set_ylabel('Unique Sequences',fontsize=6,color='green')
	ax_5.tick_params(axis='y',labelsize=6,which='both',colors='green')


	# fig.tight_layout()
	fig.savefig('./sa_plots/'+run.savename+'_plot.png')
	plt.close()

