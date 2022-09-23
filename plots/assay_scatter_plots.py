import submodels_module as mb
import pandas as pd 
import numpy as np 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import load_format_data

from sklearn.feature_selection import mutual_info_regression as mir
from scipy.stats import spearmanr as spr
from scipy.stats import ttest_ind_from_stats as ttest 


one_hot = mb.seq_to_yield_model('forest',1)
oh_loss=one_hot.model_stats['cv_avg_loss']
oh_std= one_hot.model_stats['cv_std_loss']

control_model=mb.control_to_yield_model('ridge',1)
control_loss=control_model.model_stats['cv_avg_loss']
control_model.limit_test_set([1,8,10])
exploded_df,_,_=load_format_data.explode_yield(control_model.training_df)
exp_var=np.average(np.square(np.array(exploded_df['y_std'])))

df=pd.read_pickle('./datasets/assay_to_dot_training_data.pkl')


sorts= [['Ppk37','Sort1_mean_score',[1],'$P_{PK37}$'], ['Gsh','Sort8_mean_score',[8],'$G_{SH}$'], ['Bsh','Sort10_mean_score', [10],r'$\beta_{SH}$']]

for sort in sorts:
	fig,axs = plt.subplots(1,2,figsize=[4,2],dpi=1200)

	ax = axs[0]
	# df_iq = df[~df['IQ_Average_bc'].isnull()]
	# ax.scatter(x=df_iq[sort[1]],y=df_iq['IQ_Average_bc'],color='maroon',alpha=0.25)
	# mi_iq = mir(df_iq[sort[1]].to_numpy().reshape(-1,1),df_iq['IQ_Average_bc'].to_numpy())[0]
	# rho_iq = spr(df_iq[sort[1]],df_iq['IQ_Average_bc'])[0]
	# ax.annotate('Strain $I^q$:  '+r'$\rho$='+str(round(rho_iq,2))+ ', MI='+str(round(mi_iq,2)),(-0.025,2.6),fontsize=6,color='maroon')

	df_sh = df[~df['SH_Average_bc'].isnull()]
	ax.scatter(x=df_sh[sort[1]],y=df_sh['SH_Average_bc'],color='orange',alpha=0.25)
	mi_sh = mir(df_sh[sort[1]].to_numpy().reshape(-1,1),df_sh['SH_Average_bc'].to_numpy())[0]
	rho_sh = spr(df_sh[sort[1]],df_sh['SH_Average_bc'])[0]
	# ax.annotate('Strain SH:'+r'$\rho$='+str(round(rho_sh,2))+ ', MI='+str(round(mi_sh,2)),(-0.025,2.6),fontsize=6,color='orange')

	ax.set_xlabel('Assay Score',fontsize=6)
	ax.set_ylabel('Developability',fontsize=6)
	ax.tick_params(axis='both', which='major', labelsize=6)

	ax.set_xlim([-0.05,1.05])
	ax.set_ylim([-2,2.9])


	ax= axs[1]
	arch_list = ['ridge','svm','forest','fnn']
	assay_loss = np.inf
	for arch in arch_list:
		mdl = mb.assay_to_yield_model(sort[2],arch,1)
		if mdl.model_stats['cv_avg_loss'] < assay_loss:
			assay_loss = mdl.model_stats['cv_avg_loss']
			assay_std = mdl.model_stats['cv_std_loss']
	print(ttest(assay_loss,assay_std,10,control_loss,0,1))

	seqassay_loss = np.inf
	for arch in arch_list:
		mdl = mb.seqandassay_to_yield_model(sort[2],arch,1)
		if mdl.model_stats['cv_avg_loss'] < seqassay_loss:
			seqassay_loss = mdl.model_stats['cv_avg_loss']
			seqassay_std = mdl.model_stats['cv_std_loss']
	print(ttest(seqassay_loss,seqassay_std,10,control_loss,0,1))

	x=[-1,2]

	ax.axhline(control_loss,x[0],x[1],color='red',linestyle='--',label='Strain Control')
	ax.axhline(oh_loss,x[0],x[1],color='green',linestyle='--',label='Sequence Model')
	ax.axhline(exp_var,x[0],x[1],color='purple',linestyle='--',label='Experimental Variance')


	ax.bar(x=[0,1],height=[assay_loss, seqassay_loss], yerr=[assay_std, seqassay_std ], width=0.8, color='black',ecolor='gray',label=None,error_kw=dict(lw=5, capsize=5, capthick=3))
	ax.legend(fontsize=6)
	ax.set_xticks([0,1])
	ax.set_xticklabels([sort[3], sort[3]+' & Sequence'])
	ax.set_ylabel('CV Loss (MSE)',fontsize=6)
	ax.tick_params(axis='both', which='major', labelsize=6)
	ax.set_ylim([0.3,0.9])
	fig.tight_layout()
	fig.savefig('./'+sort[0]+'.png')
	plt.close()