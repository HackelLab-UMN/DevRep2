import sys
import submodels_module as modelbank
from itertools import combinations
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind_from_stats as ttest
import load_format_data
import numpy as np
import pandas as pd 


ss_list=[0.01,0.1,0.5,1]

df_list=[]
for ss in ss_list:
	model = modelbank.seq_to_assay_model([1,8,10],'emb_cnn',ss)
	df = pd.DataFrame({'Sample Size':ss, 'Model':'Convolutional', 'Dataset':'CV','Loss':model.model_stats['cv_avg_loss'],'Error':model.model_stats['cv_std_loss']},index=[0])
	df_list.append(df)
	df = pd.DataFrame({'Sample Size':ss, 'Model':'Convolutional', 'Dataset':'Test','Loss':model.model_stats['test_avg_loss'],'Error':model.model_stats['test_std_loss']},index=[0])
	df_list.append(df)

	model = modelbank.seq_to_assay_model([1,8,10],'fnn',ss)
	df = pd.DataFrame({'Sample Size':ss, 'Model':'One Hot\nNon-Linear', 'Dataset':'CV','Loss':model.model_stats['cv_avg_loss'],'Error':model.model_stats['cv_std_loss']},index=[0])
	df_list.append(df)
	df = pd.DataFrame({'Sample Size':ss, 'Model':'One Hot\nNon-Linear', 'Dataset':'Test','Loss':model.model_stats['test_avg_loss'],'Error':model.model_stats['test_std_loss']},index=[0])
	df_list.append(df)

	model = modelbank.control_to_assay_model([1,8,10],'ridge',ss)
	df = pd.DataFrame({'Sample Size':ss, 'Model':'Assay Only\nControl', 'Dataset':'CV','Loss':model.model_stats['cv_avg_loss'],'Error':model.model_stats['cv_std_loss']},index=[0])
	df_list.append(df)
	df = pd.DataFrame({'Sample Size':ss, 'Model':'Assay Only\nControl', 'Dataset':'Test','Loss':model.model_stats['test_avg_loss'],'Error':model.model_stats['test_std_loss']},index=[0])
	df_list.append(df)

	exploded_df,_,_=load_format_data.explode_assays([1,8,10],model.training_df)
	cv_exp_var=np.average(np.square(np.array(exploded_df['y_std'])))
	df = pd.DataFrame({'Sample Size':ss, 'Model':'Experimental\nVariance', 'Dataset':'CV','Loss':cv_exp_var,'Error':0},index=[0])
	df_list.append(df)
	exploded_df,_,_=load_format_data.explode_assays([1,8,10],model.testing_df)
	test_exp_var=np.average(np.square(np.array(exploded_df['y_std'])))
	df = pd.DataFrame({'Sample Size':ss, 'Model':'Experimental\nVariance', 'Dataset':'Test','Loss':test_exp_var,'Error':0},index=[0])
	df_list.append(df)

fig, axs = plt.subplots(2,2,figsize=[4,4],sharex='row',sharey='row',dpi=600)

df = pd.concat(df_list,ignore_index=True)
df['Sample Size']=(df['Sample Size'].values*45433).astype(int)

mdl_list = ['Convolutional', 'One Hot\nNon-Linear', 'Assay Only\nControl', 'Experimental\nVariance']
color_list = ['black','C1','C2','C3']

ax = axs[0,0]
for mdl,color in zip(mdl_list,color_list):
	df_loc=df.loc[(df['Dataset']=='CV') & (df['Model']==mdl)]
	ax.errorbar(x=df_loc['Sample Size'].values,y=df_loc['Loss'],yerr=df_loc['Error'],label=mdl,c=color,lw=1,capsize=1)
# ax.legend(fontsize=6,framealpha=1)
ax.legend().set_visible(False)
ax.tick_params(labelsize=6,which='both')
ax.set_ylabel('CV Loss (MSE)',fontsize=6)
ax.set_xlabel('HT Assay Sample Size',fontsize=6)
ax.set_ylim([0.01,0.045])
ax.set_yticks([0.01,0.02,0.03,0.04])


ax = axs[0,1]
for mdl,color in zip(mdl_list,color_list):
	df_loc=df.loc[(df['Dataset']=='Test') & (df['Model']==mdl)]
	ax.errorbar(x=df_loc['Sample Size'].values,y=df_loc['Loss'],yerr=df_loc['Error'],label=mdl,c=color,lw=1,capsize=1)
ax.legend().set_visible(False)
ax.tick_params(labelsize=6)
ax.set_ylabel('Test Loss (MSE)',fontsize=6)
ax.set_xlabel('HT Assay Sample Size',fontsize=6)
ax.set_xscale('log')
ax.set_yscale('log')


####################################################
c_ss_list=[0.01,0.1,1]
d_ss_list=[0.055,0.1,0.2,0.3,0.5,1]
d_arch_list = ['ridge','svm','forest']

df_list=[]

### devrep
for c_ss in c_ss_list:
    c_prop=[[1,8,10],'emb_cnn',c_ss]
    for d_ss in d_ss_list:
        cv_loss_list,test_loss_list=[],[]
        for i in range(10):
            min_cv_loss = np.inf
            for d_arch in d_arch_list:
                model = modelbank.sequence_embeding_to_yield_model(c_prop+[i],d_arch,d_ss)
                # model.limit_test_set([1,8,10])
                # model.test_model()
                if model.model_stats['cv_avg_loss'] < min_cv_loss:
                    min_cv_loss=model.model_stats['cv_avg_loss']
                    best_model = model
            cv_loss_list.append(min_cv_loss)
            test_loss_list.append(best_model.model_stats['test_avg_loss'])
        df = pd.DataFrame({'HT Assay Sample Size':c_ss,'Yield Sample Size':d_ss, 'Model':'Convolutional', 'Dataset':'CV','Loss':np.average(cv_loss_list),'Error':np.std(cv_loss_list)},index=[0])
        df_list.append(df)
        df = pd.DataFrame({'HT Assay Sample Size':c_ss,'Yield Sample Size':d_ss, 'Model':'Convolutional', 'Dataset':'Test','Loss':np.average(test_loss_list),'Error':np.std(test_loss_list)},index=[0])
        df_list.append(df)

### one hot
for d_ss in d_ss_list:
    oh_df_list=[]
    for d_arch in d_arch_list:
        model = modelbank.seq_to_yield_model(d_arch,d_ss)
        df = pd.DataFrame({'Yield Sample Size':d_ss, 'Model':'One Hot','Arch':d_arch, 'Dataset':'CV','Loss':model.model_stats['cv_avg_loss'],'Seed':-1},index=[0])
        oh_df_list.append(df)
        df = pd.DataFrame({'Yield Sample Size':d_ss, 'Model':'One Hot','Arch':d_arch, 'Dataset':'Test','Loss':model.model_stats['test_avg_loss'],'Seed':-1},index=[0])
        oh_df_list.append(df)
        for seed in range(9):
            model.change_sample_seed(seed)
            df = pd.DataFrame({'Yield Sample Size':d_ss, 'Model':'One Hot','Arch':d_arch, 'Dataset':'CV','Loss':model.model_stats['cv_avg_loss'],'Seed':seed},index=[0])
            oh_df_list.append(df)
            df = pd.DataFrame({'Yield Sample Size':d_ss, 'Model':'One Hot','Arch':d_arch, 'Dataset':'Test','Loss':model.model_stats['test_avg_loss'],'Seed':seed},index=[0])
            oh_df_list.append(df)
    
    oh_df = pd.concat(oh_df_list,ignore_index=True)

    seed_cv_loss,seed_test_loss=[],[]
    for seed in range(-1,9,1):
        oh_df_loc = oh_df.loc[(oh_df['Seed']==seed) & (oh_df['Dataset']=='CV')]
        seed_cv_loss.append(min(oh_df_loc['Loss']))

        min_arch = oh_df_loc[oh_df_loc['Loss']==min(oh_df_loc['Loss'])]['Arch'].values[0]
        seed_test_loss.append(oh_df.loc[(oh_df['Seed']==seed) & (oh_df['Dataset']=='Test') & (oh_df['Arch']==min_arch)]['Loss'])
    df = pd.DataFrame({'Yield Sample Size':d_ss, 'Model':'One Hot', 'Dataset':'CV','Loss':np.average(seed_cv_loss),'Error':np.std(seed_cv_loss)},index=[0])
    df_list.append(df)
    df = pd.DataFrame({'Yield Sample Size':d_ss, 'Model':'One Hot', 'Dataset':'Test','Loss':np.average(seed_test_loss),'Error':np.std(seed_test_loss)},index=[0])
    df_list.append(df) 

### assay scores
for d_ss in d_ss_list:
    assay_df_list=[]
    for d_arch in d_arch_list:
        model = modelbank.assay_to_yield_model([1,8,10],d_arch,d_ss)
        df = pd.DataFrame({'Yield Sample Size':d_ss, 'Model':'Assay Score','Arch':d_arch, 'Dataset':'CV','Loss':model.model_stats['cv_avg_loss'],'Seed':-1},index=[0])
        assay_df_list.append(df)
        # model.limit_test_set([1,8,10])
        # model.test_model()
        df = pd.DataFrame({'Yield Sample Size':d_ss, 'Model':'Assay Score','Arch':d_arch, 'Dataset':'Test','Loss':model.model_stats['test_avg_loss'],'Seed':-1},index=[0])
        assay_df_list.append(df)
        for seed in range(9):
            model.change_sample_seed(seed)
            df = pd.DataFrame({'Yield Sample Size':d_ss, 'Model':'Assay Score','Arch':d_arch, 'Dataset':'CV','Loss':model.model_stats['cv_avg_loss'],'Seed':seed},index=[0])
            assay_df_list.append(df)
            # model.limit_test_set([1,8,10])
            # model.test_model()
            df = pd.DataFrame({'Yield Sample Size':d_ss, 'Model':'Assay Score','Arch':d_arch, 'Dataset':'Test','Loss':model.model_stats['test_avg_loss'],'Seed':seed},index=[0])
            assay_df_list.append(df)
    
    assay_df = pd.concat(assay_df_list,ignore_index=True)

    seed_cv_loss,seed_test_loss=[],[]
    for seed in range(-1,9,1):
        assay_df_loc = assay_df.loc[(assay_df['Seed']==seed) & (assay_df['Dataset']=='CV')]
        seed_cv_loss.append(min(assay_df_loc['Loss']))

        min_arch = assay_df_loc[assay_df_loc['Loss']==min(assay_df_loc['Loss'])]['Arch'].values[0]
        seed_test_loss.append(assay_df.loc[(assay_df['Seed']==seed) & (assay_df['Dataset']=='Test') & (assay_df['Arch']==min_arch)]['Loss'])
    df = pd.DataFrame({'Yield Sample Size':d_ss, 'Model':'Assay Score', 'Dataset':'CV','Loss':np.average(seed_cv_loss),'Error':np.std(seed_cv_loss)},index=[0])
    df_list.append(df)
    df = pd.DataFrame({'Yield Sample Size':d_ss, 'Model':'Assay Score', 'Dataset':'Test','Loss':np.average(seed_test_loss),'Error':np.std(seed_test_loss)},index=[0])
    df_list.append(df) 

### controls 
for d_ss in d_ss_list:
    cv_loss_list,test_loss_list=[],[]
    cv_var_list,test_var_list=[],[]
    model = modelbank.control_to_yield_model('ridge',d_ss)
    cv_loss_list.append(model.model_stats['cv_avg_loss'])
    test_loss_list.append(model.model_stats['test_avg_loss'])

    cv_df_seed=load_format_data.sub_sample(model.training_df,model.sample_fraction,model.sample_seed)
    exploded_df,_,_=load_format_data.explode_yield(cv_df_seed)
    cv_var_list.append(np.average(np.square(np.array(exploded_df['y_std']))))

    model.limit_test_set([1,8,10])
    # model.test_model()
    exploded_df,_,_=load_format_data.explode_yield(model.testing_df)
    test_var_list.append(np.average(np.square(np.array(exploded_df['y_std']))))

    for i in range(9):
        model.change_sample_seed(i)
        cv_loss_list.append(model.model_stats['cv_avg_loss'])
        test_loss_list.append(model.model_stats['test_avg_loss'])

        cv_df_seed=load_format_data.sub_sample(model.training_df,model.sample_fraction,model.sample_seed)
        exploded_df,_,_=load_format_data.explode_yield(cv_df_seed)
        cv_var_list.append(np.average(np.square(np.array(exploded_df['y_std']))))

        model.limit_test_set([1,8,10])
        # model.test_model()
        exploded_df,_,_=load_format_data.explode_yield(model.testing_df)
        test_var_list.append(np.average(np.square(np.array(exploded_df['y_std']))))
    df = pd.DataFrame({'Yield Sample Size':d_ss, 'Model':'Strain Only\nControl', 'Dataset':'CV','Loss':np.average(cv_loss_list),'Error':np.std(cv_loss_list)},index=[0])
    df_list.append(df)
    df = pd.DataFrame({'Yield Sample Size':d_ss, 'Model':'Strain Only\nControl', 'Dataset':'Test','Loss':np.average(test_loss_list),'Error':np.std(test_loss_list)},index=[0])
    df_list.append(df) 
    df = pd.DataFrame({'Yield Sample Size':d_ss, 'Model':'Experimental\nVariance', 'Dataset':'CV','Loss':np.average(cv_var_list),'Error':np.std(cv_var_list)},index=[0])
    df_list.append(df)
    df = pd.DataFrame({'Yield Sample Size':d_ss, 'Model':'Experimental\nVariance', 'Dataset':'Test','Loss':np.average(test_var_list),'Error':np.std(test_var_list)},index=[0])
    df_list.append(df) 

df = pd.concat(df_list,ignore_index=True)
df['Yield Sample Size']=(df['Yield Sample Size'].values*195).astype(int)


ax = axs[1,0]

df_loc = df[df['Model']=='Convolutional']
ls_list = ['-','--',':']
for c_ss,ls in zip(c_ss_list,ls_list):
    df_loc_loc = df_loc.loc[(df_loc['HT Assay Sample Size']==c_ss) & (df_loc['Dataset']=='CV')]
    ax.errorbar(x=df_loc_loc['Yield Sample Size'],y=df_loc_loc['Loss'],yerr=df_loc_loc['Error'],c='black',ls=ls, lw=1,label ='Convolutional N='+str(int(c_ss*45433)),capsize=1)

for model_name in ['Assay Score','One Hot','Strain Only\nControl','Experimental\nVariance']:
    df_loc = df.loc[(df['Model']==model_name) & (df['Dataset']=='CV')]
    ax.errorbar(x=df_loc['Yield Sample Size'],y=df_loc['Loss'],yerr=df_loc['Error'],label=model_name, lw=1, capsize=1)
# ax.legend(fontsize=6,framealpha=1)
# # ax.legend().set_visible(False)
ax.tick_params(labelsize=6,which='both')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('CV Loss (MSE)',fontsize=6)
ax.set_xlabel('Yield Sample Size',fontsize=6)

ax = axs[1,1]
df_loc = df[df['Model']=='Convolutional']
ls_list = ['-','--',':']
for c_ss,ls in zip(c_ss_list,ls_list):
    df_loc_loc = df_loc.loc[(df_loc['HT Assay Sample Size']==c_ss) & (df_loc['Dataset']=='Test')]
    ax.errorbar(x=df_loc_loc['Yield Sample Size'],y=df_loc_loc['Loss'],yerr=df_loc_loc['Error'],c='black',ls=ls, lw=1, label ='Convolutional N='+str(int(c_ss*45433)),capsize=1)

for model_name in ['Assay Score','One Hot','Strain Only\nControl','Experimental\nVariance']:
    df_loc = df.loc[(df['Model']==model_name) & (df['Dataset']=='Test')]
    ax.errorbar(x=df_loc['Yield Sample Size'],y=df_loc['Loss'],yerr=df_loc['Error'],label=model_name, lw=1,capsize=1)
# ax.legend(fontsize=6,framealpha=1)
ax.legend().set_visible(False)
ax.tick_params(labelsize=6)
ax.set_ylabel('Test Loss (MSE)',fontsize=6)
ax.set_xlabel('Yield Sample Size',fontsize=6)

fig.tight_layout()
fig.savefig('./change_ss.png')
plt.close()