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
import seaborn as sns 


def main():

    toggle_no=int(sys.argv[1])

    if toggle_no>100:
    	toggle_no=toggle_no-100
    	gpu=True
    else:
    	gpu=False

    # c_models=['ridge','fnn','emb_fnn_flat','emb_fnn_maxpool','emb_fnn_maxpool_linear','emb_rnn','small_emb_rnn','small_emb_atn_rnn','small_emb_rnn_linear',
    #     'emb_cnn','small_emb_cnn','small_emb_atn_cnn','small_emb_cnn_linear']

    c_models=['emb_fnn_flat_dense_linear','emb_cnn_dense_linear','emb_rnn_dense_linear']
    c_prop=[[1,8,10],c_models[toggle_no],1]
    # c=modelbank.seq_to_assay_model(c_prop)
    # c.save_sequence_embeddings()

    for i in range(7,10):
    	if gpu:
		    d=modelbank.sequence_embeding_to_yield_model(c_prop+[i],'fnn',1)
		    d.cross_validate_model()
		    d.limit_test_set([1,8,10])
		    d.test_model()
    	else:
		    d=modelbank.sequence_embeding_to_yield_model(c_prop+[i],'ridge',1)
		    d.cross_validate_model()
		    d.limit_test_set([1,8,10])
		    d.test_model()

		    d=modelbank.sequence_embeding_to_yield_model(c_prop+[i],'forest',1)
		    d.cross_validate_model()
		    d.limit_test_set([1,8,10])
		    d.test_model()

		    d=modelbank.sequence_embeding_to_yield_model(c_prop+[i],'svm',1)
		    d.cross_validate_model()
		    d.limit_test_set([1,8,10])
		    d.test_model()


# if __name__ == '__main__':
#     main()

control_df_list=[]

model = modelbank.control_to_yield_model('ridge',1)
cv_loss,cv_std = model.model_stats['cv_avg_loss'],model.model_stats['cv_std_loss']
df = pd.DataFrame({'Model':'Strain Only\nControl', 'Dataset':'CV','Loss':cv_loss,'Error':cv_std},index=[0])
control_df_list.append(df)
# model.limit_test_set([1,8,10])
# model.test_model()
test_loss,test_std = model.model_stats['test_avg_loss'],model.model_stats['test_std_loss']
df = pd.DataFrame({'Model':'Strain Only\nControl', 'Dataset':'Test','Loss':test_loss,'Error':test_std},index=[0])
control_df_list.append(df)

model = modelbank.assay_to_yield_model([1,8,10],'forest',1)
cv_loss,cv_std = model.model_stats['cv_avg_loss'],model.model_stats['cv_std_loss']
df = pd.DataFrame({'Model':'Experimental\nAssay Scores', 'Dataset':'CV','Loss':cv_loss,'Error':cv_std},index=[0])
control_df_list.append(df)
model.limit_test_set([1,8,10])
test_loss,test_std = model.model_stats['test_avg_loss'],model.model_stats['test_std_loss']
df = pd.DataFrame({'Model':'Experimental\nAssay Scores', 'Dataset':'Test','Loss':test_loss,'Error':test_std},index=[0])
control_df_list.append(df)

exploded_df,_,_=load_format_data.explode_yield(model.training_df)
cv_exp_var=np.average(np.square(np.array(exploded_df['y_std'])))
df = pd.DataFrame({'Model':'Experimental\nVariance', 'Dataset':'CV','Loss':cv_exp_var,'Error':0},index=[0])
control_df_list.append(df)
exploded_df,_,_=load_format_data.explode_yield(model.testing_df)
test_exp_var=np.average(np.square(np.array(exploded_df['y_std'])))
df = pd.DataFrame({'Model':'Experimental\nVariance', 'Dataset':'Test','Loss':test_exp_var,'Error':0},index=[0])
control_df_list.append(df)

control_df = pd.concat(control_df_list)

model_df_list=[]
a_arch_list = ['ridge','forest','svm']

for a_arch in a_arch_list:
	model = modelbank.seq_to_yield_model(a_arch,1)
	cv_loss,cv_std = model.model_stats['cv_avg_loss'],model.model_stats['cv_std_loss']
	df = pd.DataFrame({'Embedding':'One Hot', 'Architecture':a_arch, 'Dataset':'CV','Loss':cv_loss,'Error':cv_std},index=[0])
	model_df_list.append(df)
	model.limit_test_set([1,8,10])
	# model.test_model()
	test_loss,test_std = model.model_stats['test_avg_loss'],model.model_stats['test_std_loss']
	df = pd.DataFrame({'Embedding':'One Hot', 'Architecture':a_arch, 'Dataset':'Test','Loss':test_loss,'Error':test_std},index=[0])
	model_df_list.append(df)



c_model_list = ['emb_fnn_flat','emb_rnn','emb_cnn']
c_model_emb_list = ['Flatten', 'Recurrent', 'Convolutional']

for c_model,c_emb in zip(c_model_list,c_model_emb_list):
	c_prop=[[1,8,10],c_model,1]
	for a_arch in a_arch_list:
		cv_loss_list,test_loss_list=[],[]
		for i in range(10):
			model = modelbank.sequence_embeding_to_yield_model(c_prop+[i],a_arch,1)
			cv_loss_list.append(model.model_stats['cv_avg_loss'])
			model.limit_test_set([1,8,10])
			# model.test_model()
			test_loss_list.append(model.model_stats['test_avg_loss'])
		df = pd.DataFrame({'Embedding':c_emb, 'Architecture':a_arch, 'Dataset':'CV','Loss':np.average(cv_loss_list),'Error':np.std(cv_loss_list)},index=[0])
		model_df_list.append(df)
		df = pd.DataFrame({'Embedding':c_emb, 'Architecture':a_arch, 'Dataset':'Test','Loss':np.average(test_loss_list),'Error':np.std(test_loss_list)},index=[0])
		model_df_list.append(df)

model_df = pd.concat(model_df_list)

fig , axs = plt.subplots(2,2, figsize=[5,4], sharey='row', sharex='col', gridspec_kw={'width_ratios': [3, 4]}, dpi=600)

ax = axs[0,0]
df_loc = control_df[control_df['Dataset']=='CV']
x=np.array(list(range(len(df_loc))))
ax.bar(x=x,height=df_loc['Loss'],yerr=df_loc['Error'],width=0.6,lw=0,ecolor='black',color='black',alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(df_loc.Model.values)
ax.tick_params(axis='both', which='major', labelsize=6)
ax.set_ylabel('CV Loss (MSE)',fontsize=6)
ax.set_ylim([0.3,0.85])

ax = axs[0,1]
df_loc = model_df[model_df['Dataset']=='CV']
df_loc_loc = df_loc[df_loc['Architecture']=='ridge']
x=np.array(list(range(len(df_loc_loc))))
ax.bar(x-0.2,height=df_loc_loc['Loss'],yerr=df_loc_loc['Error'],width=0.2,lw=0,ecolor='black',label='Ridge')
df_loc_loc = df_loc[df_loc['Architecture']=='forest']
ax.bar(x,height=df_loc_loc['Loss'],yerr=df_loc_loc['Error'],width=0.2,lw=0,ecolor='black',label='Forest')
df_loc_loc = df_loc[df_loc['Architecture']=='svm']
ax.bar(x+0.2,height=df_loc_loc['Loss'],yerr=df_loc_loc['Error'],width=0.2,lw=0,ecolor='black',label='SVM')
ax.set_xticks(x)
ax.set_xticklabels(df_loc_loc.Embedding.values)
ax.tick_params(axis='both', which='major', labelsize=6)
ax.legend(fontsize=6,title='Emb. to Yield\nArchitecture',title_fontsize=6)

ax = axs[1,0]
df_loc = control_df[control_df['Dataset']=='Test']
x=np.array(list(range(len(df_loc))))
ax.bar(x=x,height=df_loc['Loss'],yerr=df_loc['Error'],width=0.6,lw=0,ecolor='black',color='black',alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(df_loc.Model.values)
ax.tick_params(axis='both', which='major', labelsize=6)
ax.set_ylabel('Test Loss (MSE)',fontsize=6)
ax.set_ylim([0.3,0.85])


ax = axs[1,1]
df_loc = model_df[model_df['Dataset']=='Test']
df_loc_loc = df_loc[df_loc['Architecture']=='ridge']
x=np.array(list(range(len(df_loc_loc))))
ax.bar(x-0.2,height=df_loc_loc['Loss'],yerr=df_loc_loc['Error'],width=0.2,lw=0,ecolor='black',label='Ridge')
df_loc_loc = df_loc[df_loc['Architecture']=='forest']
ax.bar(x,height=df_loc_loc['Loss'],yerr=df_loc_loc['Error'],width=0.2,lw=0,ecolor='black',label='Forest')
df_loc_loc = df_loc[df_loc['Architecture']=='svm']
ax.bar(x+0.2,height=df_loc_loc['Loss'],yerr=df_loc_loc['Error'],width=0.2,lw=0,ecolor='black',label='SVM')
ax.set_xticks(x)
ax.set_xticklabels(df_loc_loc.Embedding.values)
ax.tick_params(axis='both', which='major', labelsize=6)
ax.set_xlabel('Protein Embedding',fontsize=6)
# ax.legend(fontsize=6)


fig.tight_layout()
fig.savefig('./embed_to_yield_figure.png')
plt.close()

df = control_df
temp_list=[]
for c in c_model_emb_list:
	cv_min_loss=np.inf
	c_df = model_df[model_df['Embedding']==c]
	for a in a_arch_list:
		a_df = c_df[c_df['Architecture']==a]
		cv = a_df[a_df['Dataset']=='CV']
		if cv['Loss'].values[0]<cv_min_loss:
			df_min = a_df.copy()
			cv_min_loss = cv['Loss'].values[0]
	df_min['Model']=df_min['Embedding']
	df_min = df_min.drop(['Embedding','Architecture'],axis=1)
	df=df.append(df_min)
	
print(df)

fig, ax = plt.subplots(1,1,figsize=[5,2],dpi=600)
cv_df = df[df['Dataset']=='CV']
x=np.array(list(range(len(cv_df))))
ax.bar(x=x-0.2,height=cv_df['Loss'],yerr=cv_df['Error'],width=0.4,color='lightgray',lw=0,ecolor='black',label='CV')
test_df = df[df['Dataset']=='Test']
ax.bar(x=x+0.2,height=test_df['Loss'],yerr=test_df['Error'],width=0.4,color='darkgray',lw=0,ecolor='black',label='Test')
# sns.barplot(data=df,x='Model',y='Loss',hue='Dataset',ax=ax)
ax.legend(fontsize=6)
ax.set_xticks(x)
ax.set_xticklabels(test_df.Model.values)
ax.tick_params(axis='both', which='major', labelsize=6)
ax.set_xlabel('Model',fontsize=6)
ax.set_ylabel('Loss (MSE)',fontsize=6)


fig.tight_layout()
fig.savefig('./embed_to_yield_newfigure.png')
plt.close()


# # c_models=['emb_fnn_flat','emb_fnn_maxpool','emb_fnn_maxpool_linear','emb_rnn','small_emb_rnn','small_emb_atn_rnn','small_emb_rnn_linear',
# #         'emb_cnn','small_emb_cnn','small_emb_atn_cnn','small_emb_cnn_linear']
# # c_models=['emb_fnn_flat','emb_fnn_flat_dense_linear','emb_rnn','emb_rnn_dense_linear','emb_cnn','emb_cnn_dense_linear']
# c_models=['emb_rnn','emb_fnn_flat','emb_cnn']

# c_models.reverse()
# # c_names=['Flatten AA Prop','Max AA Prop','Linear Top, Max AA Prop','Recurrent','Small Recurrent','Small Recurrent + Atn','Linear Top, Small Recurrent',
# # 		'Convolutional','Small Convolutional','Small Convolutional + Atn','Linear Top, Small Convolutional']
# c_names=['Feedforward\n Embedding','Linear Top\nFeedforward Embedding','Recurrent\n Embedding','Linear Top\nRecurrent Embedding','Convolutional\n Embedding','Linear Top\nConvolutional Embedding']
# c_names=['Recurrent\n Embedding','Feedforward\n Embedding','Convolutional\n Embedding']

# c_names.reverse()
# a_models=['ridge','svm','forest']
# c_mdl_test_loss,c_mdl_test_std=[],[]
# for arch in c_models:
# 	c_prop=[[1,8,10],arch,1]
# 	arch_lost_list=[]
# 	for i in range(10):
# 		best_loss_per_top=np.inf
# 		for top_arch in a_models:
# 			mdl=modelbank.sequence_embeding_to_yield_model(c_prop+[i],top_arch,1)
# 			if mdl.model_stats['test_avg_loss']<best_loss_per_top:
# 				best_loss_per_top=mdl.model_stats['test_avg_loss']
# 		arch_lost_list.append(best_loss_per_top)
# 	c_mdl_test_loss.append(np.average(arch_lost_list))
# 	c_mdl_test_std.append(np.std(arch_lost_list))


# oh_test_loss=[]
# oh_model=modelbank.seq_to_yield_model('forest',1)
# oh_test_loss.append(oh_model.model_stats['test_avg_loss'])
# for i in range(9):
# 	oh_model.change_sample_seed(i)
# 	oh_test_loss.append(oh_model.model_stats['test_avg_loss'])
# oh_test_std=np.std(oh_test_loss)
# oh_test_loss=np.mean(oh_test_loss)

# assay_test_loss=[]
# assay_model=modelbank.assay_to_yield_model([1,8,10],'forest',1)
# assay_test_loss.append(assay_model.model_stats['test_avg_loss'])
# for i in range(9):
# 	assay_model.change_sample_seed(i)
# 	assay_test_loss.append(assay_model.model_stats['test_avg_loss'])
# assay_test_std=np.std(assay_test_loss)
# assay_test_loss=np.mean(assay_test_loss)

# control_model=modelbank.control_to_yield_model('ridge',1)
# control_loss=control_model.model_stats['test_avg_loss']
# control_model.limit_test_set([1,8,10])
# exploded_df,_,_=load_format_data.explode_yield(control_model.testing_df)
# exp_var=np.average(np.square(np.array(exploded_df['y_std'])))


# fig,ax=plt.subplots(1,1,figsize=[2.5,2.5],dpi=300)
# # c_models.append('One-Hot')
# # c_names.append('One-Hot')
# # c_mdl_test_loss.append(oh_test_loss)
# # c_mdl_test_std.append(oh_test_std)
# x=[-1,len(c_models)]

# ax.axvline(control_loss,x[0],x[1],color='green',linestyle='--',label='Average Yield')

# # ax.axvline(assay_test_loss,x[0],x[1],color='blue',linestyle='--',label='Experimental Assay Scores')
# # assay_plus=[assay_test_loss+assay_test_std]*2
# # assay_min=[assay_test_loss-assay_test_std]*2
# # ax.fill_betweenx(x,assay_plus,assay_min,alpha=0.2,color='blue')

# ax.axvline(oh_test_loss,x[0],x[1],color='orange',linestyle='--',label='One-Hot Sequence')
# oh_plus=[oh_test_loss+oh_test_std]*2
# oh_min=[oh_test_loss-oh_test_std]*2
# # ax.fill_betweenx(x,oh_plus,oh_min,alpha=0.2,color='green')

# ax.axvline(exp_var,x[0],x[1],color='purple',linestyle='--',label='Experimental Variance')


# ax.barh(range(len(c_models)),c_mdl_test_loss,xerr=c_mdl_test_std,height=0.8,color='black')
# ax.set_yticks(range(len(c_models)))
# ax.set_yticklabels(c_names)
# # ax.legend(fontsize=6,framealpha=1)
# ax.tick_params(axis='both', which='major', labelsize=6)
# ax.set_xlabel('Test Loss (MSE)',fontsize=6)
# ax.set_xlim([0.35,0.75])
# ax.set_ylim(x)
# ax.set_title('Yield Predictions',fontsize=6)
# fig.tight_layout()
# fig.savefig('./embed_to_yield_strategies.png')
# plt.close()


