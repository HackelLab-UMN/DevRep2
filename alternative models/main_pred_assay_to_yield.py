import warnings
warnings.filterwarnings('ignore')

import submodels_module as modelbank
import pandas as pd 
import numpy as np 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import load_format_data


a2y_model = modelbank.assay_to_yield_model([1,8,10],'forest',1)

s2a_list = ['ridge','fnn','emb_fnn_flat','emb_rnn','emb_cnn']
s2a_name_list =['One Hot\nLinear', 'One Hot\nNon-Linear','Flatten','Recurrent','Convolutional']

df_list=[]
for arch,name in zip(s2a_list,s2a_name_list):
	s2a_prop = [[1,8,10],arch,1]
	model = modelbank.seq_to_assay_model(*s2a_prop)
	model.save_predictions()

	df=pd.DataFrame({'Model':name,'Dataset':'CV','Loss':model.model_stats['cv_avg_loss'],'Error':model.model_stats['cv_std_loss']},index=[0])
	df_list.append(df)
	# print(f'{name} has cv avg loss: ')
	# print(hi)

	test_loss_list=[]
	for i in range(10):
		a2y_model.apply_predicted_assay_scores([s2a_prop[1],1,i]) #careful! it resets the test loss
		test_loss_list.append(a2y_model.model_stats['test_avg_loss'])
		# try: 
		# 	# print(f'{name} has cv loss of {a2y_model.model_stats['cv_avg_loss']}')
		# 	hi=a2y_model.model_stats['cv_avg_loss'] ; 0.49870203814305025
		# 	print(hi)
		# except:
		# 	print(f'{name} failed')

	df = pd.DataFrame({'Model':name,'Dataset':'Test','Loss':np.average(test_loss_list),'Error':np.std(test_loss_list)},index=[0])
	df_list.append(df)
pred_df = pd.concat(df_list)

pred_df.to_pickle('main_pred_assay_to_yield_cv_test.pkl')


print('\n')
print(f'pred df is: {pred_df}')
print('\n')

control_df_list=[]

model = modelbank.control_to_yield_model('ridge',1)
test_loss,test_std = model.model_stats['test_avg_loss'],model.model_stats['test_std_loss']
df = pd.DataFrame({'Model':'Strain Only\nControl','Test Loss':test_loss,'Error':test_std},index=[0])
control_df_list.append(df)

model.limit_test_set([1,8,10])
exploded_df,_,_=load_format_data.explode_yield(model.testing_df)
test_exp_var=np.average(np.square(np.array(exploded_df['y_std'])))
df = pd.DataFrame({'Model':'Experimental\nVariance','Test Loss':test_exp_var,'Error':0},index=[0])
control_df_list.append(df)


# from tensorflow.python.layers import base
# import tensorflow as tf
# import tensorflow.contrib.slim as slim
c_prop=[[1,8,10],'emb_cnn',1]
test_loss_list=[]
for i in range(10):
	model = modelbank.sequence_embeding_to_yield_model(c_prop+[i],'svm',1)
	test_loss_list.append(model.model_stats['test_avg_loss'])

# ##Trying to get model summary results for devrep transfer learning:
# print('\n')
# print(f'Trying to print out model summary:')
# model.summary()
# print('\n')

df = pd.DataFrame({'Model':'Transfer\nLearning','Test Loss':np.average(test_loss_list),'Error':np.std(test_loss_list)},index=[0])
control_df_list.append(df)

a2y_model = modelbank.assay_to_yield_model([1,8,10],'forest',1)
a2y_model.limit_test_set([1,8,10])
a2y_model.test_model()
df = pd.DataFrame({'Model':'True Assay\nScores','Test Loss':a2y_model.model_stats['test_avg_loss'],'Error':a2y_model.model_stats['test_std_loss']},index=[0])
control_df_list.append(df)
control_df = pd.concat(control_df_list)

fig, axs = plt.subplots(1,2,figsize=[6,2],dpi=600,sharey=True,gridspec_kw={'width_ratios':[4, 5]})

ax = axs[0]
x=list(range(len(control_df)))
ax.bar(x=x,height=control_df['Test Loss'],yerr=control_df['Error'],width=0.6,lw=0,ecolor='black',color='black',alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(control_df.Model.values)
ax.tick_params(axis='both', which='major', labelsize=6)
ax.set_ylabel('Test Loss (MSE)',fontsize=6)
ax.set_ylim([0.00,0.8])

pred_df=pd.read_pickle('main_pred_assay_to_yield_cv_test.pkl')
ax = axs[1]
loc_df=pred_df[pred_df['Dataset']=='CV']
x=np.array(range(len(loc_df)))
ax.bar(x=x-.15,height=loc_df['Loss'],yerr=loc_df['Error'],width=0.3,lw=0,ecolor='black',color='black',alpha=0.2)#,label='CV'
loc_df=pred_df[pred_df['Dataset']=='Test']
ax.bar(x=x+.15,height=loc_df['Loss'],yerr=loc_df['Error'],width=0.3,lw=0,ecolor='black',color='black',alpha=0.8)#,label='Test'
ax.set_xticks(x)
ax.set_xticklabels(loc_df.Model.values)
ax.tick_params(axis='both', which='major', labelsize=6)
ax.set_xlabel('Model Used to Predict Assay Scores',fontsize=6)
ax.set_ylim([0.00,0.8])
ax.set_yticks([0.00,.1,.2,.3,.4,.5,.6,.7,.8])
# ax.legend(fontsize=6)

fig.tight_layout()
fig.savefig('./predicted_assay_scores_to_yield_fig.png')
plt.close()











