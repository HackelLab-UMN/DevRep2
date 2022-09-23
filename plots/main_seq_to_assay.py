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


    # c_models=['ridge','fnn','emb_fnn_flat','emb_fnn_maxpool','emb_fnn_maxpool_linear','emb_rnn','small_emb_rnn','small_emb_atn_rnn','small_emb_rnn_linear',
    #     'emb_cnn','small_emb_cnn','small_emb_atn_cnn','small_emb_cnn_linear']

    c_models=['emb_fnn_flat_dense_linear','emb_cnn_dense_linear','emb_rnn_dense_linear']
    c=modelbank.seq_to_assay_model([1,8,10],c_models[toggle_no],1)
    c.cross_validate_model()
    c.test_model()
    # c.save_predictions()
    c.save_sequence_embeddings()





# if __name__ == '__main__':
#     main()


df_list=[]

model = modelbank.control_to_assay_model([1,8,10],'ridge',1)
cv_loss,cv_std = model.model_stats['cv_avg_loss'],model.model_stats['cv_std_loss']
df = pd.DataFrame({'Model':'Assay Only\nControl', 'Dataset':'CV','Loss':cv_loss,'Error':cv_std},index=[0])
df_list.append(df)
test_loss,test_std = model.model_stats['test_avg_loss'],model.model_stats['test_std_loss']
df = pd.DataFrame({'Model':'Assay Only\nControl', 'Dataset':'Test','Loss':test_loss,'Error':test_std},index=[0])
df_list.append(df)

exploded_df,_,_=load_format_data.explode_assays([1,8,10],model.training_df)
cv_exp_var=np.average(np.square(np.array(exploded_df['y_std'])))
df = pd.DataFrame({'Model':'Experimental\nVariance', 'Dataset':'CV','Loss':cv_exp_var,'Error':0},index=[0])
df_list.append(df)
exploded_df,_,_=load_format_data.explode_assays([1,8,10],model.testing_df)
test_exp_var=np.average(np.square(np.array(exploded_df['y_std'])))
df = pd.DataFrame({'Model':'Experimental\nVariance', 'Dataset':'Test','Loss':test_exp_var,'Error':0},index=[0])
df_list.append(df)

model = modelbank.seq_to_assay_model([1,8,10],'ridge',1)
cv_loss,cv_std = model.model_stats['cv_avg_loss'],model.model_stats['cv_std_loss']
df = pd.DataFrame({'Model':'One Hot\nLinear', 'Dataset':'CV','Loss':cv_loss,'Error':cv_std},index=[0])
df_list.append(df)
test_loss,test_std = model.model_stats['test_avg_loss'],model.model_stats['test_std_loss']
df = pd.DataFrame({'Model':'One Hot\nLinear', 'Dataset':'Test','Loss':test_loss,'Error':test_std},index=[0])
df_list.append(df)

model = modelbank.seq_to_assay_model([1,8,10],'fnn',1)
cv_loss,cv_std = model.model_stats['cv_avg_loss'],model.model_stats['cv_std_loss']
df = pd.DataFrame({'Model':'One Hot\nNon-Linear', 'Dataset':'CV','Loss':cv_loss,'Error':cv_std},index=[0])
df_list.append(df)
test_loss,test_std = model.model_stats['test_avg_loss'],model.model_stats['test_std_loss']
df = pd.DataFrame({'Model':'One Hot\nNon-Linear', 'Dataset':'Test','Loss':test_loss,'Error':test_std},index=[0])
df_list.append(df)

model = modelbank.seq_to_assay_model([1,8,10],'emb_fnn_flat',1)
cv_loss,cv_std = model.model_stats['cv_avg_loss'],model.model_stats['cv_std_loss']
df = pd.DataFrame({'Model':'Flatten', 'Dataset':'CV','Loss':cv_loss,'Error':cv_std},index=[0])
df_list.append(df)
test_loss,test_std = model.model_stats['test_avg_loss'],model.model_stats['test_std_loss']
df = pd.DataFrame({'Model':'Flatten', 'Dataset':'Test','Loss':test_loss,'Error':test_std},index=[0])
df_list.append(df)

model = modelbank.seq_to_assay_model([1,8,10],'emb_rnn',1)
cv_loss,cv_std = model.model_stats['cv_avg_loss'],model.model_stats['cv_std_loss']
df = pd.DataFrame({'Model':'Recurrent', 'Dataset':'CV','Loss':cv_loss,'Error':cv_std},index=[0])
df_list.append(df)
test_loss,test_std = model.model_stats['test_avg_loss'],model.model_stats['test_std_loss']
df = pd.DataFrame({'Model':'Recurrent', 'Dataset':'Test','Loss':test_loss,'Error':test_std},index=[0])
df_list.append(df)

model = modelbank.seq_to_assay_model([1,8,10],'emb_cnn',1)
cv_loss,cv_std = model.model_stats['cv_avg_loss'],model.model_stats['cv_std_loss']
df = pd.DataFrame({'Model':'Convolutional', 'Dataset':'CV','Loss':cv_loss,'Error':cv_std},index=[0])
df_list.append(df)
test_loss,test_std = model.model_stats['test_avg_loss'],model.model_stats['test_std_loss']
df = pd.DataFrame({'Model':'Convolutional', 'Dataset':'Test','Loss':test_loss,'Error':test_std},index=[0])
df_list.append(df)

df = pd.concat(df_list,ignore_index=True)
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
fig.savefig('./seq_to_assay_arch.png')
plt.close()



# # # c_models=['ridge','fnn','emb_fnn_flat','emb_fnn_maxpool','emb_fnn_maxpool_linear','emb_rnn','small_emb_rnn','small_emb_atn_rnn','small_emb_rnn_linear',
# # #         'emb_cnn','small_emb_cnn','small_emb_atn_cnn','small_emb_cnn_linear']
# c_models=['emb_fnn_flat','emb_rnn', 'emb_cnn']
# c_models.reverse()
# # # c_names=['Linear Model','One-Hot','Flatten AA Prop','Max AA Prop','Linear Top, Max AA Prop','Recurrent','Small Recurrent','Small Recurrent + Atn','Linear Top, Small Recurrent',
# # # 		'Convolutional','Small Convolutional','Small Convolutional + Atn','Linear Top, Small Convolutional']
# c_names=['Feedforward\n Embedding','Recurrent\n Embedding','Convolutional\n Embedding']
# c_names.reverse()
# c_mdl_test_loss,c_mdl_test_std=[],[]
# for arch in c_models:
# 	c_prop=[[1,8,10],arch,1]
# 	mdl=modelbank.seq_to_assay_model(*c_prop)
# 	c_mdl_test_loss.append(mdl.model_stats['test_avg_loss'])
# 	c_mdl_test_std.append(mdl.model_stats['test_std_loss'])



# control_model=modelbank.control_to_assay_model([1,8,10],'ridge',1)
# control_loss=control_model.model_stats['test_avg_loss']
# exploded_df,_,_=load_format_data.explode_assays([1,8,10],control_model.testing_df)
# exp_var=np.average(np.square(np.array(exploded_df['y_std'])))

# non_emb_model=modelbank.seq_to_assay_model([1,8,10],'fnn',1) #FNN or ridge?
# non_emb_loss=non_emb_model.model_stats['test_avg_loss']

# fig,ax=plt.subplots(1,1,figsize=[2.5,2.5],dpi=300)
# x=[-1,len(c_models)]

# ax.axvline(control_loss,x[0],x[1],color='red',linestyle='--',label='Use Average Assay Score')
# ax.axvline(exp_var,x[0],x[1],color='purple',linestyle='--',label='Experimental Accuracy')
# ax.axvline(non_emb_loss,x[0],x[1],color='green',linestyle='--',label='Non-Embedding')

# # oh_test_loss=c_mdl_test_loss[-1]
# # oh_test_std=c_mdl_test_std[-1]
# # ax.axvline(oh_test_loss,x[0],x[1],color='green',linestyle='--',label='One-Hot Linear')
# # oh_plus=[oh_test_loss+oh_test_std]*2
# # oh_min=[oh_test_loss-oh_test_std]*2
# # ax.fill_betweenx(x,oh_plus,oh_min,alpha=0.2,color='green')

# # oh_test_loss=c_mdl_test_loss[-2]
# # oh_test_std=c_mdl_test_std[-2]
# # ax.axvline(oh_test_loss,x[0],x[1],color='orange',linestyle='--',label='One-Hot FNN')
# # oh_plus=[oh_test_loss+oh_test_std]*2
# # oh_min=[oh_test_loss-oh_test_std]*2
# # ax.fill_betweenx(x,oh_plus,oh_min,alpha=0.2,color='orange')

# # c_models=c_models[:-2]
# # c_mdl_test_loss=c_mdl_test_loss[:-2]
# # c_mdl_test_std=c_mdl_test_std[:-2]
# # c_names=c_names[:-2]
# ax.barh(range(len(c_models)),c_mdl_test_loss,yerr=c_mdl_test_std,height=0.8,color='black')
# ax.set_yticks(range(len(c_models)))
# ax.set_yticklabels(c_names)
# # ax.legend(fontsize=6,framealpha=1)
# ax.tick_params(axis='both', which='major', labelsize=6)
# ax.set_xlabel('Test Loss (MSE)',fontsize=6)
# # ax.set_xlim([0.35,0.75])
# ax.set_ylim(x)
# ax.set_title('Assay Score Predictions',fontsize=6)
# fig.tight_layout()
# fig.savefig('./seq_to_assay_arch.png')
# plt.close()