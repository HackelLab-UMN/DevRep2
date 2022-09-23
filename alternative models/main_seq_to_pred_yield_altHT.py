import sys
import submodels_module as modelbank
from itertools import combinations
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import load_format_data 
import pandas as pd
import numpy as np
import seaborn as sns

def main():

    toggle_no=int(sys.argv[1])

    combos=[[1],[8],[10],[1,8],[1,10],[8,10],[1,8,10]]
    combo=combos[toggle_no]

    b_models=['ridge','svm','forest','fnn']
    best_test_lost=np.inf
    best_model,best_arch=[],[]
    for arch in b_models:
        b=modelbank.assay_to_yield_model(combo,arch,1)
        # b.limit_test_set([1,8,10])
        # b.test_model()
        if b.model_stats['test_avg_loss'] < best_test_lost:
            best_test_lost=b.model_stats['test_avg_loss']
            best_model=b
            best_arch=arch
        del(b)
    # best_model.save_predictions()
    del(best_model)
    print(combo,best_arch)



    ### use predictions of yield from model a to make a seq-to-(predicted)yield model
    c_models=['ridge','emb_fnn_flat','emb_rnn','emb_cnn']
    for arch in c_models:
        c=modelbank.seq_to_pred_yield_model([combo,best_arch,1,0],[arch,1])
        c.cross_validate_model()
        c.limit_test_set([1,8,10])
        c.test_model()
        # c.plot()




# if __name__ == '__main__':
#     main()

combos=[[1],[8],[10],[1,8],[1,10],[8,10],[1,8,10]]
combo_data=[]
for combo in combos:
    b_models=['ridge','svm','forest','fnn']
    best_test_loss=np.inf
    best_model,best_arch=[],[]
    for arch in b_models:
        b=modelbank.assay_to_yield_model(combo,arch,1)
        # b.limit_test_set([1,8,10])
        # b.test_model()
        if b.model_stats['test_avg_loss'] < best_test_loss:
            best_test_loss=b.model_stats['test_avg_loss']
            best_model=b
            best_arch=arch
        del(b)
    del(best_model)
    assay_test_loss=best_test_loss

    c_models=['ridge','emb_fnn_flat','emb_rnn','emb_cnn']
    best_pred_loss=np.inf
    for arch in c_models:
        c=modelbank.seq_to_pred_yield_model([combo,best_arch,1,0],[arch,1])
        if c.model_stats['test_avg_loss'] < best_pred_loss:
            best_pred_loss = c.model_stats['test_avg_loss']
            seq_to_pred_training_loss=c.model_stats['cv_avg_loss']
            seq_to_pred_testing_loss=c.model_stats['test_avg_loss']
            best_pred_model=c
        del(c)
    best_pred_model.plot()
    del(best_pred_model)

    combo_data.append([assay_test_loss,seq_to_pred_training_loss,seq_to_pred_testing_loss])
combo_data=np.array(combo_data)


assay_names=['$P_{PK37}$','$G_{SH}$',r'$\beta_{SH}$']

df_a=pd.DataFrame(combo_data[:,0])
df_a.columns=['Test Loss (MSE)']
df_a['Model Type']=['Assay Score to True Yield']*len(df_a)
df_a['Assay(s)']=[assay_names[0],assay_names[1],assay_names[2],assay_names[0]+', '+assay_names[1],assay_names[0]+', '+assay_names[2],assay_names[1]+', '+assay_names[2],assay_names[0]+', '+assay_names[1]+', '+assay_names[2]]


df_b=pd.DataFrame(combo_data[:,2])
df_b.columns=['Test Loss (MSE)']
df_b['Model Type']=['Seq. to Predicted Yield']*len(df_b)
df_b['Assay(s)']=[assay_names[0],assay_names[1],assay_names[2],assay_names[0]+', '+assay_names[1],assay_names[0]+', '+assay_names[2],assay_names[1]+', '+assay_names[2],assay_names[0]+', '+assay_names[1]+', '+assay_names[2]]

df=pd.concat([df_a,df_b])
df.to_pickle('./seq_to_yield_combo_data.pkl')

# fig,ax=plt.subplots(1,1,figsize=[2.5,2.5],dpi=300)
# sns.barplot(ax=ax,data=df,x='Test Loss (MSE)',y='Assay(s)',ci=None,hue='Model Type',palette='bright')

# oh_model=modelbank.seq_to_yield_model('forest',1)
# oh_test_loss=oh_model.model_stats['test_avg_loss']

# control_model=modelbank.control_to_yield_model('ridge',1)
# control_loss=control_model.model_stats['test_avg_loss']
# control_model.limit_test_set([1,8,10])
# exploded_df,_,_=load_format_data.explode_yield(control_model.testing_df)
# exp_var=np.average(np.square(np.array(exploded_df['y_std'])))

# # c_names=['Linear\n Model','Feedforward\n Embedding','Recurrent\n Embedding','Convolutional\n Embedding']

# x=[-1,len(df_a)]
# ax.axvline(control_loss,x[0],x[1],color='red',linestyle='--',label='Strain Only')
# ax.axvline(exp_var,x[0],x[1],color='purple',linestyle='--',label='Experimental\nVariance')
# ax.axvline(oh_test_loss,x[0],x[1],color='green',linestyle='--',label='Seq. Model\n(True Yields)')
# lg=ax.legend(fontsize=6,framealpha=1)
# # lg.set_visible(False)
# ax.tick_params(axis='both', which='major', labelsize=6)
# ax.set_xlabel('Test Loss (MSE)',fontsize=6)
# ax.set_ylabel('Assay(s)',fontsize=6)
# ax.set_xlim([0.35,0.75])
# fig.tight_layout()
# fig.savefig('./seq_to_pred_yield_combos.png')
# plt.close()

df_c=pd.DataFrame(combo_data[:,1])
df_c.columns=['Training Loss (MSE)']
# df_c['Model Type']=['Seq. to Predicted Yield']*len(df_b)
df_c['Assay(s)']=[assay_names[0],assay_names[1],assay_names[2],assay_names[0]+', '+assay_names[1],assay_names[0]+', '+assay_names[2],assay_names[1]+', '+assay_names[2],assay_names[0]+', '+assay_names[1]+', '+assay_names[2]]

fig,ax=plt.subplots(1,1,figsize=[2.5,2.5],dpi=300)
sns.barplot(ax=ax,data=df_c,x='Training Loss (MSE)',y='Assay(s)',ci=None,color='black',saturation=1)
ax.tick_params(axis='both', which='major', labelsize=6)
ax.set_xlabel('Training Loss (MSE)',fontsize=6)
ax.set_ylabel('Assay(s) Used to Predict Yield',fontsize=6)
# ax.set_xlim([0.35,0.75])
fig.tight_layout()
fig.savefig('./seq_to_pred_yield_combos_cv.png')
plt.close()