import submodels_module as modelbank
import sys
import load_format_data
import pandas as pd
import numpy as np
from functools import partial
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def get_alt_emb(emb_name, df):
    x_a_in=df.loc[:,emb_name]
    x_a=[x.tolist() for x in x_a_in]
    return x_a


class alt_emb_to_yield_model(modelbank.x_to_yield_model):
    def __init__(self, emb_name, model_architecture, sample_fraction):
        super().__init__(emb_name, model_architecture, sample_fraction)
        self.training_df=load_format_data.load_df('assay_to_dot_training_data_alt_emb')
        self.testing_df=load_format_data.load_df('seq_to_dot_test_data_alt_emb') 
        self.get_input_seq=partial(get_alt_emb,emb_name)


def train():
    toggle_no=int(sys.argv[1])
    alt_emb_list=['AAindex3','AAindex10','full_bert','para_bert','full_unirep','para_unirep','full_eunirep','para_eunirep']
    alt_emb=alt_emb_list[toggle_no]

    arch_list=['ridge','svm','forest']
    for arch in arch_list:
        mdl=alt_emb_to_yield_model(alt_emb,arch,1)
        mdl.cross_validate_model()
        mdl.limit_test_set([1,8,10])
        mdl.test_model()

# train()

def plot():
    c_models=['emb_fnn_flat','emb_rnn','emb_cnn']
    d_models=['ridge','svm','forest']


    best_devrep_loss=np.inf 
    for c_arch in c_models:
        loss_per_carch=[]
        for i in range(10):
            c_mdl=[[1,8,10],c_arch,1,i]
            best_loss_per_darch=np.inf
            for d_arch in d_models:
                mdl=modelbank.sequence_embeding_to_yield_model(c_mdl,d_arch,1)
                if mdl.model_stats['test_avg_loss'] < best_loss_per_darch:
                    best_loss_per_darch = mdl.model_stats['test_avg_loss']
            loss_per_carch.append(best_loss_per_darch)
        if np.average(loss_per_carch) < best_devrep_loss:
            best_devrep_loss = np.average(loss_per_carch)

    alt_emb_list=['AAindex3','AAindex10','full_bert','para_bert','full_unirep','para_unirep','full_eunirep','para_eunirep']
    alt_emb_loss_list=[]
    for alt_emb in alt_emb_list:
        best_per_altemb=np.inf
        for arch in d_models:
            mdl=alt_emb_to_yield_model(alt_emb,arch,1)
            if mdl.model_stats['test_avg_loss'] < best_per_altemb:
                best_per_altemb = mdl.model_stats['test_avg_loss']
        alt_emb_loss_list.append(best_per_altemb)


    x_values=list(range(len(alt_emb_list)+1))

    fig,ax=plt.subplots(1,1,figsize=[8,2.5],dpi=1200)
    ax.bar(x_values,[best_devrep_loss]+alt_emb_loss_list,width=0.8,color='black')
    ax.set_xticks(x_values)
    ax.set_xticklabels(['DevRep','AA Index\n3','AA Index\n10','TAPE\nFull','TAPE\nParatope','UniRep\nFull','UniRep\nParatope','evoUniRep\nFull','evoUniRep\nParatope'])

    oh_model=modelbank.seq_to_yield_model('forest',1)
    oh_test_loss=oh_model.model_stats['test_avg_loss']

    assay_model=modelbank.assay_to_yield_model([1,8,10],'forest',1)
    assay_test_loss=assay_model.model_stats['test_avg_loss']

    control_model=modelbank.control_to_yield_model('ridge',1)
    control_loss=control_model.model_stats['test_avg_loss']
    control_model.limit_test_set([1,8,10])
    exploded_df,_,_=load_format_data.explode_yield(control_model.testing_df)
    exp_var=np.average(np.square(np.array(exploded_df['y_std'])))


    x=[-1,len(alt_emb_list)+2]
    ax.axhline(control_loss,x[0],x[1],color='green',linestyle='--',label='Strain Only')
    ax.axhline(exp_var,x[0],x[1],color='purple',linestyle='--',label='Experimental\nVariance')
    ax.axhline(oh_test_loss,x[0],x[1],color='orange',linestyle='--',label='OH Model')
    # ax.axhline(assay_test_loss,x[0],x[1],color='blue',linestyle='--',label='Experimental Assay Scores')
    # lg=ax.legend(fontsize=6,framealpha=1)
    # lg.set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.set_ylabel('Test Loss (MSE)',fontsize=6)
    ax.set_xlabel('Sequence Embedding',fontsize=6)
    ax.set_ylim([0.35,0.75])
    fig.tight_layout()
    fig.savefig('./alt_embed_to_yield.png')
    plt.close()


    # fig,ax=plt.subplots(1,1,figsize=[2,2],dpi=300)
    # cluster_df=pd.read_pickle('./datasets/emb_cluster_var.pkl')

    # cluster_df = cluster_df[cluster_df['Projection']=='UMAP']

    # loss_list=[oh_test_loss,assay_test_loss,best_devrep_loss,alt_emb_loss_list[1],alt_emb_loss_list[5]]
    # cluster_df['Test Loss (MSE)']=loss_list
    # sns.scatterplot(data=cluster_df,x='Average Stdev of\nDev. Per Cluster',y='Test Loss (MSE)',ax=ax,hue='Embedding')
    # ax.tick_params(axis='both', which='major', labelsize=6)
    # ax.set_ylabel('Test Loss (MSE)',fontsize=6)
    # ax.set_xlabel('Intra Cluster Yield Variance',fontsize=6)
    # # ax.legend(fontsize=6,framealpha=1)
    # ax.legend().set_visible(False)
    # # ax.set_ylim([0.35,0.75])
    # fig.tight_layout()
    # fig.savefig('./emb_cluster_vs_loss.png')

    # return cluster_df

if __name__ == '__main__':
    cluster_df = plot()