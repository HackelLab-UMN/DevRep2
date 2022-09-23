import submodels_module as modelbank
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

c_prop=[[1,8,10],'emb_cnn',1]
model = modelbank.sequence_embeding_to_yield_model(c_prop+[0],'svm',1)
strain_list=[]
for strain in model.plotpairs_cv[2]:
	if strain[0]==1:
		strain_list.append('$I^q$')
	elif strain[1]==1:
		strain_list.append('$SH$')
	else:
		print('ahhhh')

df = pd.DataFrame({'Predicted':model.plotpairs_cv[1], 'True':model.plotpairs_cv[0],'Strain':strain_list})
print(len(model.training_df))
# df= df.sample(n=1000)

fig,ax=plt.subplots(1,1,figsize=[2,2],dpi=600)
df_a= df[df['Strain']=='$I^q$']
print(spearmanr(df_a['Predicted'].values,df_a['True'].values))
ax.scatter(x=df_a['Predicted'],y=df_a['True'],lw=0,alpha=0.2,color='purple',label='$I^q$')

df_a= df[df['Strain']=='$SH$']
print(spearmanr(df_a['Predicted'].values,df_a['True'].values))
ax.scatter(x=df_a['Predicted'],y=df_a['True'],lw=0,alpha=0.2,color='maroon',label='$SH$')


ax.set_xlabel('Predicted Yield',fontsize=6)
ax.set_ylabel('True Yield',fontsize=6)
ax.tick_params(axis='both', which='major', labelsize=6)
ax.set_xticks([-2,-1,0,1,2])
ax.set_yticks([-2,-1,0,1,2])
ax.set_xlim([-2,2.5])
ax.set_ylim([-2,2.5])
ax.set_aspect('equal')
ax.legend().remove()
# ax.legend(fontsize=6,title='Strain',title_fontsize=6,framealpha=1)


fig.tight_layout()
fig.savefig('./emb_to_yield_scatter_cv.png')
plt.close()



strain_list=[]
for strain in model.plotpairs_test[2]:
	if strain[0]==1:
		strain_list.append('$I^q$')
	elif strain[1]==1:
		strain_list.append('$SH$')
	else:
		print('ahhhh')

df = pd.DataFrame({'Predicted':model.plotpairs_test[1], 'True':model.plotpairs_test[0],'Strain':strain_list})
# df= df.sample(n=1000)
model.limit_test_set([1,8,10])
print(len(model.testing_df))

fig,ax=plt.subplots(1,1,figsize=[2,2],dpi=600)
df_a= df[df['Strain']=='$I^q$']
print(spearmanr(df_a['Predicted'].values,df_a['True'].values))
ax.scatter(x=df_a['Predicted'],y=df_a['True'],lw=0,alpha=0.2,color='purple',label='$I^q$')

df_a= df[df['Strain']=='$SH$']
print(spearmanr(df_a['Predicted'].values,df_a['True'].values))
ax.scatter(x=df_a['Predicted'],y=df_a['True'],lw=0,alpha=0.2,color='maroon',label='$SH$')


ax.set_xlabel('Predicted Yield',fontsize=6)
ax.set_ylabel('True Yield',fontsize=6)
ax.tick_params(axis='both', which='major', labelsize=6)
ax.set_xticks([-2,-1,0,1,2])
ax.set_yticks([-2,-1,0,1,2])
ax.set_xlim([-2,2.5])
ax.set_ylim([-2,2.5])
ax.set_aspect('equal')
ax.legend().remove()
# ax.legend(fontsize=6,framealpha=1)


fig.tight_layout()
fig.savefig('./emb_to_yield_scatter_test.png')
plt.close()