import submodels_module as modelbank
import pandas as pd 
import numpy as np 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import load_format_data
from scipy.stats import spearmanr
import seaborn as sns

model=modelbank.seq_to_pred_yield_model([[1,8,10],'forest',1,0],['emb_cnn',1])

fig, axs = plt.subplots(1,2,figsize=[4,2],dpi=600)

ax = axs[0]

strain_list=[]
for strain in model.plotpairs_cv[2]:
	if strain[0]==1:
		strain_list.append('$I^q$')
	elif strain[1]==1:
		strain_list.append('$SH$')
	else:
		print('ahhhh')

df = pd.DataFrame({'Predicted':model.plotpairs_cv[1], 'True':model.plotpairs_cv[0],'Strain':strain_list})

df_a= df[df['Strain']=='$I^q$']
print(spearmanr(df_a['Predicted'].values,df_a['True'].values))
# ax.scatter(x=df_a['Predicted'],y=df_a['True'],lw=0,alpha=alpha,color='purple',label=r'$I^q$ $\rho$ = 0.31')
sns.kdeplot(data=df_a,x='Predicted',y='True',color='purple',fill=True,cut=0,ax=ax,levels=5)

df_a= df[df['Strain']=='$SH$']
print(spearmanr(df_a['Predicted'].values,df_a['True'].values))
sns.kdeplot(data=df_a,x='Predicted',y='True',color='maroon',fill=True,cut=0,ax=ax,levels=5)

# ax.scatter(x=df_a['Predicted'],y=df_a['True'],lw=0,alpha=alpha,color='maroon',label=r'$SH$ $\rho$ = 0.31')

ax.set_xlabel('Sequence\nPredicted Yield',fontsize=6)
ax.set_ylabel('Assay Score\nPredicted Yield',fontsize=6)
ax.tick_params(axis='both', which='major', labelsize=6)
ax.set_xticks([-2,-1,0,1,2])
ax.set_yticks([-2,-1,0,1,2])
ax.set_xlim([-2,2.5])
ax.set_ylim([-2,2.5])
ax.set_aspect('equal')
ax.legend().remove()

# ax.legend(fontsize=6,title='Strain',title_fontsize=6,framealpha=1)

ax = axs[1]


strain_list=[]
for strain in model.plotpairs_test[2]:
	if strain[0]==1:
		strain_list.append('$I^q$')
	elif strain[1]==1:
		strain_list.append('$SH$')
	else:
		print('ahhhh')

df = pd.DataFrame({'Predicted':model.plotpairs_test[1], 'True':model.plotpairs_test[0],'Strain':strain_list})
alpha=0.2
df_a= df[df['Strain']=='$I^q$']
print(spearmanr(df_a['Predicted'].values,df_a['True'].values))
ax.scatter(x=df_a['Predicted'],y=df_a['True'],lw=0,alpha=alpha,color='purple',label=r'$I^q$')

df_a= df[df['Strain']=='$SH$']
print(spearmanr(df_a['Predicted'].values,df_a['True'].values))
ax.scatter(x=df_a['Predicted'],y=df_a['True'],lw=0,alpha=alpha,color='maroon',label=r'$SH$')

ax.set_xlabel('Sequence\nPredicted Yield',fontsize=6)
ax.set_ylabel('True\nYield',fontsize=6)
ax.tick_params(axis='both', which='major', labelsize=6)
ax.set_xticks([-2,-1,0,1,2])
ax.set_yticks([-2,-1,0,1,2])
ax.set_xlim([-2,2.5])
ax.set_ylim([-2,2.5])
ax.set_aspect('equal')
ax.legend().remove()
# ax.legend(fontsize=6,title='Strain',title_fontsize=6,framealpha=1)
fig.tight_layout()
fig.savefig('./seq_to_pred_yield_scatter.png')
plt.close()

