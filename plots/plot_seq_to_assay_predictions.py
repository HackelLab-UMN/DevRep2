import submodels_module as modelbank
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

model = modelbank.seq_to_assay_model([1,8,10],'emb_cnn',1)

assay_list=[]
for assay in model.plotpairs_cv[2]:
	if assay[0]==1:
		assay_list.append('$P_{PK37}$')
	elif assay[1]==1:
		assay_list.append('$G_{SH}$')
	elif assay[2]==1:
		assay_list.append(r'$\beta_{SH}$')
	else:
		print('ahhhh')

df = pd.DataFrame({'Predicted':model.plotpairs_cv[1], 'True':model.plotpairs_cv[0],'Assay':assay_list})

# df= df.sample(n=1000)

fig,ax=plt.subplots(1,1,figsize=[2,2],dpi=600)
df_a= df[df['Assay']=='$P_{PK37}$']
print(spearmanr(df_a['Predicted'].values,df_a['True'].values))
kde=sns.kdeplot(data=df_a,x='Predicted',y='True',fill=True,levels=5,ax=ax,cut=0,label='$P_{PK37}$')
df_a= df[df['Assay']=='$G_{SH}$']
print(spearmanr(df_a['Predicted'].values,df_a['True'].values))
kde=sns.kdeplot(data=df_a,x='Predicted',y='True',fill=True,levels=5,ax=ax,cut=0,label='$G_{SH}$')
df_a= df[df['Assay']==r'$\beta_{SH}$']
print(spearmanr(df_a['Predicted'].values,df_a['True'].values))
kde=sns.kdeplot(data=df_a,x='Predicted',y='True',fill=True,levels=5,ax=ax,cut=0,label=r'$\beta_{SH}$')
ax.set_xlabel('Predicted\nAssay Score',fontsize=6)
ax.set_ylabel('True\nAssay Score',fontsize=6)
ax.tick_params(axis='both', which='major', labelsize=6)
ax.set_xticks([0,0.5,1])
ax.set_yticks([0,0.5,1])
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_aspect('equal')
ax.legend().remove()
# ax.legend(fontsize=6,framealpha=1)


fig.tight_layout()
fig.savefig('./seq_to_assay_cv_kde.png')
plt.close()



assay_list=[]
for assay in model.plotpairs_test[2]:
	if assay[0]==1:
		assay_list.append('$P_{PK37}$')
	elif assay[1]==1:
		assay_list.append('$G_{SH}$')
	elif assay[2]==1:
		assay_list.append(r'$\beta_{SH}$')
	else:
		print('ahhhh')

df = pd.DataFrame({'Predicted':model.plotpairs_test[1], 'True':model.plotpairs_test[0],'Assay':assay_list})

# df= df.sample(n=1000)

fig,ax=plt.subplots(1,1,figsize=[2,2],dpi=600)
df_a= df[df['Assay']=='$P_{PK37}$']
print(spearmanr(df_a['Predicted'].values,df_a['True'].values))
kde=sns.kdeplot(data=df_a,x='Predicted',y='True',fill=True,levels=5,ax=ax,cut=0,label='$P_{PK37}$')
df_a= df[df['Assay']=='$G_{SH}$']
print(spearmanr(df_a['Predicted'].values,df_a['True'].values))
kde=sns.kdeplot(data=df_a,x='Predicted',y='True',fill=True,levels=5,ax=ax,cut=0,label='$G_{SH}$')
df_a= df[df['Assay']==r'$\beta_{SH}$']
print(spearmanr(df_a['Predicted'].values,df_a['True'].values))
kde=sns.kdeplot(data=df_a,x='Predicted',y='True',fill=True,levels=5,ax=ax,cut=0,label=r'$\beta_{SH}$')
ax.set_xlabel('Predicted\nAssay Score',fontsize=6)
ax.set_ylabel('True\nAssay Score',fontsize=6)
ax.tick_params(axis='both', which='major', labelsize=6)
ax.set_xticks([0,0.5,1])
ax.set_yticks([0,0.5,1])
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_aspect('equal')
ax.legend().remove()
# ax.legend(fontsize=6,framealpha=1)


fig.tight_layout()
fig.savefig('./seq_to_assay_test_kde.png')
plt.close()
