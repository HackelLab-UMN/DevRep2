from pyaaisc import Aaindex
import submodels_module
import numpy as np 
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import pandas as pd 

aaindex=Aaindex()
full_list = aaindex.get_all(dbkey='aaindex1')
aa_list=list('ACDEFGHIKLMNPQRSTVWY')
name_list,values_list=[],[]
# for i in range(10):
for i in range(len(full_list)):
	print(i)
	try:
		record=aaindex.get(full_list[i][0])
		values_dic=record.index_data
		values_list_per_record=[values_dic[aa] for aa in aa_list]

		values_list.append(values_list_per_record)
		name_list.append(record.data_description)

	except:
		print('skipped')

s2a = submodels_module.seq_to_assay_model([1,8,10],'emb_cnn',1)
s2a._model.set_model(s2a.get_best_trial()['hyperparam'],xa_len=16,cat_var_len=3,lin_or_sig=s2a.lin_or_sig)
s2a.load_model(0)
emb=s2a._model.model.trainable_variables[0].numpy()

emb_aw = emb[0:19,:]
emb_y = np.expand_dims(emb[20,:],axis=0)

emb_no_gap= np.concatenate((emb_aw,emb_y))

df = pd.DataFrame(emb_no_gap)
df.index = aa_list
class_list = ['Hydrophobic Aliphatic','Special','Negatively Charged','Negatively Charged','Aromatic','Special','Positively Charged','Hydrophobic Aliphatic','Positively Charged','Hydrophobic Aliphatic','Hydrophobic Aliphatic','Polar Uncharged','Special','Polar Uncharged','Positively Charged','Polar Uncharged','Polar Uncharged','Hydrophobic Aliphatic','Aromatic','Aromatic']

df.insert(0,'Class',class_list)
df.to_csv('./DevRep_20AA_embeddings.csv')

pca=PCA(n_components=3,svd_solver='randomized',random_state=420).fit(df.to_numpy()[:,1:])
print(pca.explained_variance_ratio_)
pca_values=pca.transform(df.to_numpy()[:,1:])

for i in range(3):
	pca = pca_values[:,i]
	max_rho = 0
	for name,value in zip(name_list,values_list):
		rho,p = spearmanr(pca,value)
		if rho > max_rho:
			max_rho = rho
			max_name = name
	print(max_name,max_rho)
