import numpy as np
from scipy.stats import mode
from main_alt_emb_to_yield import alt_emb_to_yield_model
from submodels_module import seq_to_assay_model, sequence_embeding_to_yield_model, seq_to_yield_model, control_to_yield_model

import os
from jax_unirep.layers import mLSTM
from jax import vmap
from functools import partial
from jax_unirep.utils import get_embeddings, load_params, load_embedding, get_embedding
import multiprocessing 


os.chdir('..')

s2a_params=[[1,8,10],'emb_cnn',1]
s2a = seq_to_assay_model(*s2a_params)
s2a._model.set_model(s2a.get_best_trial()['hyperparam'],xa_len=16,cat_var_len=3,lin_or_sig=s2a.lin_or_sig)
s2a.load_model(0)
s2e =  s2a._model.get_seq_embeding_layer_model()

e2y = sequence_embeding_to_yield_model(s2a_params+[0],'svm',1)
e2y.load_model(0)
e2y = e2y._model.model

s2y = seq_to_yield_model('forest',1)
s2y.load_model(0)
s2y = s2y._model.model

c2y = control_to_yield_model('ridge',1)
c2y.load_model(0)
c2y = c2y._model.model

up2y = alt_emb_to_yield_model('para_unirep','svm',1)
up2y.load_model(0)
up2y = up2y._model.model
params = load_params()[1]
_, apply_fun = mLSTM(output_dim=1900)


os.chdir('./sampling/')

def devrep(sequences):

	devrep=s2e.predict(np.hstack([sequences,[[0]*3]*len(sequences)]))

	iq=e2y.predict(np.hstack([devrep,[[1,0]]*len(devrep)]))
	sh=e2y.predict(np.hstack([devrep,[[0,1]]*len(devrep)]))
	developability=iq+sh
	return developability

def get_devrep_rep(sequences):
	return s2e.predict(np.hstack([sequences,[[0]*3]*len(sequences)]))


def get_onehot_rep(sequences):
	oh_sequences=[]
	for seq in sequences:
		oh_seq = np.array([[0]*21]*16)
		for i,j in enumerate(seq):
			oh_seq[i,j]=1
		oh_sequences.append(oh_seq.flatten())
	return oh_sequences


def onehot(sequences):
	oh_sequences=[]
	for seq in sequences:
		oh_seq = np.array([[0]*21]*16)
		for i,j in enumerate(seq):
			oh_seq[i,j]=1
		oh_sequences.append(oh_seq.flatten())

	iq=s2y.predict(np.hstack([oh_sequences,[[1,0]]*len(oh_sequences)]))
	sh=s2y.predict(np.hstack([oh_sequences,[[0,1]]*len(oh_sequences)]))
	developability=iq+sh
	return developability

def strain(sequences):
	iq=c2y.predict([[1,0]]*len(sequences))
	sh=c2y.predict([[0,1]]*len(sequences))
	developability=iq+sh
	return developability

def get_potts_rep(q,sequences):
	seqs = np.stack(sequences)
	rep = []
	for i in range(q):
		rep.append(np.sum(seqs==i,axis=1))
	return np.vstack(rep).T

def potts(sequences):
	seqs = np.stack(sequences)
	developability = np.array([0]*len(seqs))
	seq_len = np.shape(seqs)[1]
	for i in range(seq_len-1):
		developability = developability + (seqs[:,i]==seqs[:,i+1])

	developability = developability + (seqs[:,-1]==seqs[:,0]) #periodic BC
	# developability= developability *-1
	return developability

def potts_mean(sequences):
	seqs = np.stack(sequences)
	developability = np.array([0]*len(seqs))
	seq_len = np.shape(seqs)[1]
	for i in range(seq_len-1):
		developability = developability + (seqs[:,i]==seqs[:,i+1])

	developability = developability + (seqs[:,-1]==seqs[:,0]) #periodic BC
	_,c=mode(seqs,axis=1)
	c=np.squeeze(c)

	developability=np.sum([developability,c],axis=0)

	return developability

def potts_mean_h0(sequences):
	seqs = np.stack(sequences)
	developability = np.array([0]*len(seqs))
	seq_len = np.shape(seqs)[1]
	for i in range(seq_len-1):
		developability = developability + (seqs[:,i]==seqs[:,i+1])

	developability = developability + (seqs[:,-1]==seqs[:,0]) #periodic BC
	
	_,c=mode(seqs,axis=1)
	c=np.squeeze(c)
	
	h = np.sum(np.where(seqs==0,2/seq_len,0),axis=1)

	developability=np.sum([developability,c,h],axis=0)

	return developability

def get_full_seq(seq):
	aa_list=list('ACDEFGHIKLMNPQRSTVWXY')

	para_a = ''.join([aa_list[i] for i in seq[0:8] if not i==19])
	para_b = ''.join([aa_list[i] for i in seq[8:] if not i==19])

	seq_start='KFWATV'
	seq_mid='FEVPVYAETLDEALQLAEWQY'
	seq_end='VTRVRP'

	start_len=len('!KFWATV') 
	mid_len=len(seq_mid)

	para_loc_a=list(range(start_len,start_len+len(para_a)))
	b_start=start_len+len(para_a)+mid_len
	para_loc_b=list(range(b_start,b_start+len(para_b)))

	full_seq = seq_start + para_a + seq_mid + para_b + seq_end
	while len(full_seq)<49:
		full_seq = full_seq + 'X' #add end gaps to make same length, doesnt effect paratope average
	para_loc = para_loc_a + para_loc_b

	return full_seq,para_loc

def para_average(u_p):
	unirep=u_p[0]
	para=u_p[1]
	return np.average(unirep[para,:],axis=0)

def get_unirep_rep(sequences):
	seq_list,para_list = zip(*[get_full_seq(x) for x in sequences])
	x=get_embeddings(seq_list)
	params = load_params()[1]
	_, apply_fun = mLSTM(output_dim=1900)
	h_final, c_final, h = vmap(partial(apply_fun, params))(x)
	unirep_list = h.squeeze()
	para_unirep = [np.average(unirep[para,:],axis=0) for unirep,para in zip(unirep_list,para_list)]

	return para_unirep

def unirep_paratope(sequences):
	pool=multiprocessing.Pool(processes=32)

	seq_list,para_list = zip(*pool.map(get_full_seq,sequences))
	x=get_embeddings(seq_list)
	h_final, c_final, h = vmap(partial(apply_fun, params))(x)
	unirep_list = h.squeeze()
	u_p= zip(unirep_list,para_list)

	para_unirep = pool.map(para_average,u_p)
	pool.close()
	# para_unirep = [np.average(unirep[para,:],axis=0) for unirep,para in zip(unirep_list,para_list)]
	iq=up2y.predict(np.hstack([para_unirep,[[1,0]]*len(para_unirep)]))
	sh=up2y.predict(np.hstack([para_unirep,[[0,1]]*len(para_unirep)]))
	developability=iq+sh
	return developability