import sys

import numpy as np
rng = np.random.default_rng()
from _walker import Sequence,Potts
from sequence_models import devrep,onehot,strain,potts,unirep_paratope,potts_mean,potts_mean_h0
import multiprocessing 
from functools import partial
import pandas as pd
import pickle 
import time

def get_proposed_seq(n_positions,walk):
	return walk.propose_MC_step(n_positions)


def try_sa_step(T,walk_p_s_d):
	walk=walk_p_s_d[0]
	return walk.SA_step(T,walk_p_s_d[1],walk_p_s_d[2]) #returns seq, develop, accept


class SA_run():
	def __init__(self,toggle_no):

		if toggle_no == 0 :
			self.walker_type = Potts
			self.model = potts
			self.model_name = 'potts'
			self.n_pos = 25
			self.q = 2
			self.n_walkers = 250

		if toggle_no == 1:
			self.walker_type=Potts
			self.model= devrep
			self.model_name = 'devrep'
			self.n_pos=16
			self.q=21
			self.n_walkers=1000

		if toggle_no == 2:
			self.walker_type=Sequence
			self.model= unirep_paratope
			self.model_name = 'unirep_paratope'
			self.n_pos=16
			self.q=21
			self.n_walkers=1000

		if toggle_no == 3:
			self.walker_type=Sequence
			self.model= onehot
			self.model_name = 'onehot'
			self.n_pos=16
			self.q=21
			self.n_walkers=1000

		if toggle_no == 4:
			self.walker_type=Potts
			self.model= devrep
			self.model_name = 'devrep'
			self.n_pos=16
			self.q=21
			self.n_walkers=100

		if toggle_no == 5:
			self.walker_type=Sequence
			self.model= unirep_paratope
			self.model_name = 'unirep_paratope'
			self.n_pos=16
			self.q=21
			self.n_walkers=100

		if toggle_no == 6:
			self.walker_type=Sequence
			self.model= onehot
			self.model_name = 'onehot'
			self.n_pos=16
			self.q=21
			self.n_walkers=100
		
		if 'potts' in self.model_name:
			self.max_stepsize=int(self.n_pos/4)
		else:
			self.max_stepsize=14
		
		self.savename = self.model_name +'_'+ str(self.n_walkers) +'_'+ str(self.n_pos) +'_'+ str(self.q) 
		self.T_list=np.hstack([np.geomspace(10,0.001,num=100),[0]])

		self.walkers = []
		self.n_positions = 1

	def init_walkers(self):
		self.walkers = [self.walker_type(n_pos=self.n_pos,q=self.q) for _ in range(self.n_walkers)]
		sequences=np.stack([walk.sequence for walk in self.walkers])
		dev_list = self.model(sequences)
		[walk.set_init_develop(dev) for walk,dev in zip(self.walkers,dev_list)]
		self.save(np.inf,1,0)

	def save(self,T,p_accept,i):
		pickle.dump([self.walkers,self.n_positions,T,p_accept], open('./sa_walkers/'+self.savename+'_'+str(T)+'_'+str(i)+'.pkl', "wb" ) )

	# def load(self):
	# 	try:
	# 		self.walkers,self.n_positions = pickle.load(open('./ns_walkers/'+self.savename+'.pkl', "rb"))
	# 	except:
	# 		self.init_walkers()

	def restart(self,T):
		i=499 #restart from end of last temp
		self.walkers,self.n_positions,T,p_accept = pickle.load(open('./sa_walkers/'+self.savename+'_'+str(T)+'_'+str(i)+'.pkl', "rb"))
		new_T_list = self.T_list
		new_T_list = new_T_list[new_T_list<T]
		self.T_list = new_T_list
		print('sucessful restart')
		self.go()


	def go(self):

		pool=multiprocessing.Pool(processes=32)
		tic = time.time()

		for T in self.T_list:
			#warm up
			for _ in range(50):
				get_proposed_seq_filled = partial(get_proposed_seq,self.n_positions)
				proposed_seq = pool.map(get_proposed_seq_filled,self.walkers)
				proposed_dev = self.model(proposed_seq)
				walk_p_s_d=zip(self.walkers,proposed_seq,proposed_dev)

				try_sa_step_filled = partial(try_sa_step,T)
				seq_list,d_list,acc_list = zip(*pool.map(try_sa_step_filled,walk_p_s_d))
				[walk.set_s_d_r(seq,d) for walk,seq,d in zip(self.walkers,seq_list,d_list)]
				
				p_accept=sum(acc_list)/self.n_walkers
				if p_accept > 0.3 and self.n_positions < self.max_stepsize:
					self.n_positions = self.n_positions + 1
				elif p_accept < 0.2 and self.n_positions >1:
					self.n_positions = self.n_positions - 1

			#constant size step
			get_proposed_seq_filled = partial(get_proposed_seq,self.n_positions)
			for i in range(500):
				proposed_seq = pool.map(get_proposed_seq_filled,self.walkers)
				proposed_dev = self.model(proposed_seq)
				walk_p_s_d=zip(self.walkers,proposed_seq,proposed_dev)

				try_sa_step_filled = partial(try_sa_step,T)
				seq_list,d_list,acc_list = zip(*pool.map(try_sa_step_filled,walk_p_s_d))
				[walk.set_s_d_r(seq,d) for walk,seq,d in zip(self.walkers,seq_list,d_list)]

				if i%50 ==0:
					print(T,i,np.average(d_list))
					self.save(T,p_accept,i)
			self.save(T,p_accept,i)
		
		pool.close()
		print(time.time() - tic)



if __name__=='__main__':
	toggle_no=int(sys.argv[1])
	a=SA_run(toggle_no)

	if toggle_no==2:
		a.restart(0.034304692863149154)
	elif toggle_no==5:
		a.restart(0.0030538555088334123)
	# a.init_walkers()
	# a.go()



