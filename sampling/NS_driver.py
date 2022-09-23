import sys

import numpy as np
rng = np.random.default_rng()
from _walker import Sequence,Potts
from sequence_models import devrep,onehot,strain,potts,unirep_paratope,potts_mean,potts_mean_h0
import multiprocessing 
from functools import partial
import pandas as pd
from _thermo import calc_thermo
from _graph_vis import make_graph,plot_graph_3d,discont_plot
import pickle 
import time

def get_proposed_seq(n_positions,cysteine,walk):
	return walk.propose_MC_step(n_positions,cysteine)

def try_ns_step(threshold_dev,threshold_rand,walk_p_s_d):
	walk=walk_p_s_d[0]
	return walk.NS_step(threshold_dev,threshold_rand,walk_p_s_d[1],walk_p_s_d[2]) #returns seq, develop, rand, accept


class NS_run():
	def __init__(self,toggle_no):

		if toggle_no == 0 :
			self.walker_type = Potts
			self.model = potts
			self.model_name = 'potts' 
			self.n_pos = 25#number of positions of a given sequence we mutate 
			self.q = 2 #number of AAs/characters you have in your pool/library to mutate to/from
			self.n_walkers = 250 #number of mutating sequences over each nested sampling iteration="walker"
			self.cysteine = 2 #[0 1 2] = [0 cysteins; max 1 cysteine; infinite allowed cysteines]

		if toggle_no == 1:
			self.walker_type=Sequence
			self.model= devrep
			self.model_name = 'devrep'
			self.n_pos=16
			self.q=21
			self.n_walkers=1000
			self.cysteine = 2

		if toggle_no == 2:
			self.walker_type=Sequence
			self.model= unirep_paratope
			self.model_name = 'unirep_paratope'
			self.n_pos=16
			self.q=21
			self.n_walkers=1000
			self.cysteine = 2

		if toggle_no == 3:
			self.walker_type=Sequence
			self.model= onehot
			self.model_name = 'onehot'
			self.n_pos=16
			self.q=21
			self.n_walkers=1000
			self.cysteine = 2

		if toggle_no ==4:
			self.walker_type = Potts
			self.model = potts_mean
			self.model_name = 'potts_mean'
			self.n_pos = 25
			self.q = 2
			self.n_walkers = 250
			self.cysteine = 2

		if toggle_no ==5:
			self.walker_type = Potts
			self.model = potts_mean
			self.model_name = 'potts_mean'
			self.n_pos = 25
			self.q = 3
			self.n_walkers = 250
			self.cysteine = 2

		if toggle_no ==6:
			self.walker_type = Potts
			self.model = potts_mean
			self.model_name = 'potts_mean'
			self.n_pos = 25
			self.q = 4
			self.n_walkers = 250
			self.cysteine = 2

		if toggle_no ==7:
			self.walker_type = Potts
			self.model = potts_mean_h0
			self.model_name = 'potts_mean_h0'
			self.n_pos = 25
			self.q = 4
			self.n_walkers = 250
			self.cysteine = 2

		if toggle_no == 8:
			self.walker_type=Sequence
			self.model= devrep
			self.model_name = 'devrep'
			self.n_pos=16
			self.q=21
			self.n_walkers=100
			self.cysteine = 2

		if toggle_no == 9:
			self.walker_type=Sequence
			self.model= unirep_paratope
			self.model_name = 'unirep_paratope'
			self.n_pos=16
			self.q=21
			self.n_walkers=100
			self.cysteine = 2

		if toggle_no == 10:
			self.walker_type=Sequence
			self.model= onehot
			self.model_name = 'onehot'
			self.n_pos=16
			self.q=21
			self.n_walkers=100
			self.cysteine = 2
		
		if toggle_no == 11:
			self.walker_type=Sequence
			self.model= devrep
			self.model_name = 'devrep'
			self.n_pos=16
			self.q=21
			self.n_walkers=100
			self.cysteine = 0
		
		if toggle_no == 12:
			self.walker_type=Sequence
			self.model= devrep
			self.model_name = 'devrep'
			self.n_pos=16
			self.q=21
			self.n_walkers=100
			self.cysteine = 1
		
		#this is the original gold standard unlimited cysteine devrep sequence 16 position toggle_no
		if toggle_no == 13:
			self.walker_type=Sequence
			self.model= devrep
			self.model_name = 'devrep'
			self.n_pos=16
			self.q=21
			self.n_walkers=100
			self.cysteine = 2

		if 'potts' in self.model_name:
			self.max_stepsize=int(self.n_pos/4)
		else:
			self.max_stepsize=14
			
		self.savename = self.model_name +'_'+ str(self.n_walkers) +'_'+ str(self.n_pos) +'_'+ str(self.q) + '_' + str(self.cysteine)
		self.threshold_seqs = []
		self.threshold_devs = []
		self.walkers = []
		self.n_positions = 1

	def init_walkers(self):
		self.walkers = [self.walker_type(n_pos=self.n_pos,q=self.q,cysteine=self.cysteine) for _ in range(self.n_walkers)]#n_pos and q only seem to be used in Potts()
		sequences=np.stack([walk.sequence for walk in self.walkers])
		dev_list = self.model(sequences)
		[walk.set_init_develop(dev) for walk,dev in zip(self.walkers,dev_list)]

		d_and_r_list=np.array([(w.develop,w.rand) for w in self.walkers],dtype=np.dtype([('d', float), ('r', float)]))
		idx = np.argsort(d_and_r_list,order=('d','r')) #rank by dev then rand
		self.threshold_dev,self.threshold_rand = d_and_r_list[idx[0]] #lowest dev is idx[0]

		self.threshold_seqs.append(self.walkers[idx[0]].sequence)
		self.threshold_devs.append(self.threshold_dev)

		self.ns_loop=0
		self.save()

	def save(self):
		pickle.dump([self.walkers,self.n_positions], open('./ns_walkers/'+self.savename+'_'+str(self.ns_loop)+'.pkl', "wb" ) )
		df = pd.DataFrame({'Sequence':self.threshold_seqs,'Develop':self.threshold_devs})
		df.to_pickle('./ns_threshold/'+self.savename+'.pkl')

	def restart(self,ns_loop):
		self.ns_loop = ns_loop
		self.walkers,self.n_positions = pickle.load(open('./ns_walkers/'+self.savename+'_'+str(self.ns_loop)+'.pkl', "rb"))
		d_and_r_list=np.array([(w.develop,w.rand) for w in self.walkers],dtype=np.dtype([('d', float), ('r', float)]))
		idx = np.argsort(d_and_r_list,order=('d','r')) #rank by dev then rand
		self.threshold_dev,self.threshold_rand = d_and_r_list[idx[0]] #lowest dev is idx[0]


		df = pd.read_pickle('./ns_threshold/'+self.savename+'.pkl')
		self.threshold_seqs = df.Sequence.to_list()
		self.threshold_devs = df.Develop.to_list()

		print('sucessful restart')
		self.go()


	def go(self):

		pool=multiprocessing.Pool(processes=32)
		tic = time.time()
		while True:
		# for ns_loop in range(10000): #could change 
			self.ns_loop = self.ns_loop + 1
			p_accept_list=[]
			n_mc_steps=5
			for mc_step in range(n_mc_steps):
				get_proposed_seq_filled = partial(get_proposed_seq,self.n_positions,self.cysteine)#this is just setting a default for the get_proposed_seq() function
				
				#shift to parallel for the MC step! Each "thread" will work on a single walker as seen in "self.walkers"
				proposed_seq = pool.map(get_proposed_seq_filled,self.walkers)#like "do this function in parallel..." "on this set of walkers"
				proposed_dev = self.model(proposed_seq) #record proposed sequence's developability
				
				#shift to a protected/"safe" checkpoint in which you record all walkers' (sequence,proposed_dev)
				walk_p_s_d=zip(self.walkers,proposed_seq,proposed_dev)

				#now proceed with parallelization: each walker is going to be mutated
				try_ns_step_filled = partial(try_ns_step,self.threshold_dev,self.threshold_rand)#autofill the try_ns_step function with all the necessary toggle params
				seq_list,d_list,r_list,acc_list = zip(*pool.map(try_ns_step_filled,walk_p_s_d)) #first: perform the ns_step attempt for all walkers in parallel; then: take all of the 3 outputs for every single proposed ns_step and return them in list form for that MC_step
				d_and_r_list=np.array([(d,r)for d,r in zip(d_list,r_list)],dtype=np.dtype([('d', float), ('r', float)]))#we have the max amount of info above, BUT the proposed_ns_step just wants (dev, random_float) for Nested Sampling (not SA)
				[walk.set_s_d_r(seq,d_r[0],d_r[1]) for walk,seq,d_r in zip(self.walkers,seq_list,d_and_r_list)]#d_r is actually an element in d_and_r_list which just has the d and r for the individual walker
				
				p_accept=sum(acc_list)/self.n_walkers #number of accepted mutations divided by number of all walkers
				p_accept_list.append(p_accept)# record the running/current percent of accepted mutations from the current to proposed sequence
				# p_accept_list.append(sum([walk.NS_step(self.threshold_dev,self.threshold_rand,p_seq,p_dev) for walk,p_seq,p_dev in zip(self.walkers,proposed_seq,proposed_dev)])/self.n_walkers)
			
			# print(d_and_r_list)
			
			idx = np.argsort(d_and_r_list,order=('d','r'))#quicksort, return list least to greatest; first compare developabilities
			self.threshold_dev,self.threshold_rand = d_and_r_list[idx[0]]#set the corresponding lowest threshold for both dev and rand
			self.threshold_seqs.append(seq_list[idx[0]])
			self.threshold_devs.append(self.threshold_dev)
			
			#termination condition: there is only 1 unique sequence in walkers
			if len(np.unique(np.stack([walk.sequence for walk in self.walkers]),axis=0))==1:
				break

			idx_copy = rng.choice(idx[1:])
			self.walkers[idx[0]].replace(self.walkers[idx_copy])

			#assess the difficulty of performing k mutations (n_positions to mutate); if not enough accepted mutations, decrease the
			#number of positions that can be mutated for one instance of generating a proposed_seq
			p_accept=np.average(p_accept_list)
			if p_accept > 0.3 and self.n_positions <self.max_stepsize:
				self.n_positions = self.n_positions + 1
			elif p_accept < 0.2 and self.n_positions >1:
				self.n_positions = self.n_positions - 1

			#print the loop number, threshold (lowest) dev in that loop, threshold rand (rand associated with threshold dev), and k number of positions allowed to mutate at that step
			if self.ns_loop%50==0:
				print(self.ns_loop,self.threshold_dev,self.threshold_rand,self.n_positions)
				self.save()

		pool.close()
		print(time.time() - tic)
		self.save()



if __name__=='__main__':
	print(f'sys.argv is: {sys.argv}')
	print('\n')
	toggle_no=int(sys.argv[1])
	# toggle_no=12
	
	a=NS_run(toggle_no)
	# a.init_walkers()
	# a.go()
	
	# if toggle_no==2:
	# 	a.restart(6250)
	# else:
	
	# calc_thermo(a)
	# make_graph(a)
	# plot_graph_3d(a)
	discont_plot(a)


