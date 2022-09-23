import numpy as np
rng = np.random.default_rng()

class Sequence():
	def __init__(self,**kwargs):
		self.sequence=self.get_init_sequence(kwargs['cysteine'])
		self.rand = rng.random()
		self.develop = np.nan

	def get_init_sequence(self,cysteine):
		# aa_list=list('ACDEFGHIKLMNPQRSTVWXY') Gap=='X' #this is the key to ordinal encoding of AAs; gaps are 19th index
		#cysteine = [0 1 2] = [0 cysteins; max 1 cysteine; infinite allowed cysteines]

		reg_aa=list(range(19))+[20] #exclude gaps
		reg_aa_prune_c=reg_aa.copy()
		reg_aa_prune_c.remove(1) #1 is ordinal encoded for C and we want to remove C: no gaps nor cysteine
		middle_aa=list(range(21)) #allow everything
		middle_aa_prune_c=middle_aa.copy()
		middle_aa_prune_c.remove(1) #allow everything but not cysteine
		
		# print(reg_aa)
		# print(reg_aa_prune_c)
		# print(middle_aa)
		# print(middle_aa_prune_c)

		if cysteine == 2: ##default/infinite amount of cysteines are allowed; this is Alex's original method
			seq=[]
			for i in range(2):
				start=rng.choice(reg_aa,3)
				c=rng.choice(middle_aa,1)
				if c == 19:
					b=rng.choice(middle_aa,1)
				else:
					b=rng.choice(reg_aa,1)
				end=rng.choice(reg_aa,3)
				seq.append(np.concatenate([start,b,c,end]))
			seq=np.concatenate(seq)

		elif cysteine == 0: ##no cysteines are allowed ever: only choose from the "pruned c" pools
			seq=[]
			for i in range(2):
				start=rng.choice(reg_aa_prune_c,3)
				c=rng.choice(middle_aa_prune_c,1)
				if c == 19:
					b=rng.choice(middle_aa_prune_c,1)
				else:
					b=rng.choice(reg_aa_prune_c,1)
				end=rng.choice(reg_aa_prune_c,3)
				seq.append(np.concatenate([start,b,c,end]))
			seq=np.concatenate(seq)	

		elif cysteine == 1: ##maximum of 1 cysteine in the entire sequence is allowed
			seq=[-1 for i in range(16)] #make a list of numerical elts of 16 length with placeholders of -1
			for j in range(16):
				if 1 in seq: ### we already have a c so now restrict what we can mutate
					if j==4 or j==12: #we assign positions 3,4 and 11,12 together so if we see these positions skip them
						pass

					elif j==3 or j==11: #this is the 4th or 11th position that would have a gap [0 1 2 GAP 4 5 6 7] + [8 9 10 GAP 12 13 14 15]
						c=rng.choice(middle_aa_prune_c,1)
						if c==19:
							b=rng.choice(middle_aa_prune_c,1)
						else:
							b=rng.choice(reg_aa_prune_c,1)
						seq[j]=b
						seq[j+1]=c
					
					else: #we can choose from all regular aas that have c removed from them
						seq[j]=rng.choice(reg_aa_prune_c,1)
				
				else: ### we dont have a c so dont restrict what we can mutate
					if j==4 or j==12: #we assign positions 3,4 and 11,12 together so if we see these positions skip them
						pass
					elif j==3 or j==11: #this is the 4th or 11th position that would have a gap [0 1 2 GAP 4 5 6 7] + [8 9 10 GAP 12 13 14 15]
						c=rng.choice(middle_aa,1)
						if c==19:
							b=rng.choice(middle_aa,1)
						else:
							b=rng.choice(reg_aa,1)
						seq[j]=b
						seq[j+1]=c	
					else: #we can choose from all regular aas including c
						seq[j]=rng.choice(reg_aa,1)
			# print(f'before seq is type {type(seq)} and is {seq}')
			seq=np.concatenate(seq)
			# print(f'In self.cysteine=1, type of seq is currently {type(seq)} and is {seq} with shape {seq.shape}')

		else: ##error message: we should only have cysteine param values of 0 1 2 for now
			print('\n')
			print(f'Invalid cysteine parameter value for init: expected {[0,1,2]} but received {cysteine}')
			print('\n')
			
		return seq


	def replace(self,walker):
		self.sequence = walker.sequence
		self.develop = walker.develop
		self.rand = walker.rand

	def set_init_develop(self,develop):
		self.develop = develop

	def set_s_d_r(self,s,d,r=np.nan):
		self.sequence=s
		self.develop=d
		self.rand=r

	def SA_step(self,T,proposed_seq,proposed_dev):
		accept = False
		if proposed_dev >= self.develop:
			accept = True
		else:
			p_accept = np.exp((proposed_dev-self.develop)/T)
			alpha = rng.random()
			if p_accept >= alpha:
				accept = True

		if accept:
			self.sequence = proposed_seq
			self.develop = proposed_dev

		return self.sequence, self.develop, accept

	def NS_step(self,threshold_dev,threshold_rand,proposed_seq,proposed_dev):
		proposed_rand = rng.random()

		accept = False
		if proposed_dev > threshold_dev:
			accept = True
		elif np.isclose(proposed_dev,threshold_dev) and (proposed_rand > threshold_rand):
			accept = True

		if accept:
			self.sequence = proposed_seq
			self.rand = proposed_rand
			self.develop = proposed_dev

		return self.sequence, self.develop, self.rand, accept


	def always_accept_step(self,proposed_seq):
		self.sequence = proposed_seq

	def propose_MC_step(self,num_positions,cysteine):
		proposed_seq=self.sequence #proposed_seq is a LIST of length 16
		legal_moves = 0 
		pos_not_changed = list(range(16)) #list of the positions 

		# aa_list=list('ACDEFGHIKLMNPQRSTVWXY') Gap=='X' #this is the key to ordinal encoding of AAs; gaps are 19th index
		#cysteine = [0 1 2] = [0 cysteins; max 1 cysteine; infinite allowed cysteines]

		reg_aa=list(range(19))+[20] #exclude gaps
		reg_aa_prune_c=reg_aa.copy()
		reg_aa_prune_c.remove(1) #1 is ordinal encoded for C and we want to remove C: no gaps nor cysteine
		middle_aa=list(range(21)) #allow everything
		middle_aa_prune_c=middle_aa.copy()
		middle_aa_prune_c.remove(1) #allow everything but not cysteine

		if cysteine == 2: ##default/infinite amount of cysteines are allowed; this is Alex's original method
			while legal_moves < num_positions:
				pos = rng.choice(pos_not_changed)
				if pos == 3: #pos 3 can only be gap if pos 4 is gap
					if proposed_seq[4]==19:
						aa_posib=middle_aa.copy()
					else:
						aa_posib=reg_aa.copy()
				elif pos == 4: #pos 4 cant mutate if pos 3 is gap
					if proposed_seq[3]==19:
						continue
					else:
						aa_posib=middle_aa.copy()
				elif pos == 11:
					if proposed_seq[12]==19:
						aa_posib=middle_aa.copy()
					else:
						aa_posib=reg_aa.copy()
				elif pos == 12:
					if proposed_seq[11] == 19:
						continue
					else:
						aa_posib=middle_aa.copy()
				else: 
					aa_posib=reg_aa.copy()
				#remove current amino acid from set of possibilities 
				cur_aa = proposed_seq[pos] 
				aa_posib.remove(cur_aa)

				proposed_seq[pos]=rng.choice(aa_posib)
				legal_moves=legal_moves+1
				pos_not_changed.remove(pos)

		elif cysteine == 0: ##no cysteines are allowed ever: only choose from the "pruned c" pools
			while legal_moves < num_positions:
				pos = rng.choice(pos_not_changed)
				if pos == 3: #pos 3 can only be gap if pos 4 is gap
					if proposed_seq[4]==19:
						aa_posib=middle_aa_prune_c.copy()
					else:
						aa_posib=reg_aa_prune_c.copy()
				elif pos == 4: #pos 4 cant mutate if pos 3 is gap
					if proposed_seq[3]==19:
						continue
					else:
						aa_posib=middle_aa_prune_c.copy()
				elif pos == 11:
					if proposed_seq[12]==19:
						aa_posib=middle_aa_prune_c.copy()
					else:
						aa_posib=reg_aa_prune_c.copy()
				elif pos == 12:
					if proposed_seq[11] == 19:
						continue
					else:
						aa_posib=middle_aa_prune_c.copy()
				else: 
					aa_posib=reg_aa_prune_c.copy()
				#remove current amino acid from set of possibilities 
				cur_aa = proposed_seq[pos] 
				aa_posib.remove(cur_aa)

				proposed_seq[pos]=rng.choice(aa_posib)
				legal_moves=legal_moves+1
				pos_not_changed.remove(pos)

		elif cysteine == 1: ##maximum of 1 cysteine in the entire sequence is allowed
			while legal_moves < num_positions:
				pos = rng.choice(pos_not_changed)
				
				###
				if 1 in proposed_seq: ### we already have a c so now restrict what we can mutate
					if pos == 3: #pos 3 can only be gap if pos 4 is gap
						if proposed_seq[4]==19:
							aa_posib=middle_aa_prune_c.copy()
						else:
							aa_posib=reg_aa_prune_c.copy()
					elif pos == 4: #pos 4 cant mutate if pos 3 is gap
						if proposed_seq[3]==19:
							continue
						else:
							aa_posib=middle_aa_prune_c.copy()
					elif pos == 11:
						if proposed_seq[12]==19:
							aa_posib=middle_aa_prune_c.copy()
						else:
							aa_posib=reg_aa_prune_c.copy()
					elif pos == 12:
						if proposed_seq[11] == 19:
							continue
						else:
							aa_posib=middle_aa_prune_c.copy()
					else: 
						aa_posib=reg_aa_prune_c.copy()
					#remove current amino acid from set of possibilities 
					cur_aa = proposed_seq[pos] 
					# print(f'detected 1; current aa is:  {cur_aa} and the possible aa are: {aa_posib}')
					if cur_aa in aa_posib:
						aa_posib.remove(cur_aa)

					proposed_seq[pos]=rng.choice(aa_posib)
					legal_moves=legal_moves+1
					pos_not_changed.remove(pos)

				###
				else:
					if pos == 3: #pos 3 can only be gap if pos 4 is gap
						if proposed_seq[4]==19:
							aa_posib=middle_aa.copy()
						else:
							aa_posib=reg_aa.copy()
					elif pos == 4: #pos 4 cant mutate if pos 3 is gap
						if proposed_seq[3]==19:
							continue
						else:
							aa_posib=middle_aa.copy()
					elif pos == 11:
						if proposed_seq[12]==19:
							aa_posib=middle_aa.copy()
						else:
							aa_posib=reg_aa.copy()
					elif pos == 12:
						if proposed_seq[11] == 19:
							continue
						else:
							aa_posib=middle_aa.copy()
					else: 
						aa_posib=reg_aa.copy()
					#remove current amino acid from set of possibilities 
					cur_aa = proposed_seq[pos] 
					# print(f'didnt detect 1; current aa is:  {cur_aa} and the possible aa are: {aa_posib}')
					
					if cur_aa in aa_posib:
						aa_posib.remove(cur_aa)
					# aa_posib.remove(cur_aa)

					proposed_seq[pos]=rng.choice(aa_posib)
					legal_moves=legal_moves+1
					pos_not_changed.remove(pos)

		else: ##error message: we should only have cysteine param values of 0 1 2 for now
			print('\n')
			print(f'Invalid cysteine parameter value for MC step: expected {[0,1,2]} but received {cysteine}')
			print('\n')


		return proposed_seq

class Potts(Sequence):
	def __init__(self,**kwargs):
		self.sequence=self.get_init_sequence(kwargs['n_pos'],kwargs['q'])
		self.rand = rng.random()
		self.develop = 0.0

	def get_init_sequence(self,n_pos,q):
		self.aa_posib = list(range(q))
		return rng.choice(self.aa_posib,size=n_pos,replace=True)

	def propose_MC_step(self,num_positions):
		proposed_seq=self.sequence
		legal_moves = 0 
		pos_not_changed = list(range(len(proposed_seq)))

		while legal_moves < num_positions:
			pos = rng.choice(pos_not_changed)
			cur_aa = proposed_seq[pos]
			cur_aa_possib = self.aa_posib.copy()
			cur_aa_possib.remove(cur_aa)
			proposed_seq[pos]=rng.choice(cur_aa_possib)
			legal_moves = legal_moves +1
			pos_not_changed.remove(pos)

		return proposed_seq