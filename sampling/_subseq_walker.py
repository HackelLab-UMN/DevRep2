from _walker import Sequence
import numpy as np
rng = np.random.default_rng()


class Sequence_66(Sequence):
	def __init__(self,**kwargs):
		Sequence.__init__(self,**kwargs)

	def get_init_sequence(self):
		# aa_list=list('ACDEFGHIKLMNPQRSTVWXY') Gap=='X'
		reg_aa=list(range(19))+[20]
		middle_aa=list(range(21))

		seq=[]
		for i in range(2):
			start=rng.choice(reg_aa,3)
			c=[19]
			b=[19]
			end=rng.choice(reg_aa,3)
			seq.append(np.concatenate([start,b,c,end]))

		seq=np.concatenate(seq)
		return seq