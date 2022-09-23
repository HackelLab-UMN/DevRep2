from nested_sampling import compute_heat_capacity, compute_log_dos
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import comb


def calc_thermo(NS_run):

	df = pd.read_pickle('./ns_threshold/'+NS_run.savename+'.pkl')
	d_list = df['Develop'].to_numpy()

	threshold_list=d_list.astype('d')
	
	log_dos_per_step=compute_log_dos(threshold_list,np.double(1),np.double(NS_run.n_walkers),False)
	dos_per_step=np.array([np.exp(ldos) for ldos in log_dos_per_step])
	
	unique_energies = np.unique(threshold_list)[::-1]
	ns_dos=[]
	for e in unique_energies:
		e_dos=dos_per_step[np.where(threshold_list==e)[0]]
		ns_dos.append(sum(e_dos))

	dos_df=pd.DataFrame({'Develop':unique_energies, 'DoS':ns_dos})
	cum_dos=[]
	for d in dos_df.Develop:
		cum_dos.append(sum(dos_df[dos_df['Develop']>=d].DoS))
	dos_df['Cum_DoS']=cum_dos
	dos_df.to_pickle('./ns_dos/'+NS_run.savename+'.pkl')

	# do the computation
	# T, Cv, U, U2 = compute_heat_capacity(energies, K, npar=args.P,
	#                                      ndof=args.ndof, Tmin=args.Tmin, Tmax=args.Tmax,
	#                                      nT=args.nT, live_replicas=args.live)
	if (NS_run.model_name=='potts') and (NS_run.q==2):
		Tmin, Tmax, nT = -10,-1,100
	else:
		# Tmin, Tmax, nT = -10,-.01,100
		Tmin, Tmax, nT = -5,-.001,10000
	T_list, Cv, U, U2 , _F= compute_heat_capacity(threshold_list,NS_run.n_walkers,np.double(1),0,Tmin, Tmax, nT, False)
	F = []
	for t,u in zip(T_list,U):
		z=0
		for g,E in zip(ns_dos,unique_energies):
			if (-E/t) < 400:
				z = z + (g*np.exp(-E/t))
			else:
				z = z + (g*np.exp(400))
		f = -t*np.log(z)
		if f > u:
			f = u
		F.append(f)
	
	U = np.array(U)
	S = np.array([(u-f)/(t) for u,f,t in zip(U,F,T_list)])
	if NS_run.model_name=='potts':
		S = S/NS_run.n_pos
		U = U/NS_run.n_pos
	S = S - max(S) + 1


	fig,axs = plt.subplots(3,2,figsize=[3.5,3.5],dpi=1200,sharex='col')

	# axs[0,0].axis('off')

	ax=axs[0,0]
	ax.plot(unique_energies,ns_dos,'black')
	ax.set_ylabel('Density of States',fontsize=6)
	if 'potts' in NS_run.model_name:
		ax.set_xlabel('Energy',fontsize=6)
	else:
		ax.set_xlabel('Developability',fontsize=6)
	ax.set_yscale('log')
	ax.tick_params(axis='both', which='major', labelsize=6,labelbottom=True,bottom=True)

	ax=axs[1,0]
	ax.plot(unique_energies,cum_dos,'black')
	ax.set_ylabel('Frac. Seqs with Higher Develop.',fontsize=6)
	if 'potts' in NS_run.model_name:
		ax.set_xlabel('Energy',fontsize=6)
	else:
		ax.set_xlabel('Developability',fontsize=6)
	ax.set_yscale('log')
	ax.set_ylim([1e-25,1e0])
	ax.tick_params(axis='both', which='major', labelsize=6,labelbottom=True,bottom=True)

	axs[2,0].axis('off')
	
	ax=axs[0,1]
	ax.plot(-1/T_list,U,'black')
	if 'potts' in NS_run.model_name:
		ax.set_ylabel('<Energy>/L',fontsize=6)
	else:
		ax.set_ylabel('<Developability>',fontsize=6)
	ax.tick_params(axis='both', which='major', labelsize=6)
	ax.set_xscale('log')

	ax=axs[1,1]
	ax.plot(-1/T_list,Cv,'black')
	ax.set_ylabel('Heat Capacity',fontsize=6)
	ax.tick_params(axis='both', which='major', labelsize=6)
	ax.set_xscale('log')

	ax=axs[2,1]
	ax.plot(-1/T_list,S,'black')
	ax.set_xlabel(r'$\beta$',fontsize=6)
	ax.set_ylabel('Entropy',fontsize=6)
	ax.tick_params(axis='both', which='major', labelsize=6)
	ax.set_xscale('log')
	ax.set_ylim([-100,5])

	if NS_run.model_name=='potts' and NS_run.q==2:
		n_sites=NS_run.n_pos
		
		E_list = []
		dos_exact= []
		for k in range(n_sites):
			E_list.append( -n_sites + 1 + 2*k )
			dos_exact.append((2*comb(n_sites-1,k)) / (2**n_sites) )
		E_list = (np.array(E_list)+n_sites)/2 
		axs[1,0].plot(E_list,dos_exact,'r--')

		e_dos= []
		for T in T_list:
			e=0
			z=0
			for E,g in zip(E_list,dos_exact):
				e = e + E*g*np.exp(-E/T)
				z = z + g*np.exp(-E/T)
			e_dos.append(e/z)
		e_dos=np.array(e_dos)/n_sites
		axs[0,1].plot(-T_list,e_dos,'r--')

		J=0.5
		C_exact = [n_sites*(J/T)**2 * (1/(np.cosh(J/T)**2)) for T in T_list]
		axs[1,1].plot(-T_list,C_exact,'r--')
		S_exact_eq = np.array([np.log(2)+n_sites*( (-J/T)*np.tanh(J/T) + np.log(2*np.cosh(J/T)) ) for T in T_list])/n_sites
		S_exact_Eq = S_exact_eq - max(S_exact_eq) + 1
		axs[2,1].plot(-T_list,S_exact_Eq,'r--')

	fig.tight_layout()
	fig.savefig('./thermo_images/'+NS_run.savename+'_thermo.png')

