#Add directory with Frederik's code to path
from sys import path
path.append("../Frederik/")

from units import *

from numpy import exp,sign,sqrt,pi,load,save,linspace,zeros,complex64,array
from numpy.random import default_rng
from numpy.linalg import norm,svd
from math import factorial

from time_evo import find_evolutions, time_evolve
from SSE_lab import SSE_lab

from multiprocessing import Pool
from functools import partial
#from numba import jit

from time import perf_counter
 
#@jit(nopython=True)
def g(omega,Temp,Lambda,omega_0=1):
	"""
	Bath jump correlator g(\omega) = \sqrt{J(\omega)/2pi}
	"""
	if abs(omega) < 1e-14:
		J = Temp/omega_0
	else:
		S_0 = abs(omega)*exp(-(omega**2)/(2*Lambda**2))/omega_0
		BE = (1)/(1-exp(-omega/Temp))*sign(omega)
		J = S_0 * BE

	return sqrt(J/(2*pi))


def bump_function(u,k=2,tol=1e-3):
	"""
	A smooth function which is identically zero for  u <= 0 or u >= 1, and quickly rises to unity
		for u in 0 < u < 1.
	Params:
		u: float
			The argument of the function
		k: int
			Parameter controlling how quickly the function rises to 1 on 0 <= u <= 1. Must be a positive 
				integer. Defaults to 2.

	Returns:
		Value of the bump function at the given u
	"""
	if u <= tol or u >= 1-tol:
		return 0.0
	else:
		x = 2*u - 1
		return 1 - exp(-1/(x**(2*k)-1))/(exp(-1/(x**(2*k)-1)) + exp(-1/(2-x**(2*k))))
	

def bump_function_sq(u,k=2,tol=1e-3):
	return bump_function(u,k,tol)**2


if __name__ == "__main__":
	from sys import argv
	#from time import perf_counter

	if len(argv) != 13:
		print("Usage: delta_t E_J gamma T Lambda num_threads num_samples periods max_wells num_binary_window <infile_wells> <outfile>")
		print("Units: \n -> All times should be given in picoseconds\n -> E_J in GHz*hbar \n -> gamma in meV\n -> T in Kelvin\n -> Lambda in GHz")
		exit(0)

	#Command line arguments
	delta_t = float(argv[1])*picosecond
	E_J = float(argv[2])*GHz*hbar
	gamma = float(argv[3])*meV
	Temp = float(argv[4])*Kelvin
	Lambda = float(argv[5])*GHz
	num_threads = int(argv[6])
	num_samples = int(argv[7])
	num_periods = int(argv[8])
	max_wells = int(argv[9])
	num_binary_window = int(argv[10]) #Number of times in the first window to sample for binary search
	infile_wells = argv[11]
	outfile = argv[12]


	parameters = {"delta_t":delta_t , "E_J":E_J, "gamma":gamma, "Temp":Temp, "Lambda":Lambda}



	print("Let's do it!")

	#Load in the data about the lattice of well eigenstates
	well_nums , well_energies, well_vecs, rms_vals, H_LCs, X1s, X2s, quarter_cycle_matrices = load(infile_wells,allow_pickle=True) 
	num_wells  = len(well_nums)
	zero_well_ind = (num_wells-1)//2

	#Keep wells only up to a maximal well index (max_wells from command line)
	if max_wells >= well_nums[-1]:
		well_indices = list(range(num_wells))
		well_numbers = well_nums
	else:
		well_indices = list(range(zero_well_ind - max_wells,zero_well_ind+max_wells+1))
		well_numbers = list(range(-max_wells,max_wells+1))

	
		
	print("Doing first window jump ops... ",end="")
	#Compute the jump operators in first window
	L1s , L2s , = {} , {}
	for i in well_indices:
		d = len(well_energies[i])
		g_mat = zeros((d,d))
		for m in range(d):
			for n in range(d):
				g_mat[m,n] = g(well_energies[i][n]-well_energies[i][m],Temp,Lambda)

		L1s[well_nums[i]] = 2*pi*sqrt(gamma)*g_mat*X1s[i]
		L2s[well_nums[i]] = 2*pi*sqrt(gamma)*g_mat*X2s[i]
	
	print("Done")

	print("Now doing time evo operators... ",end="")
	#### Compute the time evolution operators in the first window in each well
	#	NB: there is no interwell coupling in this step!
	H_eff_herm , H_eff_NH = {} , {}
	timescale = delta_t/2**num_binary_window
	for i in well_indices:
		well_num = well_nums[i]
		H_eff_herm[well_num]= diag(well_energies[i])
		H_eff_NH[well_num] =  -0.5j*(L1s[well_num].conj().T @ L1s[well_num] + L2s[well_num].conj().T @ L2s[well_num])
		
		_ , S , _ = svd(H_eff_herm[well_num] + H_eff_NH[well_num])
		timescale = min(timescale,hbar/S[0])

	dt = 0.1*timescale
	print("Identified timestep: {}".format(dt/picosecond))

	#Parameters (for parallelization)
	num_tpoints = 2**num_binary_window
	t_vals_window = linspace(0,delta_t,num=num_tpoints+1)
	#dt = t_vals_window[1] - t_vals_window[0]
	params = []

	dUs_window = [{} for i in range(num_tpoints)]	
	Us_window = [{} for i in range(num_tpoints+1)]
	for i in well_indices:
		Us_window[0][well_nums[i]] = eye(len(well_energies[i]))


	#Define a function for computing the time evolution operators in parallel
	f = partial(find_evolutions,dt=dt,H_effs_herm=H_eff_herm,H_effs_NH=H_eff_NH,W=bump_function_sq,well_nums=well_numbers,delta_t=delta_t)

	#Define the endpoints of each numerical integration as arguments for parallelization
	t_args = zip(t_vals_window[:-1],t_vals_window[1:])

	# #Compute the evolutions U(t+dt) in parallel for different ts, and then stitch together the results
	with Pool(processes=num_threads) as p:
		for i, dUs in enumerate(p.starmap(f,t_args,chunksize=int(ceil((num_tpoints-1)/num_threads)))):
			for j,well_num in enumerate(well_numbers):
				dUs_window[i][well_num] = dUs[j]
				Us_window[i+1][well_num] = Us_window[i][well_num] @ dUs[j]


	#_, S , _ = svd(Us_window[-1][0])
	#print("Largest three singular values in well 0: {}, {}, {}".format(*S[:3]))
	# for i in range(num_tpoints):
	# 	_, S , _ = svd(dUs_window[i][0])
	# 	print("Largest singular value of dU at t/Delta = {}: {}".format(t_vals_window[i]/delta_t,S[0]))

	# exit(0)

	print("Done!")



	## ****                   **** ##
	##	    The SSE evolution      ##
	## ****                   **** ##

	SSE_obj = SSE_lab(E_J,max_wells,well_vecs,Us_window,dUs_window,L1s,L2s,delta_t,num_tpoints)
	
	rng = default_rng()

	psi = {}
	width = 14.134517595668662*pi*rng.random()#7*pi
	_norm = sqrt(sum([exp(-(2*pi*w)**2/width**2) for w in range(-max_wells,max_wells+1,2)]))
	i = 0
	for w in range(-max_wells,max_wells+1,2):
		psi[w] = zeros(len(well_energies[well_indices[i]]))
		if w >= 0:
			psi[w] = exp(-(2*pi*w)**2/(2*width**2))*(rng.normal(size=H_eff_herm[w].shape[0]) + 1j*rng.normal(size=H_eff_herm[w].shape[0]))
			# psi[w][0] = -exp(-(2*pi*w)**2/(2*width**2))/(sqrt(3)*_norm)
			# psi[w][1] = -exp(-(2*pi*w)**2/(2*width**2))/(sqrt(3)*_norm)
			# psi[w][2] = -exp(-(2*pi*w)**2/(2*width**2))/(sqrt(3)*_norm)
		else:
			psi[w] = exp(-(2*pi*w)**2/(2*width**2))*(rng.normal(size=H_eff_herm[w].shape[0]) + 1j*rng.normal(size=H_eff_herm[w].shape[0]))
			# psi[w][0] = exp(-(2*pi*w)**2/(2*width**2))/(sqrt(3)*_norm)
			# psi[w][1] = exp(-(2*pi*w)**2/(2*width**2))/(sqrt(3)*_norm)
			# psi[w][2] = exp(-(2*pi*w)**2/(2*width**2))/(sqrt(3)*_norm)

	#_norm = sqrt(sum([norm(vec)**2 for vec in psi.values()]))
	#psi = dict([(w,vec/_norm) for w,vec in psi.items()])


	SSE_params = [(psi,s,num_periods) for s in range(num_samples)]
	psi_mids , psi_ends = [] , []
	psi_pre_jumps, psi_post_jumps = [] , []
	jumps = []
	with Pool(processes=num_threads) as p2:
		for psi_mid, psi_end,jump_times,pre_jumps,post_jumps in p2.starmap(SSE_obj.SSE_evolution,SSE_params,chunksize=int(ceil(num_samples/num_threads))):
			psi_mids.append(psi_mid)
			psi_ends.append(psi_end)
			jumps.append(jump_times)
			psi_pre_jumps.append(pre_jumps)
			psi_post_jumps.append(post_jumps)


	#Dynamics within a single period
	# from SSE_lab import apply_diagonal_operator
	# psis = []
	# for i,t in enumerate(t_vals_window):
	# 	psis.append(apply_diagonal_operator(psi,Us_window[i]))

	# out_arr = array([parameters,psis,t_vals_window])
	out_arr = array([parameters,psi_mids,psi_ends,jumps,psi_pre_jumps,psi_post_jumps,psi],dtype=object)
	save(outfile,out_arr)