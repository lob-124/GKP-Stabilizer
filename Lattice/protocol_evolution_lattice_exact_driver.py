#Add directory with Frederik's code to path
from sys import path
path.append("../Frederik/")

from units import *

from numpy import exp,sign,sqrt,pi,load,save,linspace,zeros,complex64,array
from numpy.random import default_rng
from numpy.linalg import norm
from math import factorial

from time_evo import find_evolutions, time_evolve
from SSE import SSE,SSE2

from multiprocessing import Pool
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
		

if __name__ == "__main__":
	from sys import argv
	#from time import perf_counter

	print(len(argv))
	if len(argv) != 17:
		print("Usage: dt_1 dt_2 T_LC E_J gamma T Lambda num_threads num_samples periods max_wells num_binary_window_1 resistor_window_2 <infile_wells> <infile_overlaps> <outfile>")
		print("Units: \n -> All times should be given in picoseconds\n -> E_J in GHz*hbar \n -> gamma in meV\n -> T in Kelvin\n -> Lambda in GHz")
		exit(0)

	#Command line arguments
	dt_1 = float(argv[1])*picosecond
	dt_2 = float(argv[2])*picosecond
	T_LC = float(argv[3])*picosecond
	E_J = float(argv[4])*GHz*hbar
	gamma = float(argv[5])*meV
	Temp = float(argv[6])*Kelvin
	Lambda = float(argv[7])*GHz
	num_threads = int(argv[8])
	num_samples = int(argv[9])
	num_periods = int(argv[10])
	max_wells = int(argv[11])
	num_binary_window1 = int(argv[12]) #Number of times in the first window to sample for binary search
	resistor_window_2 = int(argv[13])	#Whether to have the resistor connected in the second window
	infile_wells = argv[14]
	infile_overlaps = argv[15]
	outfile = argv[16]


	parameters = {"dt_1":dt_1 , "dt_2":dt_2, "E_J":E_J, "gamma":gamma, "Temp":Temp, "Lambda":Lambda}


	#The full driving period
	T = dt_1 + dt_2 + T_LC/2

	print("Let's do it!")

	#Load in the data about the lattice of well eigenstates
	well_nums , well_energies, rms_vals, H_LCs, X1s, X2s = load(infile_wells,allow_pickle=True)
	_ , overlaps = load(infile_overlaps,allow_pickle=True)  
	num_wells  = len(well_nums)
	zero_well_ind = (num_wells-1)//2

	#Keep wells only up to a maximal well index (max_wells from command line)
	if max_wells >= well_nums[-1]:
		well_indices = list(range(num_wells))
		well_numbers = well_nums
	else:
		well_indices = list(range(zero_well_ind - max_wells,zero_well_ind+max_wells+1))
		well_numbers = list(range(-max_wells,max_wells+1))


	dt_window2 = hbar/(10*E_J)	#Time step for integration in the second window


	#Put the LC Hamiltonian and the overlap matrices into dictionaries	
	H_LC_window2 , overlaps_window2 = {} , {}
	for i in well_indices:
		H_LC_window2[well_nums[i]] = H_LCs[i]
		

	print("Doing first window jump ops... ",end="")
	#Compute the jump operators in first window
	L1s_window1 , L2s_window1 , = {} , {}
	num_tpoints1 = 2**num_binary_window1
	t_vals_window1 = linspace(0,dt_1,num=num_tpoints1+1)
	dt = t_vals_window1[1] - t_vals_window1[0]
	for i in well_indices:
		d = len(well_energies[i])
		g_mat = zeros((d,d))
		for m in range(d):
			for n in range(d):
				g_mat[m,n] = g(well_energies[i][n]-well_energies[i][m],Temp,Lambda)

		L1s_window1[well_nums[i]] = 2*pi*sqrt(gamma)*g_mat*X1s[i]
		L2s_window1[well_nums[i]] = 2*pi*sqrt(gamma)*g_mat*X2s[i]
	
	print("Done")

	print("Now doing time evo operators... ",end="")
	#### Compute the time evolution operators in the first window in each well
	#	NB: there is no interwell coupling in this step!
	H_eff_herm , H_eff_NH = {} , {}
	for i in well_indices:
		well_num = well_nums[i]
		H_eff_herm[well_num] = diag(well_energies[i])
		H_eff_NH[well_num] =  -0.5j*(L1s_window1[well_num].conj().T @ L1s_window1[well_num] + L2s_window1[well_num].conj().T @ L2s_window1[well_num])
		

	#Parameters (for parallelization)
	params = []
	for t in t_vals_window1[:-1]:
		params.append((t,dt,H_eff_herm,H_eff_NH,bump_function,well_numbers,dt_1))

	dUs_window1 = [{} for i in range(num_tpoints1)]	
	Us_window1 = [{} for i in range(num_tpoints1+1)]
	for i in well_indices:
		Us_window1[0][well_nums[i]] = eye(len(well_energies[i]))

	# #Compute the evolutions U(t+dt) in parallel for different ts, and then stitch together the results
	with Pool(processes=num_threads) as p:
		for i, dUs in enumerate(p.starmap(find_evolutions,params,chunksize=int(ceil(num_tpoints1/num_threads)))):
			for j,well_num in enumerate(well_numbers):
				dUs_window1[i][well_num] = dUs[j]
				Us_window1[i+1][well_num] = Us_window1[i][well_num] @ dUs[j]
				
	print("Done!")

	print("Now doing second window jump ops... ",end="")
	#Compute the jump operators in second window
	if resistor_window_2:
		L1s_window2 , L2s_window2 , = {} , {}
		for i,index in enumerate(well_indices):
			d = len(well_energies[index])

			#Compute L2 (derived from translation BACKWARDS by 2 wells)
			if i > 1:
				d_minus = len(well_energies[index-2])
				g_minus = zeros((d_minus,d))
				for m in range(d_minus):
					for n in range(d):
						g_mat[m,n] = g(well_energies[index][n]-well_energies[index-2][m],Temp,Lambda)

				L2s_window2[well_nums[index]] = 2*pi*sqrt(gamma)*g_mat*(overlaps[index-2].conj().T)
				
			#Compute L1 (derived from translation FORWARDS by 2 wells)
			if i < 2*max_wells-1:
				d_plus = len(well_energies[index+2])
				g_minus = zeros((d_plus,d))
				for m in range(d_plus):
					for n in range(d):
						g_mat[m,n] = g(well_energies[index][n]-well_energies[index+2][m],Temp,Lambda)

				L1s_window2[well_nums[index]] = 2*pi*sqrt(gamma)*g_mat*overlaps[index]

				# print("--- Well Number {} ----".format(well_nums[index]))
				# well_number = well_nums[index]
				# if well_number <= -max_wells+1:
				# 	print("Max matrix element: {}".format(abs(L1s_window2[well_nums[index]]).max()))
				# elif well_number >= max_wells-1:
				# 	print("Max matrix element: {}".format(abs(L2s_window2[well_nums[index]]).max()))
				# else:
				# 	print("Max matrix elements: {} , {}".format(abs(L1s_window2[well_nums[index]]).max(),abs(L2s_window2[well_nums[index]]).max()))

	print("Done")



	## ****                   **** ##
	##	    The SSE evolution      ##
	## ****                   **** ##

	if resistor_window_2:
		SSE_obj = SSE2(E_J,num_periods,max_wells,Us_window1,dUs_window1,L1s_window1,L2s_window1,L1s_window2,L2s_window2,H_LC_window2,overlaps_window2,num_tpoints1,dt_2,dt_window2)
	else:
		SSE_obj = SSE(E_J,num_periods,max_wells,Us_window1,dUs_window1,L1s_window1,L2s_window1,H_LC_window2,overlaps_window2,num_tpoints1,dt_2,dt_window2)
	psi = {}
	psi[0] = zeros(len(well_energies[zero_well_ind]),dtype=complex64)
	psi[0][0] = 1.0

	SSE_params = [(psi,s) for s in range(num_samples)]
	psi_mids , psi_ends = [] , []
	with Pool(processes=num_threads) as p2:
		for psi_mid, psi_end in p2.starmap(SSE_obj.SSE_evolution,SSE_params,chunksize=int(ceil(num_samples/num_threads))):
			psi_mids.append(psi_mid)
			psi_ends.append(psi_end)


	out_arr = array([parameters,psi_mids,psi_ends],dtype=object)
	save(outfile,out_arr)