#Add directory with Frederik's code to path
from sys import path
path.append("../Frederik/")

from units import *

from numpy import exp,sign,sqrt,pi,load,save,linspace,zeros,complex64,array
from numpy.random import default_rng
from numpy.linalg import norm
from math import factorial

from time_evo import find_evolutions, time_evolve

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

	if len(argv) != 16:
		print("Usage: dt_1 dt_2 T_LC E_J gamma T Lambda num_threads num_samples periods max_wells num_binary_window_1 <infile_wells> <infile_overlaps> <outfile>")
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
	infile_wells = argv[13]
	infile_overlaps = argv[14]
	outfile = argv[15]


	parameters = {"dt_1":dt_1 , "dt_2":dt_2, "E_J":E_J, "gamma":gamma, "Temp":Temp, "Lambda":Lambda}


	#The full driving period
	T = dt_1 + dt_2 + T_LC/2


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
		if i < num_wells-2:
			overlaps_window2[well_nums[i]] = overlaps[i]
		
	#Compute the jump operators in first window
	L1s_window1 , L2s_window1 , = {} , {}
	for i in well_indices:
		d = len(well_energies[i])
		g_mat = zeros((d,d))
		for m in range(d):
			for n in range(d):
				g_mat[m,n] = g(well_energies[i][n]-well_energies[i][m],Temp,Lambda)

		L1s_window1[well_nums[i]] = 2*pi*sqrt(gamma)*g_mat*X1s[i]
		L2s_window1[well_nums[i]] = 2*pi*sqrt(gamma)*g_mat*X2s[i]
		

	#### Compute the time evolution operators in the first window in each well
	#	-> Note there is no interwell coupling in this step
	H_eff_herm , H_eff_NH = {} , {}
	for i in well_indices:
		well_num = well_nums[i]
		H_eff_herm[well_num] = diag(well_energies[i])
		H_eff_NH[well_num] =  -0.5j*(L1s_window1[well_num].conj().T @ L1s_window1[well_num] + L2s_window1[well_num].conj().T @ L2s_window1[well_num])

	# #The effective Hamiltonian at time t
	# def H_eff(t,well_num):
	# 	return H_eff_herm[well_num] + bump_function(t/dt_1)*H_eff_NH[well_num]

	#Parameters (for parallelization)
	num_tpoints1 = 2**num_binary_window1
	t_vals_window1 = linspace(0,dt_1,num=num_tpoints1+1)
	dt = t_vals_window1[1] - t_vals_window1[0]
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
				print(abs(dUs[j]).max(),end=" ")
				Us_window1[i+1][well_num] = Us_window1[i][well_num] @ dUs[j]
				print(abs(Us_window1[i+1][well_num]).max())



	## ****                   **** ##
	##	    The SSE evolution      ##
	## ****                   **** ##

	#We represent the wavefunctions as a dictionary, with psi[k] being a vector
	#	of the components of psi in the basis of eigenfunctions of well k
	#More precisely:
	#			psi[k][i] = <k,i|psi>
	#	where |k,i> is the ith level in well k.
	#We also only keep track of the wells where psi has support, so psi.items() gives
	#	a list of the well indices where components of psi are non-zero 
	psi = {}
	psi[0] = zeros(len(well_energies[zero_well_ind]),dtype=complex64)
	psi[0][0] = 1.0
	rng = default_rng()

	#Function to apply operators to psi (where the operator is assumed to act only WITHIN each well)
	def apply_operator(psi,O):
		"""
		Given a wavefunction and operator O, represented in our grid structure, apply the 
			unitary to psi. We assume U does not couple between wells
		"""
		return dict(map(lambda tup: (tup[0],O[tup[0]] @ tup[1]), psi.items()))

	#Function to compute the norm squared of psi
	def my_norm(psi):
		return sum(list(map(lambda v: norm(v)**2,psi.values())))
	
	#Function to compute the evolution (t -> t+dt) during the second window
	def evolution_window_2(psi,dt,order=5):
		if order < 1:
			print("Error: order must be at least one.")
			exit(1)

		psi_tot = dict(psi)
		dpsi_prev = dict(psi)
		for i in range(1,order+1):
			#Apply H_LC to the previous order's correction
			dpsi_curr = apply_operator(dpsi_prev,H_LC_window2)

			#Apply H_JJ (<-> well translation) to the previous order's correction
			for well_num, v in dpsi_prev.items():
				#print("On well {}".format(well_num))
				if well_num > -max_wells+1:
					#Translation backward by two wells
					dpsi_curr[well_num-2] = dpsi_curr.get(well_num-2,0.0+0j) - (E_J/2)*(overlaps_window2[well_num-2].conj().T @ v)
				if well_num < max_wells-1:
					#Translation forward by two wells
					dpsi_curr[well_num+2] = dpsi_curr.get(well_num+2,0.0+0j) - (E_J/2)*(overlaps_window2[well_num] @ v)
	
			for well_num,dv in dpsi_curr.items():
				psi_tot[well_num] = psi_tot.get(well_num, 0.0+0j) + ((-1j*dt/hbar)**i/factorial(i))*dv

			dpsi_prev = dpsi_curr

		return psi_tot
	

	r = rng.random()
	psis = []
	for i in range(num_periods):
		if i % 50 == 0:
			print("On period {} of {}".format(i,num_periods))

		#Indices used by binary search within the window
		start_index = 0
		ind_left = 0
		ind_right = num_tpoints1

		#t1 = perf_counter()

		#Perform the evolution over the first window
		psi_new = apply_operator(psi,Us_window1[-1])

		#t2 = perf_counter()
		#print("Time to apply evo operator: {}".format(t2-t1))
		
		#Check if there was a quantum jump within this window
		while (1 - my_norm(psi_new)) > r:
			#Binary search for the time at which the jump occurred
			while (ind_right - ind_left) > 1:
				ind_mid = round((ind_right+ind_left)/2)

				#Evolve the wavefunction to the time given by ind_mid
				if start_index != 0:
					#If we've already had a jump thios period, evolve from the time
					#	of the last jump
					#NB: psi is the wavefunction at the previous jump time (see ****)
					psi_temp = psi
					for index in range(start_index,ind_mid):
						psi_temp = apply_operator(psi_temp,dUs_window1[index])
				else:
					#Otherwise, evolve from the beginning of the period
					psi_temp = apply_operator(psi,Us_window1[ind_mid])

				#Check if the jump has happened yet
				if (1 - my_norm(psi_temp)) >= r:
					ind_right = ind_mid
				else:
					ind_left = ind_mid

			#end binary search while loop

			#We've now found the time t at which the jump occurred
			ind = ind_right
			jump_time = i*T + t_vals_window1[ind]

			#Advance the wavefunction to time t (****)
			if start_index != 0:
				#Evolve from prior jump (if needed)
				for index in range(start_index,ind):
					psi = apply_operator(psi,dUs_window1[index])
			else:
				#Otherwise, evolve from beginning of period
				psi = apply_operator(psi,Us_window1[ind])

			#Determine the type of jump, and jump the wavefunction
			_psi1 = apply_operator(psi,L1s_window1)
			_psi2 = apply_operator(psi,L2s_window1)
			p_1 , p_2 = my_norm(_psi1) , my_norm(_psi2)
			if rng.random() < p_1/(p_1+p_2):
				psi = dict([(well,vec/sqrt(p_1)) for well,vec in _psi1.items()])
			else:
				psi = dict([(well,vec/sqrt(p_2)) for well,vec in _psi2.items()])

			#Reset the random variable r
			r = rng.random()

			#Evolve psi to the end of the period, and reset the binary search markers
			for index in range(ind,num_tpoints1):
				psi_new = apply_operator(psi,dUs_window1[index])
			ind_left = ind
			ind_right = num_tpoints1

		#end jump while block


		#Update psi, now that we are done with this window driving period
		psi = dict(psi_new)

		#print(my_norm(psi))
		#lol = input("Press enter to proceed")

		#t1 = perf_counter()
		#Now we evolve over the second window. Note the resistor is not on here, so this is 
		#	unitary evolution
		_t = 0
		while _t + dt_window2 < dt_2:
		#	print("On time t/T_2 = {}".format(_t/dt_2))
			psi = evolution_window_2(psi,dt_window2)
			_t += dt_window2

		psi = evolution_window_2(psi,dt_2-_t)


		#t2 = perf_counter()
		#print("Time to apply 2nd JJ evolution: {}".format(t2-t1))


		#Store the wavefunction at the end of this driving period
		psis.append(psi)


	#end loop over periods


	save(outfile,[parameters,psis])