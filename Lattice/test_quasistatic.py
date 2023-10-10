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
			if i < 2*max_wells:
				d_plus = len(well_energies[index+2])
				g_minus = zeros((d_plus,d))
				for m in range(d_plus):
					for n in range(d):
						g_mat[m,n] = g(well_energies[index][n]-well_energies[index+2][m],Temp,Lambda)

				L1s_window2[well_nums[index]] = 2*pi*sqrt(gamma)*g_mat*overlaps[index]
