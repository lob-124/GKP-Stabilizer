#Add directory with Frederik's code to path
from sys import path
path.insert(0,"../Frederik/")

from gaussian_bath import bath,get_J_ohmic
from basic import get_tmat

#import jump_static as js
import jump_static_optimized as js
from integrate import time_evolution, find_timestep
from params import *

from numpy.linalg import eigh,eigvals,norm
from scipy.linalg import expm
from numpy import linspace,diag,cos,exp,around,log2
from numpy.random import rand

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from time import perf_counter

if __name__ == "__main__":
	from sys import argv

	if len(argv) != 2:
		print("Usage: <outfile>")
		exit(0)

	outfile = argv[1]

## Shamelessly stolen from Frederik
	# Derive physical quantities from parameters
	impedance   = planck_constant/(2*quantization_parameter*e_charge**2)
	L           = impedance/cavity_frequency
	C           = 1/(cavity_frequency*impedance)
	inductance  = 1*L
	resistance  = hbar*omega0/(2*pi*e_charge**2*gamma)

	# Grid of discretized phi values 
	phi_cutoff  = N_wells*2*pi
	phi         = linspace(-phi_cutoff,phi_cutoff,D)
	dphi        = phi[1]-phi[0]

	well_projector = 1-(-1)**(phi/(2*pi)//1)-1 #What's with the extra -1??


	## OPERATORS ##
	# Dimensionless phase \phi 
	Phi     = diag(phi)
	X1 = expm(1j*Phi)
	X2 = expm(-1j*Phi)

	# Canonically conjugate momentum to Phi (q)
	Tmat    = get_tmat(D,dtype=complex)
	Pi0     = 1j*(Tmat-eye(D))/dphi

	# Squared momentum (note the symmetrization of the operator)
	Pi_2 = 0.5*(Pi0@Pi0.conj().T + Pi0.conj().T@Pi0)

	# Squared charge operator 
	Q2 = ((2*e_charge)/hbar)**2 * Pi_2 

	# Circuit Hamiltonian (w/o Josephson junction)
	H0pot = Phi**2/(2*L)*(hbar**2/(4*e_charge))
	H0kin = Q2/(2*C)

	H0 = H0pot + H0kin

	# Josephson potential 
	V  = Josephson_energy * diag(cos(phi))
	H1 = H0 + V

	#The Hamiltonian during the window the resistor is on, and outside of it
	#Can swap these to create different protocols
	H_window_full = H1
	H_outside_full = H0

	#Diagonalize H_window (needed for computing jump operators)
	E_window_full, V_window_full = eigh(H_window_full)

	#Truncate the Hilbert space to keep only the N_trunc lowest lying
	#	energy levels
	E_window , V_window = E_window_full[:N_trunc] , V_window_full[:,:N_trunc] 

	#H_window and H_outside in the truncated Hilbert space
	H_window = diag(E_window)
	H_outside = V_window.conj().T @ H_outside_full @ V_window


	## BATH ##

	# Spectral function
	J = get_J_ohmic(Temp,omega_c,omega0=omega0)

	# Frequency resolution  (only used to  calculating bath timescales in ULE module).
	wvec = linspace(-10*2*pi/dt_JJ,10*2*pi/dt_JJ,10000)
	dw = wvec[1]-wvec[0]

	# Bath object
	B = bath(J,10*omega_c,10*omega_c/4000)

#Now onto original(-ish) stuff by me

	#Fourier components for the resistor coupling W(t)
	#W_fourier = [1j*(1-exp(1j*Omega*q*dt_JJ))/(2*pi*q) for q in range(-q_max,q_max+1)]
	W_fourier = [sqrt(pi)*tau*exp(-Omega*q*(4j*t_0+q*Omega*tau**2)/4) for q in range(-q_max,q_max+1)]
	frequencies = [Omega*q for q in range(-q_max,q_max+1)]




	## TIME EVOLUTION OPERATORS ##

	#Full Hamiltonian over one period:
	#      H = H_window 0 <= t < \Delta t
	#      H = H_outside \Delta t < t < T

	#Evolution during the window resistor is on (ie, 0 <= t1,t2 < \Delta t (mod T)) 
	def time_evo_window(t1,t2,energy_basis=True):
		if energy_basis:	#return in basis of H0 eigenstates
			return diag(exp(-1j*(t2-t1)*E_window/hbar))
		else:	#return in phi basis
			return expm(-1j*(t2-t1)*H_window_full/hbar)

	#Evolution while resistor is off (ie, \Delta t < t1,t2 < T (mod T))
	def time_evo_outside(t1,t2,energy_basis=True):
		#_U = expm(-1j*(t2-t1)*H1/hbar)
		if energy_basis:	#return in basis of H0 eigenstates
			return expm(-1j*(t2-t1)*H_outside/hbar)
		else:	#return in phi basis
			return expm(-1j*(t2-t1)*H_outside_full/hbar)

	#Makes use of the two above functions to compute a general time evolution for the step protocol
	#NB: Assumes -T/2 < t1 < t2 < T/2. Time evolution with t2 < t1 can be found by hermitian conjugation
	def time_evo(t1,t2,delta_t):
		if t2 < t1:
			print("Error: expecting t2 > t1")
			exit(1)
		
		if t1 < 0:
			#If t1 < 0, there are three cases:
			#	1) t2 <= 0 as well, in which case all the evolution is generated by H_outside
			#	2) 0 < t2 <= delta_t, in which case we evolve by H_outside from t1 to 0, and by 
			#		H_window from 0 to t2
			#	3) t2 > delta_t, in which case we evolve by H_outside from t1 to 0, then H_window 
			#		from 0 to delta_t, then H_outside again from delta_t to t2
			if t2 <= 0:
				return time_evo_outside(t1,t2)
			else:
				if t2 <= delta_t:
					return time_evo_window(0,t2) @ time_evo_outside(t1,0)
				else:
					return time_evo_outside(delta_t,t2) @ time_evo_window(0,delta_t) @ time_evo_outside(t1,0)

		elif t1 < delta_t:
			#If 0 < t1 < delta_t, there are two cases:
			#	1) t2 <= delta_t as well, in which we simply evolve by H_window from t1 to t2
			#	2) t2 > delta_t, in which case we evolve by H_window from t1 to delta_t, then 
			#		H_outside from delta_t to t2
			if t2 <= delta_t:
				return time_evo_window(t1,t2)
			else:
				return time_evo_outside(delta_t,t2) @ time_evo_window(t1,delta_t)

		else:
			#If t1 > delta_t, then the evolution is generated only by H_outside
			return time_evo_outside(t1,t2)


	#Projects an arbitrary time t into the one period interval [-T/2 , T/2]
	def mod(t):
		_temp = t % T
		if _temp > T/2:
			return _temp - T
		else: 
			return _temp 


	## EFFECTIVE HAMILTONIAN FOR SSE EVOLUTION ##

	#The system Hamiltonian (as a function of time)
	def H_t(t,trunc=True):
		if trunc:
			if (t % T) <= dt_JJ:
				return H_window
			else: 
				return H_outside
		else:
			if (t % T) <= dt_JJ:
				return H_window_full
			else: 
				return H_outside_full

	#Transform the physical operators to the energy basis (eigenstates of H_window)
	X1_E = V_window.conj().T @ X1 @ V_window
	X2_E = V_window.conj().T @ X2 @ V_window

	#The jump operators at time t
	def jump_ops(t):
		t_proj = mod(t)	#Project t into the interval [-T/2,T/2]

		#Compute the time evolution U(delta_t,t), and its conjugate, appearing in the jump
		#	operators. Note we have to handle separately the case t < delta_t because of how 
		#	time_evo() above is defined 
		if t_proj > dt_JJ:
			U = time_evo(dt_JJ,t_proj,dt_JJ)
			U_dag = U.conj().T
		else:
			U_dag = time_evo(t_proj,dt_JJ,dt_JJ)
			U = U_dag.conj().T

		L_1 = U @ js.L_tilde_energy_basis(X1_E,t_proj,E_window,(W_fourier,frequencies),dt_JJ,gamma,Temp,omega_c,omega0) @ U_dag
		L_2 = U @ js.L_tilde_energy_basis(X2_E,t_proj,E_window,(W_fourier,frequencies),dt_JJ,gamma,Temp,omega_c,omega0) @ U_dag

		return L_1,L_2


	#Compute the effective Hamiltonian governing SSE Evolution
	def H_eff(t,return_jump_ops=False):
		L_1 , L_2 = jump_ops(t)
		_H = H_t(t) - (1j*hbar/2)*(L_1.conj().T @ L_1 + L_2.conj().T @ L_2)

		if return_jump_ops:		#Option to return jump operators as well
			return _H , L_1, L_2
		else: 
			return _H



	# ## SSE EVOLUTION ##

	#Initial wavefunction 
	psi_0 = zeros(N_trunc)
	psi_0[0] = 1/sqrt(2)
	psi_0[1] = 1/sqrt(2) 

	#time values within each driving period
	#t_vals = arange(0,T+dt/2,dt)
	t_vals = linspace(0,T,num=N_steps+1)
	dt = t_vals[1] - t_vals[0]
	
	#Compute the time evolution operators U_eff(0,t) & U_eff(t,T) for each t in t_vals for evolution
	#	according to H_eff
	#We'll use U_eff(0,T) to compute the evolution over each driving period, and
	#	all the other U(0,t) to do binary search within the period when a jump occurrs.
	#After finding the jump location, we use U(t,T) to evolve to the end of the period and check for another jump
	U_current = eye(N_trunc,dtype=complex128)
	dU = {}
	U_0_to_t = {0:U_current}	#Include the identity at t=0 (useful later)
	U_t_to_T = {N_steps:U_current}	#Store also the evolution U_eff(t,T)
	L_ops = {}

	max_exponent = 0
	#t1 = perf_counter()
	for i in range(0,N_steps):
		print("i = {}".format(i))
		_H , _L1, _L2 = H_eff(t_vals[i],return_jump_ops=True)

		#Use H_eff(t) to compute U_eff(t,t+dt) & concatenate it to U_eff(0,t)
		dU[i] = time_evolution(dt,_H,order=order)
		U_current = dU[i] @ U_current
		U_0_to_t[i+1] = U_current
			
		L_ops[i] = (_L1 , _L2)	#Store the jump operators as well!

		
	#Now construct the evolution operators U_eff(t,T)
	indices = list(U_0_to_t.keys())
	indices.reverse()
	#for i in range(1,len(indices)):  
	for i in range(1,N_steps):
		print("i = {}".format(i))
		U_t_to_T[indices[i]] = U_t_to_T[indices[i-1]] @ dU[indices[i]]


	L_ops[N_steps] = L_ops[0]	#Store the jump operators at t=T (useful later)
	U_T = U_0_to_t[N_steps]		#The evolution operator over a full period

	# t2 = perf_counter()
	# print("Time taken: {} seconds".format(t2-t1))
	# exit(0)



	# jumps = []
	# well_imbalance = []
	# # other_paulis = []	#figure out from Frederik what this is
	# # #rhos = []
	# weights = []
	# psis = []
	# # #wigners = []
	# for sample_num in range(nsamples):
	# 	#Keep track of jumps & imbalance for this sample
	# 	jumps_this_sample = []
	# 	imbalance_this_sample = []
	# 	other_paulis_this_sample = []
	# 	weights_this_sample = []

	# 	r = rand()	#Random number to compare norm of wavefunction to
	# 	#print("Starting r: {}".format(r))

	# 	psi = psi_0		#The current wavefunction, initialized to psi_0
	# 	times = []
		
	# 	for i in range(num_periods):
						
	# 		#Indices used by binary search within period
	# 		start_index = 0			#time of previous jump
	# 		ind_left = 0			#left endpoint of current search interval
	# 		ind_right = N_steps 	#right endpoint of current search interval

	# 		psi_new = U_T @ psi 	#Evolve by a full period 

	# 		#Check if there was a quantum jump within this period
	# 		while (1 - norm(psi_new)**2) > r:
	# 			#print("-------- JUMP DETECTED --------")

	# 			#Binary search for the time at which the jump occurred
	# 			while (ind_right - ind_left) > 1:
	# 				ind_mid = round((ind_right+ind_left)/2)
					

	# 				if start_index != 0:
	# 					#If we've had a previous jump within this period, we evolve from
	# 					#	the time of the previous jump - NOT the start of the period!
	# 					#NB: "psi" is the wavefunction at the previous jump time (see ****)
	# 					psi_temp=psi
	# 					for index in range(start_index,ind_mid):
	# 						psi_temp = dU[index] @ psi_temp
	# 				else:
	# 					psi_temp = U_0_to_t[ind_mid] @ psi

	# 				if (1-norm(psi_temp)**2) >= r:
	# 					ind_right = ind_mid
	# 				else:
	# 					ind_left = ind_mid

	# 			#We've now found the time at which the jump ocurred
	# 			ind = round((ind_right+ind_left)/2)
	# 			#print("Jump index: {} of {}".format(ind,N_steps))
	# 			jump_time = i*T + dt*ind

	# 			#Advance psi to time t (****)
	# 			if start_index != 0:
	# 				#Evolve from prior jump (if there was one)
	# 				for index in range(start_index,ind):
	# 					psi = dU[index] @ psi
	# 			else:
	# 				#Otherwise, evolve from beginning of period
	# 				psi = U_0_to_t[ind] @ psi 	

	# 			#Jump operators at time t
	# 			L_1 , L_2 = L_ops[ind]

	# 			#Determine the type of jump, and jump the wavefunction accordingly
	# 			p_1 = norm(L_1 @ psi)**2
	# 			p_2 = norm(L_2 @ psi)**2
	# 			if rand() < p_1/(p_1+p_2):
	# 				jumps_this_sample.append((1,jump_time))
	# 				psi = L_1 @ psi
	# 			else:
	# 				jumps_this_sample.append((2,jump_time))
	# 				psi = L_2 @ psi

	# 			#In either case, re-normalize the wavefunction and re-set the random variable r
	# 			#print("Norm before normalization: {}".format(norm(psi)))
	# 			psi = psi/norm(psi)
	# 			#print("Norm after normalization: {}".format(norm(psi)))
	# 			r = rand()
	# 			#print("\t"+"New r : {}".format(r))
			
	# 			#Evolve to the end of the period (NOW from the jump time!!), reset the binary search
	# 			#	markers, and rinse/repeat (if necessary)
	# 			psi_new = U_t_to_T[ind] @ psi 	
	# 			#print("Norm after evolution to end of period: {}".format(norm(psi_new)))
	# 			start_index  = ind 	#Update start_index to mark the time of this jump
	# 			ind_left = ind 
	# 			ind_right = N_steps

	# 		#end jump while block

	# 		#We've reached the end of this driving period. Update "psi" accordingly
	# 		psi = psi_new
	# 		weights_this_sample.append(abs(psi)**2/norm(psi)**2)
			
	# # 		#t_end = perf_counter()
	# # 		#print("Time elapsed: {}".format(t_end-t_begin)) 

	# 		#Measure once only every oscillation period of the bare LC circuit 
	# 		if ((i+1) % 4) == 0:
	# 		 	psi_full = V_window @ psi 	#Convert back to phi basis
	# 		 	imbalance_this_sample.append(sum(abs(psi_full)**2*well_projector)/norm(psi_full)**2)
	# 		 	#other_paulis_this_sample.append(sum(psi_full @ psi_full[::-1].conj())/norm(psi_full)**2)
		
	# 	#end for loop over cycles
	# 	# t2 = perf_counter()
	# 	# print("Time elapsed: {} seconds".format(t2-t1))
	# 	#weights.append(weights_this_sample)

	#  	#end for loop over cycles
	# 	jumps.append(jumps_this_sample)
	# 	psis.append(psi)
	# 	weights.append(weights_this_sample)
	# 	well_imbalance.append(imbalance_this_sample)

	# save(outfile,[jumps,psis,weights,well_imbalance])