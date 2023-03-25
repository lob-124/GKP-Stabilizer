#Add directory with Frederik's code to path
from sys import path
path.insert(0,"../Frederik/")

from basic import get_tmat

#import jump_static as js
import jump_ops as jo
from integrate import time_evolution
from params import *

from numpy.linalg import eigh,eigvals,norm
from numpy import linspace,diag,cos,exp,around,log2
from numpy.random import rand

from scipy.linalg import schur

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
	X_1 = expm(1j*Phi)
	X_2 = expm(-1j*Phi)

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
	

#Now onto original(-ish) stuff by me

	#Fourier components for the resistor coupling W(t)
	W_fourier = [sqrt(pi)*tau*exp(-Omega*q*(4j*t_0+q*Omega*tau**2)/4) for q in range(-q_max,q_max+1)]
	frequencies = [Omega*q for q in range(-q_max,q_max+1)]
	W_ft = (W_fourier,frequencies)


	#The system Hamiltonian (as a function of time)
	#The time-dependent JJ coupling V_t(t) should be defined in params.py
	def H_t(t,trunc=True):
		return H0 + V_t(t)*V


## BIG STEP: Compute the Floquet states |\phi_k(t)> ##
	# First, compute evolution over a full period
	U_T = eye(D)
	dt_unitary = T/N_steps_unitary
	for i in range(N_steps_unitary):
		_H = H_t(dt_unitary*i)
		for j in range(D):
			U_T[:,j] = time_evolve(U_T[:,j],_H,dt_unitary,order=order)

	#Diagonalize this operator to obtain the stationary states and quasienergies
	_D, _U = schur(U_T) 
	quasi_energies = 1j*log(diag(_D))/T

	H_floq = diag(quasi_energies)	#Hamiltonian in the basis of Floquet modes

	#Now we compute the Floquet modes by evolving and factoring out the "dynamical" 
	#	phases from the quasienergies
	floq_states = {0:_U}
	curr = _U
	for i in range(N_steps_unitary):
		_H = H_t(dt_unitary*i)
		for j in range(D):
			curr[:,j] = time_evolve(curr[:,j],_H,dt_unitary,order=order)
			curr[:,j] = exp(1j*quasi_energies[j]*dt_unitary)*curr[:,j]

		floq_states[i] = curr


	#Now we compute the Fourier components of <\phi_j(t)|X|\phi_k(t)>
	X_ft_1 = []
	X_ft_2 = []
	for n in range(-n_max,n_max+1):
		X_ft_1_this_n = zeros((D,D))
		X_ft_2_this_n = zeros((D,D))
		for i in range(D):
			for j in range(D):
				sum_1 = 0.0
				sum_2 = 0.0
				for k in range(N_steps_unitary):
					phi_i , phi_j = floq_states[k][:,i] , floq_states[k][:,j]
					sum_1 += phi_i.conj() @ X_1 @ phi_j
					sum_2 += phi_i.conj() @ X_2 @ phi_j

				X_ft_1_this_n[i,j] = sum_1*dt/T
				X_ft_2_this_n[i,j] = sum_2*dt/T

		X_ft_1.append(X_ft_1_this_n)
		X_ft_2.append(X_ft_2_this_n)
	#end loop over n

	X_ft_1 = array(X_ft_1)
	X_ft_2 = array(X_ft_2)

	

	#The jump operators at time t
	def jump_ops(t):
		#Compute the jump operators in the basis of Floquet modes 
		L_1 = jo.L(t,X_ft_1,None,quasi_energies,W_ft,gamma,temp,Lambda,omega_0=1)
		L_2 = jo.L(t,X_ft_2,None,quasi_energies,W_ft,gamma,temp,Lambda,omega_0=1)

		return L_1,L_2


	#Compute the effective Hamiltonian governing SSE Evolution (in the basis of Floquet modes)
	def H_eff(t,return_jump_ops=False):
		L_1 , L_2 = jump_ops(t)
		_H = H_floq - (1j*hbar/2)*(L_1.conj().T @ L_1 + L_2.conj().T @ L_2)

		if return_jump_ops:		#Option to return jump operators as well
			return _H , L_1, L_2
		else: 
			return _H



	# ## SSE EVOLUTION ##

	#Initial wavefunction (in phi basis)
	psi_0 = zeros(N_trunc)
	psi_0[0] = 1/sqrt(2)
	psi_0[1] = 1/sqrt(2) 

	#time values within each driving period
	#t_vals = arange(0,T+dt/2,dt)
	t_vals = linspace(0,T,num=N_steps+1)
	dt = t_vals[1] - t_vals[0]
	print(dt)


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
	t1 = perf_counter()
	for i in range(0,N_steps):
		print("i = {}".format(i))
		_H , _L1, _L2 = H_eff(t_vals[i],return_jump_ops=True)

		dt_found = find_timestep(_H,order,dt,tol=tol)
		print(dt_found)

		if dt_found < 0:#dt:
			#Handle the case that we had to dynamically decrease the timestep
			#We build up U(t,t+dt) by computing the intermediate steps
			exponent = int(around(log2(dt/dt_found)))
			print("Exponent: {}".format(exponent))
			max_exponent = max(max_exponent,exponent)
			step = 2**(-exponent)

			_dU = time_evolution(dt_found,_H,order=order)
			for j in range(1,2**exponent):
				_H = H_eff(t_vals[i]+j*step,return_jump_ops=False)
				_dU = time_evolution(dt_found,_H,order=order) @ _dU

			dU[i] = _dU
			U_current = dU[i] @ U_current
			U_0_to_t[i+1] = U_current
				
		else:
			#Use H_eff(t) to compute U_eff(t,t+dt) & concatenate it to U_eff(0,t)
			dU[i] = time_evolution(dt,_H,order=order)
			U_current = dU[i] @ U_current
			U_0_to_t[i+1] = U_current
			
			L_ops[i] = (_L1 , _L2)	#Store the jump operators as well!
	#end loop over driving period

	#Now construct the evolution operators U_eff(t,T)
	indices = list(reverse(U_0_to_t))
	for i in range(1,len(indices)):  
		U_t_to_T[indices[i]] = U_t_to_T[indices[i-1]] @ dU[indices[i]]


	L_ops[N_steps] = L_ops[0]	#Store the jump operators at t=T (useful later)
	U_T = U_0_to_t[N_steps]		#The evolution operator over a full period

	t2 = perf_counter()
	print("Time taken: {} seconds".format(t2-t1))
	exit(0)



	jumps = []
	well_imbalance = []
	# other_paulis = []	#figure out from Frederik what this is
	# #rhos = []
	weights = []
	psis = []
	# #wigners = []
	for sample_num in range(nsamples):
		#Keep track of jumps & imbalance for this sample
		jumps_this_sample = []
		imbalance_this_sample = []
		other_paulis_this_sample = []
		weights_this_sample = []

		r = rand()	#Random number to compare norm of wavefunction to
		print("Starting r: {}".format(r))

		psi = psi_0		#The current wavefunction, initialized to psi_0
		times = []
		
		for i in range(num_periods):
						
			#Indices used by binary search within period
			start_index = 0			#time of previous jump
			ind_left = 0			#left endpoint of current search interval
			ind_right = N_steps 	#right endpoint of current search interval

			psi_new = U_T @ psi 	#Evolve by a full period 

			#Check if there was a quantum jump within this period
			while (1 - norm(psi_new)**2) > r:
				print("-------- JUMP DETECTED --------")

				#Binary search for the time at which the jump occurred
				while (ind_right - ind_left) > 1:
					ind_mid = round((ind_right+ind_left)/2)
					

					if start_index != 0:
						#If we've had a previous jump within this period, we evolve from
						#	the time of the previous jump - NOT the start of the period!
						#NB: "psi" is the wavefunction at the previous jump time (see ****)
						psi_temp=psi
						for index in range(start_index,ind_mid):
							psi_temp = dU[index] @ psi_temp
					else:
						psi_temp = U_0_to_t[ind_mid] @ psi

					if (1-norm(psi_temp)**2) >= r:
						ind_right = ind_mid
					else:
						ind_left = ind_mid

				#We've now found the time at which the jump ocurred
				ind = round((ind_right+ind_left)/2)
				print("Jump index: {} of {}".format(ind,N_steps))
				jump_time = i*T + dt*ind

				#Advance psi to time t (****)
				if start_index != 0:
					#Evolve from prior jump (if there was one)
					for index in range(start_index,ind):
						psi = dU[index] @ psi
				else:
					#Otherwise, evolve from beginning of period
					psi = U_0_to_t[ind] @ psi 	

				#Jump operators at time t
				L_1 , L_2 = L_ops[ind]

				#Determine the type of jump, and jump the wavefunction accordingly
				p_1 = norm(L_1 @ psi)**2
				p_2 = norm(L_2 @ psi)**2
				if rand() < p_1/(p_1+p_2):
					jumps_this_sample.append((1,jump_time))
					psi = L_1 @ psi
				else:
					jumps_this_sample.append((2,jump_time))
					psi = L_2 @ psi

				#In either case, re-normalize the wavefunction and re-set the random variable r
				print("Norm before normalization: {}".format(norm(psi)))
				psi = psi/norm(psi)
				print("Norm after normalization: {}".format(norm(psi)))
				r = rand()
				print("\t"+"New r : {}".format(r))
			
				#Evolve to the end of the period (NOW from the jump time!!), reset the binary search
				#	markers, and rinse/repeat (if necessary)
				psi_new = U_t_to_T[ind] @ psi 	
				print("Norm after evolution to end of period: {}".format(norm(psi_new)))
				start_index  = ind 	#Update start_index to mark the time of this jump
				ind_left = ind 
				ind_right = N_steps

			#end jump while block

			#We've reached the end of this driving period. Update "psi" accordingly
			psi = psi_new
			weights_this_sample.append(abs(psi)**2/norm(psi)**2)
			
	# 		#t_end = perf_counter()
	# 		#print("Time elapsed: {}".format(t_end-t_begin)) 

			#Measure once only every oscillation period of the bare LC circuit 
			if ((i+1) % 4) == 0:
			 	psi_full = V_window @ psi 	#Convert back to phi basis
			 	imbalance_this_sample.append(sum(abs(psi_full)**2*well_projector)/norm(psi_full)**2)
			 	#other_paulis_this_sample.append(sum(psi_full @ psi_full[::-1].conj())/norm(psi_full)**2)
		
		#end for loop over cycles
		# t2 = perf_counter()
		# print("Time elapsed: {} seconds".format(t2-t1))
		#weights.append(weights_this_sample)

	 	#end for loop over cycles
		jumps.append(jumps_this_sample)
		psis.append(psi)
		weights.append(weights_this_sample)
		well_imbalance.append(imbalance_this_sample)

	save(outfile,[jumps,psis,weights,well_imbalance])