#Add directory with Frederik's code to path
from sys import path
path.insert(0,"../Frederik/")

from gaussian_bath import bath,get_J_ohmic
from basic import get_tmat

#import jump_static as js
import jump_static_optimized as js
from integrate import time_evolution, find_timestep
from params import *

from numpy.linalg import eigh,eigvals,norm,svd
from scipy.linalg import expm
from scipy.special import gamma as gamma_fn
from numpy import linspace,diag,cos,exp,around,log2
from numpy.random import rand,seed

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from time import perf_counter


def spectral_norm(M,herm=False):
	"""
	Return the spectral norm (largest singular value) of M
	"""
	if herm:
		eigenvalues = eigvalsh(M)
		return max(abs(eigenvalues[0]),eigenvalues[-1])
	else:
		return svd(M,compute_uv=False)[0]




if __name__ == "__main__":
	from sys import argv

	if len(argv) != 3:
		print("Usage: <infile> <line>")
		exit(0)

	infile = argv[1]
	line = int(argv[2])

	with open(infile,'r') as f:
		lines = f.readlines()
		_line = lines[line].split(" ")
		start_ind = int(_line[0])
		stop_ind = int(_line[1])
		outfile = _line[2]

	print("{} {} {}".format(start_ind,stop_ind,outfile))

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

#Now onto original(-ish) stuff by me

	#Fourier components for the resistor coupling W(t)
	#W_fourier = [1j*(1-exp(1j*Omega*q*dt_JJ))/(2*pi*q) for q in range(-q_max,q_max+1)]
	W_fourier = [sqrt(pi)*tau*exp(-Omega*q*(-4j*t_0+q*Omega*tau**2)/4) for q in range(-q_max,q_max+1)]
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
			#If 0 <= t1 < delta_t, there are two cases:
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



	# Constructing jump operators #

	#The times we will be sampling the jump + time evolution operators at
	dt = T/N_steps
	t_vals = dt*array(range(start_ind,stop_ind+1))
	U_0_to_t = []
	dU = []
	L_ops = []
	for i,t in enumerate(t_vals[:-1]):
		_H, _L1, _L2 = H_eff(t,return_jump_ops=True)
		L_ops.append((_L1,_L2))

		#Estimate the timestep needed from the spectral norm
		rho = spectral_norm(_H)
		fac = gamma_fn(order+2)
		_dt = min((fac*tol)**(1/(order+1))*hbar/rho,dt)
		_dt = dt/ceil(dt/_dt)	#Largest timestep <= _dt that divides dt

		#Integrate using the above timestep
		_dU = time_evolution(_dt,_H,order=order)
		_t = t+_dt
		while _t <= t_vals[i+1]+_dt/2:
			_H = H_eff(_t,return_jump_ops=False)
			_dU = time_evolution(_dt,_H,order=order) @ _dU
			_t += _dt

		#Record time evolution operators
		dU.append(_dU)
		if(i==0):
			U_0_to_t.append(_dU)
		else:
			U_0_to_t.append(_dU @ U_0_to_t[-1])


	


	
	save(outfile,[U_0_to_t,dU,L_ops])