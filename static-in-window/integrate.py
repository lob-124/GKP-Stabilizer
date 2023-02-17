"""
Module containing functions to integrate the time-dependent Schrodingr equation
"""

#Add directory with Frederik's code to path
from sys import path
path.insert(0,"../Frederik/")

from units import *
from numpy import complex128,exp,angle
from scipy.special import factorial
from scipy.linalg import expm

from numpy.linalg import norm


def time_evolve(psi,H,t,dt,order=3):
	"""
	Computes the time-evolution psi(t) -> psi(t+dt)

	Parameters
    ----------
    psi : array
        The wavefunction at time t 
    H: callable
    	Given a time t, H(t) returns the Hamiltonian at time t, expressed as a matrix 
    t: float
    	The end time
	dt: float
		Timestep to evolve by 
	order: int, optional
		Order of approximation to compute to. Defaults to 3

    Returns
    -------
    psi_prime : array
        The wave function at time t+dt 
	"""
	_H = H(t)
	psi_prime = psi.astype(complex128)

	dpsi = psi
	for n in range(1,order+1):
		dpsi = (-1j*dt/(n*hbar))*(_H @ dpsi)
		psi_prime += dpsi

	return psi_prime


def time_evolve(psi,H,dt,order=3):
	"""
	Computes the time-evolution psi(t) -> psi(t+dt)

	Parameters
    ----------
    psi : array
        The wavefunction at time t 
    H: ndarray
    	The Hamiltonian H(t) at time t, expressed as a matrix 
	dt: float
		Timestep to evolve by 
	order: int, optional
		Order of approximation to compute to. Defaults to 3

    Returns
    -------
    psi_prime : array
        The wave function at time t+dt 
	"""
	psi_prime = psi.astype(complex128)

	dpsi = psi
	for n in range(1,order+1):
		dpsi = (-1j*dt/(n*hbar))*(H @ dpsi)
		psi_prime += dpsi
		
	return psi_prime


def time_evolution(t,dt,H,order=3):
	"""
	Computes the time-evolution operator U(t,t1+dt)

	Parameters
    ----------
    t1 : float
        The initial time
    dt: float
    	The time step 
    H: callable
    	Given a time t, H(t) returns the Hamiltonian at time t, expressed as a matrix 
	order: int, optional
		Order of approximation to compute to. Defaults to 3

    Returns
    -------
    U : ndarray
        The time-evolution operator U(t1,t1+dt)
	"""
	_H = H(t)

	U = eye(_H.shape[0],dtype=complex128)
	for i in range(_H.shape[0]):
		if i>=0:
			print("For i = {}".format(i))
			U[:,i] = time_evolve(U[:,i],_H,dt,order,_print=True)
		else:
			U[:,i] = time_evolve(U[:,i],_H,dt,order,_print=False)
	
	return U


def time_evolution(dt,H,order=3):
	"""
	Computes the time-evolution operator U(t,t+dt)

	Parameters
    ----------
    dt: float
    	The time step 
    H: ndarray
    	The Hamiltonian at H(t) at time t, expressed as a matrix 
	order: int, optional
		Order of approximation to compute to. Defaults to 3. If order < 0,
			compute using expm() instead

    Returns
    -------
    U : ndarray
        The time-evolution operator U(t1,t1+dt)
	"""
	if order < 0:
		return expm(-1j*H*dt)
	else:
		U = eye(H.shape[0],dtype=complex128)
		for i in range(H.shape[0]):
			U[:,i] = time_evolve(U[:,i],H,dt,order)
		return U


def find_order(H,dt,tol=1e-10):
	"""
	For the given matrix H, estimate the order "n" at which 
		||(dt H)^n/N!|| <= tol
	"""
	bound = 0.0

	#Bound the maximal eigenvalue using Gershgorin disk theorem
	for row in H:
		#The complex magnitude of the point in this Gershgorin disk farthest from 
		#	the origin is the sum of the absolute values of all elements in this row
		#	->ie, Magnitude of disk center + radius of disk
		R = sum(abs(row))
		bound = max(bound,R)


	log_tol = log(tol)
	n = 1
	while True:
		#We're interested in the inequality:
		#		(dt*bound)**/n! <= tol
		#To sidestep overflow issues, we compute the logarithm of both sides
		log_n_fac = log(2*pi*n)/2 + n*log(n) - n
		# if ((dt*bound)**n/factorial(n)) <= tol:
		# 	return n
		if ((n*log(dt*bound) - log_n_fac) <= log_tol):
			return n
		elif n > 1000:
			print("Error: Possible overflow detected. aborting")
			return -1
		else:
			n += 1


def find_timestep(H,order,dt_start,tol=1e-10):
	"""
	For the given matrix H and order n, estimate the timestep dt at which 
		||(dt H)^n/n!|| <= tol
	"""
	bound = 0.0

	#Bound the maximal eigenvalue using Gershgorin disk theorem
	for row in H:
		#The complex magnitude of the point in this Gershgorin disk farthest from 
		#	the origin is the sum of the absolute values of all elements in this row
		#	->ie, Magnitude of disk center + radius of disk
		R = sum(abs(row))
		bound = max(bound,R)


	log_tol = log(tol)
	dt=dt_start
	while True:
		#We're interested in the inequality:
		#		(dt*bound)**n/n! <= tol
		#To sidestep overflow issues, we compute the logarithm of both sides
		log_n_fac = log(2*pi*order)/2 + order*log(order) - order
		
		if ((order*log(dt*bound) - log_n_fac) <= log_tol):
			return dt
		elif dt < 1e-15:
			print("Error: Reached numerical precision. Aborting")
			return -1
		else:
			dt = dt/2


# def time_evo(t_1,t_2,H,dt=None):
# 	"""
# 	Computes the time-evolution operator U(t_2,t_1) from t_1 to t_2

# 	Parameters
#     ----------
#     t_1 : float
#         The starting time 
#     t_2: float
#     	The end time
#     H: callable
#     	Given a time t, H(t) returns the Hamiltonian, expressed as a matrix in a chosen basis 
# 	energies: array
# 		An array of the energy eigenvalues correpsonding to each eigenvector in basis above
# 	dt: float
# 		Timestep to use 

#     Returns
#     -------
#     L_tilde : ndarray
#         The (non-time evolved) jump-operator in the energy basis. The components are:
#         		\tilde{L}_{mn} = f(E_n-E_m; t)X_{mn} 
# 	"""

# 	D = H(0).shape[0]		#Hilbert space dimension

# 	#Right-hand side of the ODE i*h_bar*y' = H(t)y
# 	#	-> TODO: is hbar = 1??
# 	def _f(t,y):
# 		return -1j*H(t) @ y


# 	schr_eqn = ode(_f)	#the ODE object
# 	schr_eqn.set_integrator("zvode")
# 	U = zeros((D,D),dtype=complex128)	#matrix where we store the output

# 	#Time evolve each basis vector to get the columns of the unitary U
# 	#	!!!!!!!!!!!!!!!
# 	#	-> TODO: This isn't unitary. Need to replace
# 	#	!!!!!!!!!!!!!!!
# 	for i in range(D):
# 		c = zeros(D,dtype=complex128)
# 		c[i] = 1.0
# 		schr_eqn.set_initial_value(c,t_1)
# 		U[:,i] = schr_eqn.integrate(t_2)
		

# 	return U