#Add directory with Frederik's code to path
from sys import path
path.insert(0,"../Frederik/")

from units import *
from numpy import complex128


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


#This method doesn't seem to work very well...
def time_evolution(t1,t2,dt,H,order=1):
	"""
	Computes the time-evolution operator U(t1,t2)

	Parameters
    ----------
    t1 : float
        The initial time
    t2: float
    	The end time
    dt: float
    	The time step 
    H: callable
    	Given a time t, H(t) returns the Hamiltonian at time t, expressed as a matrix 
	order: int, optional
		Order of approximation to compute to. Defaults to 1

    Returns
    -------
    U : ndarray
        The time-evolution operator U(t1,t2) 
	"""
	t_vals = arange(t1,t2+dt/2,dt)

	U = eye(D,dtype=complex128)
	for t in t_vals:
		for i in range(D):
			U[:,i] = time_evolve(U[:,i],H,t,dt,order)
		break

	return U


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