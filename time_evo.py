"""
Module containing functions to integrate the time-dependent Schrodinger equation
"""

#Add directory with Frederik's code to path
from sys import path
path.append("../Frederik/")

from units import *
from numpy import eye, complex128, zeros
from scipy.integrate import complex_ode


def time_evolve(psi,H,dt,order=5):
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


def time_evolution(dt,H,order=5):
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
	U = eye(H.shape[0],dtype=complex128)
	for i in range(H.shape[0]):
		U[:,i] = time_evolve(U[:,i],H,dt,order)
	return U


def find_evolutions_two_part(t_vals,W,H_1,H_2,W_params=None,order=10):
	"""
	Computes the time-evolution operators {U(t_i+1,t_i)}_i for a 
		time-dependent Hamiltonian of the form:
			H(t) = H_1 + W(t)*H_2					(1)

	Parameters
    ----------
    t_vals: array
    	Array of times t_i
    W: callable
    	When called with argument t, returns the value of W(t) in the Hamiltonian (1) above 
	H_1: ndarray
		The static Hamiltonian H_1 in (1) above, expressed as a matrix
	H_2: ndarray
		The modulated Hamiltonian H_2 in (1) above, expressed as a matrix
	order: int, optional
		The order of numerical integration (passed to time_evolve()). Defaults to 3
	W_params: tuple, optional
		Tuple of extra arguments to include when calling W. Defaults to None

    Returns
    -------
    Us : list of ndarray
        A list of the time-evolution operators {U(t_i+1,t_i)}_i
	"""
	dUs = []
	evos = []
	for i,t in enumerate(t_vals[:-1]):
		if W_params:
			_H = H_1 + W(t,*W_params)*H_2
		else:
			_H = H_1 + W(t)*H_2
		dt = t_vals[i+1]-t
		dUs.append(time_evolution(dt,_H,order=order))
		if evos:
			evos.append(dUs[-1]@evos[-1])
		else:
			evos.append(dUs[-1])

	return dUs, evos


def find_evolutions(t,dt,H_effs_herm,H_effs_NH,W,well_nums,delta_t=1.0):
	"""
	Compute the (effective non-unitary) evolution operastors for each well from t -> t+dt 

	Parameters
    ----------
    t : float
        The start time t 
    dt: float
    	The time step to the end time t+dt 
    H_effs_herm: dict
    	A dictionary of the hermitian part of H_eff for each well
    H_effs_NH: dict
    	A dictionary of the non-hermitian part ((-i/2)*(L^dagL + ...)) of H_eff for each well
    W: callable
    	Function encoding the time-dependence of H_eff according to H_eff = H_eff_herm + W(t)*H_eff_NH 
	well_nums: array-like
		List of well numbers
	delta_t: float, optional
		Parameter used to scale the argument of W, e.g. W is called with argument t/delta_t.
			Defaults to 1

    Returns
    -------
    Us : list
        A list of evolution operators U(t,t+dt) for each well
	"""
	#print("On time t/Delta = {}".format(t/delta_t))
	Us = []
	for well_num in well_nums:
		H_herm , H_NH = H_effs_herm[well_num] , H_effs_NH[well_num]
		D = H_herm.shape[0]
		U = zeros((D,D),dtype=complex128)

		_f = lambda t,v: (-1j/hbar)*(H_herm @ v + W(t/delta_t)*(H_NH @ v))
		_ode = complex_ode(_f)
		for i in range(D):
			psi_0 = zeros(D,dtype=complex128)
			psi_0[i] = 1.0

			_ode.set_initial_value(psi_0,t)
			U[:,i] = _ode.integrate(t+dt)
			
		Us.append(U)
		# if well_num==0:
		# 	print("Max element: {}".format(abs(U).max))



	return Us


def find_evolutions_exact(t,dt,H_effs_herm,L1s,L2s,W,well_nums,delta_t=1.0):
	"""
	Compute the (effective non-unitary) evolution operastors for each well from t -> t+dt 

	Parameters
    ----------
    t : float
        The start time t 
    dt: float
    	The time step to the end time t+dt 
    H_effs_herm: dict
    	A dictionary of the hermitian part of H_eff for each well
    H_effs_NH: dict
    	A dictionary of the non-hermitian part ((-i/2)*(L^dagL + ...)) of H_eff for each well
    W: callable
    	Function encoding the time-dependence of H_eff according to H_eff = H_eff_herm + W(t)*H_eff_NH 
	well_nums: array-like
		List of well numbers
	delta_t: float, optional
		Parameter used to scale the argument of W, e.g. W is called with argument t/delta_t.
			Defaults to 1

    Returns
    -------
    Us : list
        A list of evolution operators U(t,t+dt) for each well
	"""
	#print("On time t/Delta = {}".format(t/delta_t))
	Us = []
	for well_num in well_nums:
		H_herm , H_NH = H_effs_herm[well_num] , H_effs_NH[well_num]
		D = H_herm.shape[0]
		U = zeros((D,D),dtype=complex128)

		_f = lambda t,v: (-1j/hbar)*(H_herm @ v + W(t/delta_t)*(H_NH @ v))
		_ode = complex_ode(_f)
		for i in range(D):
			psi_0 = zeros(D,dtype=complex128)
			psi_0[i] = 1.0

			_ode.set_initial_value(psi_0,t)
			U[:,i] = _ode.integrate(t+dt)
			
		Us.append(U)
		# if well_num==0:
		# 	print("Max element: {}".format(abs(U).max))



	return Us