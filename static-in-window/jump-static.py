#Add directory with Frederik's code to path
from sys import path
path.insert(0,"../Frederik/")

import gaussian_bath as gb
from units import *

from numpy import sqrt,exp,zeros,eye,arange,vdot,complex128
from numpy.linalg import norm
from scipy.integrate import ode


def f(bath,W_ft,delta_t,gamma):
	"""
	
	Return the scalar-valued function f appearing in the jump operators

	Parameters
    ----------
    bath : bath
        The bath in question
    W_ft: tuple of array(complex), array(float)
    	The Fourier components of the switching function W, and the frequencies
    delta_t: float
    	The time window over which the switching function is non-zero
    gamma: float
    	The system-bath coupling strength

    Returns
    -------
    _f : method
        jump-operator function f(E;t)
	"""

	def _f(E,t):
		_temp = [sqrt(2*pi*gamma*bath.J(E+w))*W_w*exp(-1j*w*t +1j*E*(t-delta_t)) for W_w, w in zip(*W_ft)]
		return sum(_temp)

	return _f


def L_tilde_energy_basis(X,f,t,energies):
	"""
	Computes the (non-time evolved) jump operator \tilde{L}, in the energy basis of the 
		static system Hamiltonian H_0

	Parameters
    ----------
    X : ndarray
        The physical operator the jump operator is formed out of. Should be a matrix (2d array)
        	of components in the energy basis 
    f: callable
    	The scalar function f, e.g. as returned by f() above
    t: float
    	Time. Passed as an argument to the function f 
	energies: array
		An array of the energy eigenvalues correpsonding to each eigenvector in basis above

    Returns
    -------
    L_tilde : ndarray
        The (non-time evolved) jump-operator in the energy basis. The components are:
        		\tilde{L}_{mn} = f(E_n-E_m; t)X_{mn} 
	"""
	D = len(energies)
	L_tilde = zeros((D,D))

	for i in range(D):
		for j in range(D):
			L_tilde[m,n] = f(energies[n]-energies[m],t)*X[m,n]

	return L_tilde


if __name__ == "__main__":

	from numpy.random import rand
	from numpy import diag,ones,allclose
	from scipy.linalg import expm

	####****     ****####
	####    TESTS    ####
	####****     ****####

	D = 100

	####
	##  Test 1: Generate time evo for random, static Hamiltonian
	####
	_H = rand(D,D)
	_H = (_H + _H.T)/2
	def H(t):
		return _H
	
	dt = .005
	for t in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
		psi_0 = rand(D)
		psi_0 = psi_0/norm(psi_0)

		psi = psi_0.astype(complex128)
		time = 0
		while time <= t:
			psi = time_evolve(psi,H,time,dt,5)
			time += dt
		print([1.0,vdot(psi,expm(-1j*t*_H/hbar)@psi_0)])


		U = time_evolution(0,t,dt,H,5)
		print(norm(U-expm(-1j*t*_H/hbar),ord="fro"))
		
		#assert allclose([1.0,U@psi.dot(expm(-1j*t*_H)@psi)]) , "failed at time t = {}".format(t)

	print("Test 1 passed!")




	####
	##  Test 2: Generate time evo for simple, exactly solvable time-dependent Hamiltonian
	####
	from numpy import cos, sin, pi
	def H(t):
		return diag([cos(pi*i*t/D) for i in range(1,D+1)])

	def U_exact(t):
		return expm(-1j*diag([(D/(i*pi))*sin(pi*i*t/D) for i in range(1,D+1)]))

	dt = .005
	for t in [0.1,0.2,0.3,0.4,0.5]:
		U = time_evolution(0,t,dt,H)
		assert allclose(U,U_exact(t)) , "failed at time t = {}".format(t)

	print("Test 2 passed!")


	####
	##  Test 3: Test unitarity 
	####
	from numpy import eye
	H_diag = diag(list(range(D)))
	H_off_diag = diag(ones(D-1),1) + diag(ones(D-1),-1)

	def H(t,omega=1.0):
		return cos(2*pi*t/omega)*H_diag + sin(2*pi*t/omega)*H_off_diag

	dt = .005
	for t in [0.1,0.2,0.3,0.4,0.5]:
		U = time_evolution(0,t,dt,H)
		assert allclose(U.conj().T @ U,eye(D)) , "failed at time t = {}".format(t)

	print("Test 3 passed!")