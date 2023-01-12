#Add directory with Frederik's code to path
from sys import path
path.insert(0,"../Frederik/")

import gaussian_bath as gb


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
	Return the (non-time evolved) jump operator \tilde{L}, in the energy basis of the 
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
	omega = 0.1
	E1 , E2 = 2.0,1.0
	print(gb.window_function(omega,E1,E2))