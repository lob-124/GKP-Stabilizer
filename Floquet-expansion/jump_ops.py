from numpy import sqrt,abs,sign,exp,pi,zeros,complex128,outer

from numba import jit


@jit(nopython=True)
def L(t,X,floq_states,quasi_energies,W_ft,gamma,temp,Lambda,omega_0=1):
    """
    Computes the jump operator L(t), assuming the system Hamiltonian is periodic

    Parameters
    ----------
    t: float
        Time t. 
    X : tuple of (ndarray(complex), array(float))
        The Fourier components 
                X^{jk}_n = (1/T)\int_0^T <\phi_j(t)|X|\phi_k(t)>exp(in\Omega t)dt
        of the system operator X, and the frequencies. The first tuple element is assumed to be 
        a three index array: Fourier index (n), followed by the basis labels (j,k). The second tuple 
        element is the frequencies
    floq_states: ndarray or None
        An array of the Floquet states (i.e. the eigenstates of the Floquet effective Hamiltonian
            evolved by the micromotion operator).
        If None, returns the jump operator in the basis spanned by the Floquet states 
    quasi_energies: array
        An array of the quasienergies corresponding to the floquet states above
    W_ft: tuple of array(complex), array(float)
        The Fourier components of the switching function W, and the frequencies
    gamma: float
        The system-bath coupling strength
    temp: float
        Temperature of the bath
    lambda: float
        Decay length appearing in the bath spectral function
    omega0: float, optional
        Cutoff frequency in bath spectral function. Defaults to unity

    Returns
    -------
    L : ndarray
        The jump operator at time t, as a matrix in the given basis
    """
    D = len(quasi_energies)
    L_X = len(X[1])
    L_W = len(W_ft[1])
    _L = zeros((D,D),dtype=complex128)

    for k in range(D):
        for l in range(D):
            delta_E = quasi_energies[l] - quasi_energies[k]

            #Sum over Fourier components of H_sys and W
            _sum = 0.0+0.0j
            for n in range(L_X):
                for q in range(L_W):
                    fourier_E = X[1][n] + W_ft[1][q]
                    omega = delta_E + fourier_E

                    #Compute the jump correlator J
                    if abs(omega) < 1e-14:
                        J = temp/omega_0
                    else:
                        S_0 = abs(omega)*exp(-(omega**2)/(2*Lambda**2))/omega_0
                        BE = (1)/(1-exp(-omega/temp))*sign(omega)
                        J = S_0 *BE

                    _sum += X[0][n,j,k]*W_ft[0][q]*sqrt(2*pi*J)*exp(-1j*t*fourier_E) 

            _L[k,l] = _sum*sqrt(gamma)
    
    if floq_states is None:
        #Return L in the basis of floquet states at time t
        return _L
    else:
        #Transform back to the given basis
        return floq_states @ _L @ floq_states.conj().T


