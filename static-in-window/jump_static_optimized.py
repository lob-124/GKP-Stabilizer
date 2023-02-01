from numpy import sqrt,abs,sign,exp,pi,zeros,eye,arange,vdot,complex128
from numpy.linalg import norm

from numba import jit



@jit(nopython=True)
def f(E,t,W_ft,delta_t,gamma,temp,Lambda,omega_0=1):
    """
    
    Return the scalar-valued function f appearing in the jump operators

    Parameters
    ----------
    E: float
        The energy argument
    t: float
        The time argument
    W_ft: tuple of array(complex), array(float)
        The Fourier components of the switching function W, and the frequencies
    delta_t: float
        The time window over which the switching function is non-zero
    gamma: float
        The system-bath coupling strength
    temp: float
        Effective temeprature of the bath
    lambda: float
        Decay length appearing in the bath spectral function
    omega0: float, optional
        Cutoff frequency in bath spectral function. Defaults to unity

    Returns
    -------
    val : float
        Value of jump-operator function f(E;t)
    """
    val = 0
    for W_w,w in zip(*W_ft):
        omega = E + w
        if abs(omega) < 1e-14:
            J = temp/omega_0
        else:
            S_0 = abs(omega)*exp(-(omega**2)/(2*Lambda**2))/omega_0
            BE = (1)/(1-exp(-omega/temp))*sign(omega)
            J = S_0 *BE

        val += sqrt(2*pi*gamma*J)*W_w*exp(-1j*w*t + 1j*E*(t-delta_t))

    return val 


def ff(E,t,W_ft,delta_t,gamma,temp,Lambda,omega_0=1):
    """
    
    Return the scalar-valued function f appearing in the jump operators

    Parameters
    ----------
    E: float
        The energy argument
    t: float
        The time argument
    W_ft: tuple of array(complex), array(float)
        The Fourier components of the switching function W, and the frequencies
    delta_t: float
        The time window over which the switching function is non-zero
    gamma: float
        The system-bath coupling strength
    temp: float
        Effective temeprature of the bath
    lambda: float
        Decay length appearing in the bath spectral function
    omega0: float, optional
        Cutoff frequency in bath spectral function. Defaults to unity

    Returns
    -------
    val : float
        Value of jump-operator function f(E;t)
    """
    val = 0
    for W_w,w in zip(*W_ft):
        omega = E + w
        if abs(omega) < 1e-14:
            J = temp/omega_0
        else:
            S_0 = abs(omega)*exp(-(omega**2)/(2*Lambda**2))/omega_0
            BE = (1)/(1-exp(-omega/temp))*sign(omega)
            J = S_0 *BE

        temp = abs(sqrt(2*pi*gamma*J)*W_w*exp(-1j*w*t + 1j*E*(t-delta_t)))
        print(temp)
        val += sqrt(2*pi*gamma*J)*W_w*exp(-1j*w*t + 1j*E*(t-delta_t))

    return val 


@jit(nopython=True)
def L_tilde_energy_basis(X,t,energies,W_ft,delta_t,gamma,temp,Lambda,omega_0=1):
    """
    Computes the (non-time evolved) jump operator \tilde{L}, in the energy basis of the 
        static system Hamiltonian H_0

    Parameters
    ----------
    X : ndarray
        The physical operator the jump operator is formed out of. Should be a matrix (2d array)
            of components in the energy basis 
    t: float
        Time. Passed as an argument to the function f 
    energies: array
        An array of the energy eigenvalues correpsonding to each eigenvector in basis above
    W_ft: tuple of array(complex), array(float)
        The Fourier components of the switching function W, and the frequencies
    delta_t: float
        The time window over which the switching function is non-zero
    gamma: float
        The system-bath coupling strength
    temp: float
        Effective temeprature of the bath
    lambda: float
        Decay length appearing in the bath spectral function
    omega0: float, optional
        Cutoff frequency in bath spectral function. Defaults to unity

    Returns
    -------
    L_tilde : ndarray
        The (non-time evolved) jump-operator in the energy basis. The components are:
                \tilde{L}_{mn} = f(E_n-E_m; t)X_{mn} 
    """
    D = len(energies)
    L_tilde = zeros((D,D),dtype=complex128)

    for m in range(D):
        for n in range(D):
            #L_tilde[m,n] = exp(-(energies[n]-energies[m])/t)*X[m,n]
            L_tilde[m,n] = f(energies[n]-energies[m],t,W_ft,delta_t,gamma,temp,Lambda,omega_0)*X[m,n]

    return L_tilde


if __name__ =="__main__":
    from numpy import array

    X = array([[0.0,1.0],[1.0,0.0]],dtype=complex128)
    energies = [1.0,2.0]

    print(f(1.0,0.2,([2.0,3.0],[1.0,2.0]),.05,0.5,1.0,1.0))
    print(f(2.0,0.2,([2.0,3.0],[1.0,2.0]),.05,0.5,1.0,1.0))
    print(L_tilde_energy_basis(X,0.2,energies,([2.0,3.0],[1.0,2.0]),.05,0.5,1.0,1.0))