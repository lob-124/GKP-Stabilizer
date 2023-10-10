#Add directory with Frederik's code to path
from sys import path
path.insert(0,"../Frederik/")

from units import *
from numpy import linspace,pi
from basic import get_tmat

D        = 1000        # Dimension of Hilbert space
phi_0    = linspace(-pi,pi,D)
dphi     = phi_0[1]-phi_0[0]


## **** SYSTEM PARAMETERS **** ##
omega = 50e-3*GHz
E_J = 100*GHz*hbar

quantization_parameter  = 1.0
Z  = planck_constant/(2*quantization_parameter*e_charge**2)
L = Z/omega
C = 1/(Z*omega)



## **** PHYSICAL OPERATORS **** ##

# canonically conjugate momentum to Phi
Tmat    = get_tmat(D,dtype=complex)
Tmat[0,-1]=0
Pi0     = 1j*(Tmat-eye(D))/dphi
Pi_2 = 0.5*(Pi0@Pi0.conj().T + Pi0.conj().T@Pi0)	#squared momentum
Q2 = ((2*e_charge)/hbar)**2 * Pi_2 	#Squared charge