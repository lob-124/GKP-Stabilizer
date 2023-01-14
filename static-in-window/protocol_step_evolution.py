#Add directory with Frederik's code to path
from sys import path
path.insert(0,"../Frederik/")



import jump_static as js
from params import *



if __name__ == "__main__":

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
	X1 = (expm(1j*Phi))
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


	#Full Hamiltonian over one period:
	#      H = H0 0 <= t < \Delta t
	#      H = H1 \Delta t < t < T
