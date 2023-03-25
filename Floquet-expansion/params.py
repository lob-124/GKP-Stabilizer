#Add directory with Frederik's code to path
from sys import path
path.insert(0,"../Frederik/")

from units import *
from numpy import linspace


nsamples = 10            # Number of samples in SSE simulation
num_periods = 5000          # Number of cycles to include in simulation
D        = 1000        # Dimension of Hilbert space
N_wells  = 10
pvec     = linspace(-2,2,400)   # Momenta to smaple for wigner function

#Number of eigenvectors in Hilbert space to keep
N_trunc = 500
 
omega_c  = 0.5*THz         # Bath cutoff frequency
gamma    = 1e-6*meV#1.2e-3*meV      # Bare system-bath coupling. Translates to a resistance. 
Temp     = 0.001*Kelvin#2e-3*Kelvin   # Bath temperature 
omega0   = 1*meV         # Normalizing frequency that enters in ohmic spectral function

Josephson_energy = 100*GHz 
cavity_frequency = 5*GHz

# Period of the cycle
Tc = 2*pi/cavity_frequency

# Duration at which josephson junction is turned on
#dt_JJ = 0.032*Tc
#dt_0 = Tc/4*0.972	#What was this?
#T = Tc/4+dt_JJ	#Why was this here?
T = Tc/4
dt_JJ = .1*T#.2*T
Omega = 2*pi/T


#Time-dependent JJ coupling
def V_t(t):
	if 0 <= t < dt_JJ:
		return 1.0
	else: 
		return 0.0 


#Number of timesteps per driving period and order for numerical integration
#	Choose N_steps to be a power of 2 since we'll binary search in each period
N_b_unitary = 9
N_steps_unitary = 2**N_b_unitary
N_b_NH = 9
N_steps_NH = 2**N_b_NH
#dt = .005*T
order = 5
tol = 1e-6


quantization_parameter  = 1.0

#Frequency to truncate fourier expansion of switching function at
#	Assumes omega_q = 2*pi*q/T
n_max = 100
q_max = 115 

#Center and decay length of Gassian switching function
#	Assumes W(t) = exp(-((t-t_0)/tau)^2)
t_0 = dt_JJ/2
tau = dt_JJ/6
