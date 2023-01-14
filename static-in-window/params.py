#Add directory with Frederik's code to path
from sys import path
path.insert(0,"../Frederik/")

from units import *


nsamples = 1            # Number of samples in SSE simulation
N_cycles = 2000          # Number of cycles to include in simulation
D        = 1000        # Dimension of Hilbert space
N_wells  = 10
pvec     = linspace(-2,2,400)   # Momenta to smaple for wigner function

 
omega_c  = 0.5*THz         # Bath cutoff frequency
gamma    = 1.2e-3*meV      # Bare system-bath coupling. Translates to a resistance. 
Temp     = 0.001#2e-3*Kelvin   # Bath temperature 
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
dt_JJ = 
Omega = 2*pi/T

quantization_parameter  = 0.999
