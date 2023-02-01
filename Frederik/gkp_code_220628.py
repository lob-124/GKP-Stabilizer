                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 13:46:29 2022

@author: frederiknathan

Simulations of GKP circuit. The JJ and the resistor is turend on simultaneously (for duration dt_JJ), in a stepwisee fashion
Using quasistatic approximation, which probably isnt justified!!!

###
Here are some paramters that give nice data:


nsamples = 1            # Number of samples in SSE simulation
NT       = 2000          # Number of cycles to include in simulation
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
dt_JJ = 0.032*Tc
dt_0 = Tc/4*0.972

T = Tc/4+dt_JJ
Omega = 2*pi/T
# Accuracty of timing. (not implemented yet)
timing_accuracy=100

quantization_parameter  = 0.999


"""

# print("%"*80)
# print("Warning: Using quasistatic approximation, which probably isnt justified!")
# print("%"*80)
from basic import *
from numpy import *
from numpy.linalg import *
from matplotlib.pyplot import *
from scipy.special import factorial,hermite
from units import *
from gaussian_bath import bath,get_J_ohmic
from wigner import wigner 
from basic import tic,toc
from scipy.interpolate import interp1d

# ----------------------------------------------------------------------------
# 1. Parameters 

nsamples = 1            # Number of samples in SSE simulation
NT       = 2000          # Number of cycles to include in simulation
D        = 1000        # Dimension of Hilbert space
N_wells  = 10
pvec     = linspace(-2,2,400)   # Momenta to smaple for wigner function

 
omega_c  = 0.5*THz         # Bath cutoff frequency
gamma    = 3e-3*meV      # Bare system-bath coupling. Translates to a resistance. 
Temp     = 0.001#2e-3*Kelvin   # Bath temperature 
omega0   = 1*meV         # Normalizing frequency that enters in ohmic spectral function

Josephson_energy = 100*GHz 
cavity_frequency = 5*GHz

# Period of the cycle
Tc = 2*pi/cavity_frequency

# Duration at which josephson junction is turned on
dt_JJ = 0.02*Tc
dt_0 = Tc/4*0.987

print(dt_JJ/(dt_0+dt_JJ))
T = Tc/4+dt_JJ
Omega = 2*pi/T
# Accuracty of timing. (not implemented yet)
timing_accuracy=100

quantization_parameter  = 0.999

# Derive physical quantities from parameters
impedance   = planck_constant/(2*quantization_parameter*e_charge**2)
L           = impedance/cavity_frequency
C           = 1/(cavity_frequency*impedance)
inductance  = 1*L
resistance  = hbar*omega0/(2*pi*e_charge**2*gamma)

### Phi grid 
phi_cutoff  = N_wells*2*pi
phi         = linspace(-phi_cutoff,phi_cutoff,D)
dphi        = phi[1]-phi[0]

well_projector = 1-(-1)**(phi/(2*pi)//1)-1

### canonically conjugate operators 

# dimensionless phase 
Phi     = diag(phi)
X1 = (expm(1j*Phi))
X2 = expm(-1j*Phi)

# canonically conjugate momentum to Phi
Tmat    = get_tmat(D,dtype=complex)
Pi0     = 1j*(Tmat-eye(D))/dphi

# squared momentum
Pi_2 = 0.5*(Pi0@Pi0.conj().T + Pi0.conj().T@Pi0)

# Squared charge operator 
Q2 = ((2*e_charge)/hbar)**2 * Pi_2 

# ----------------------------------------------------------------------------
# Construct Hamiltonian

# Cavity Hamiltonian
H0pot = Phi**2/(2*L)*(hbar**2/(4*e_charge))
H0kin = Q2/(2*C)

H0 = H0pot + H0kin

# Josephson potential. 
V  = Josephson_energy * diag(cos(phi))
H1 = H0 + V



### Finding fourier coefficients of micromotin operator
# N0 = 6* T/dt_JJ
dt = dt_JJ/5
N0 = T/dt
# assert (abs((N0+1)%2-1)<1e-10), "dt_JJ/2 must be a rational multiple of the driving period"
N0 = int(N0+0.5)

f_c = 20
zrange = arange(-f_c,f_c+1).reshape((2*f_c+1,1,1))

dU0 = expm(-1j*dt*H0)
dU1 = expm(-1j*dt*H1)


# ----------------------------------------------------------------------------
# Get bath (all ULE stuff is using a module I created previously titled gaussian_baths)

# Spectral function
J = get_J_ohmic(Temp,omega_c,omega0=omega0)

# Frequency resolution  (only used to  calculating bath timescales in ULE module).
wvec = linspace(-10*2*pi/dt_JJ,10*2*pi/dt_JJ,10000)
dw = wvec[1]-wvec[0]

# Bath object
B = bath(J,10*omega_c,10*omega_c/4000)

# Get jump operators and diagonalize the Hamiltoinan
X1 = (expm(1j*Phi))
X2 = expm(-1j*Phi)

# Get jump operators and eigensystem of Hamiltonian H1 (this is a valuable byproduct in the calculation)
[L1,L2],[E,U] = B.get_ule_jump_operator([X1,X2],H1,return_ed=True)

# ----------------------------------------------------------------------------
# Get objects for time evolution

U0 = expm(-1j*H0*dt_0)

Heff = H1 - 0.5j*gamma * (L1.conj().T@L1+L2.conj().T@L2)

n_it = 10

"""FN: This is a slightly compliicated optimization step I implemented for solving the stochastic schrodinger equation (not necessary, 
so perphas the standard implementation of the SSE is best as a first step, and then leave optimization for later). 

Baiscally I calculate the unitary time-evolution operator over a duration dt*2^n for n = 1 .. n_it. 
This allows me to efficiently do a logarithmic search for the time-step at which a quantum jump occurs
later on in the SSE part of the code(cf. the SSE algoorthm)"""
U1list = zeros((n_it,D,D),dtype=complex)
dt_JJ_i = dt_JJ/(2**n_it)
U1 = expm(-1j*Heff*dt_JJ_i)
a = 1/(2**n_it)
tlist = zeros((n_it))
flist = zeros((n_it),dtype=int)
b = 1 
for n in range(0,n_it):
    U1list[n,:,:] = U1 
    U1 = U1 @ U1
    tlist[n]= a
    a = a+a 
    flist[n] = b
    b = b+b


U1 = U1list[-1]
### U1 = evolution operator over the full step in the protocol where the JJ and resistor is turned on (
nss= 0
Wo = 0

### Initialize output lists for SSE simulation
WIlist = zeros(NT)
W2list = zeros(NT)
Clist = zeros(NT,dtype=complex)
X = zeros((NT,D))
P = zeros((NT,D))
psilist = zeros((NT,D),dtype=complex)

### Set initial state for SSE evolution (This is something we can play with).
E0,V0 = eigh(H0)
psi0 = 1*V0[:,0]+1*V0[:,1]

psi0 = psi0/(norm(psi0))

### Plot initial wavefunction as a function of phi.
figure(37)
plot(phi/(2*pi),psi0)
ylabel("$\psi_0$")
xlabel("$\phi/(2*pi)$")


### Run SSE simulation. Using waiting-time algorithm (look up on google -- its a very simple optimization  for the SSE).
rho = 0 
for ns in range(0,nsamples):
    
    print(f"At sample {nss}/{nsamples}")
    tic()
    psi = 1*psi0
    r = rand()
    
    
    nx = 0
    nj = 0
    np = 0
    for n in range(0,NT):
        if n%500==0:
            print(f"   at step {n:<5}/{NT}. Number of jumps: {nj}")
        psi = U0@psi
        psi1 = U1@psi 
  
        N = norm(psi1)**2


        ### Do logarithmic search for the time at which the quantum jump occurs


        if N<r : #  if N< a quantum jump occurs during this cycle.
            
            

            ### find the exact time (up to a resolution dt_JJ*2^(-n_it)) where the jump occurs.
            t = 0# 1/2**n_it
            f = 0 
            converged = False 
            n_j = 0
            while f<(2**n_it):
    
                mlist = zeros(n_it)
                for m in range(n_it-1,-1,-1):
                    if flist[m]+f<(2**n_it):
                        
                        
                        M = U1list[m]
                        psi2 = M @ psi
                        
                        if norm(psi2)**2>r:
                            t = t+tlist[m]
                            f = f + flist[m]
                            psi = psi2
   
                    
                psi = U1list[0]@psi
                f = f+1
                t = t + tlist[0]

                if norm(psi2)**2<r:
                    n1 = norm(L1@psi)**2
                    n2 = norm(L2@psi)**2
                    
                    m =n1+n2
                    
                    n1 = n1/m
                    n2=  n2/m
                    r12 = rand()
                    
                    if r12<n1:
                        psi = L1@psi 
                    else:
                        psi = L2@psi
                    
                    nj+=1 
                    

                    r = rand()
                    psi = psi/norm(psi)
                    
                    n_j+=1 
                    
            assert n_j>0
            
                
        ### Record output every 4 cycle for smooth output (since wavefunction rotates approx quarter period around in phase spacce for each driving cycle).
        if (n+1)%4==0:
        
            X[nx] += abs(psi)**2/(norm(psi)**2)/nsamples
            DN = D//(2*N_wells)
            WIlist[nx]+= sum(abs(psi)**2*well_projector)/(norm(psi)**2)
            Clist[nx] += sum(psi@psi[::-1].conj())/nsamples/norm(psi)**2 
            psilist[nx]=psi
            nx  +=1 
            
        if n%4 ==0:
            P[np] += abs(psi)**2/(norm(psi)**2)/nsamples
            
            W2list[np]+= sum(abs(psi)**2 * sin(phi/2))/nsamples/norm(psi)**2
            np+=1

        else:
            psi = psi1     
            
                    
                    
                
    
    U12 = expm(-1j*Heff*dt_JJ_i/2)   
    psi = U12@psi
    psi = psi/norm(psi)    
            
            
                
                
                
                
    ### Get density matrix
    rho = rho+outer(psi,psi.conj())

    ### Get wigner function (this is quite heavy computationally, maybe there is a way to optimize/parelallize, or maybe the Wigner function is not even necessary to compute)
    W,xo,po = wigner(psi,phi,pvec=pvec)

    Wo += W
        
    nss +=1

# Get average density matrix and wigner function for ensemble.
Wo = Wo/nss
rho = rho/nss


### Plot final wavefunction for the last trajectory of the ensemble
figure(19)
plot(phi,abs(psi)**2)

### Do a matrix plot of the density matrix vs. phi
figure(18)
pcolormesh(phi,phi,abs(rho))
nvec = arange(-5,5)
xticks(4*pi*nvec-pi,["$"+str(n*4-1)+"\pi$" for n in nvec])
yticks(4*pi*nvec-pi,["$"+str(n*4-1)+"\pi$" for n in nvec])
xlim((-12*pi,12*pi))
ylim((-12*pi,12*pi))
grid(linestyle="--",linewidth=0.5)
xlabel("$\phi$")
ylabel("$\phi'$")
title("Plot of density matrix $\\rho(\phi,\phi')$")
colorbar()

### Plot Wigner function
figure(20)
VMAX = amax(abs(Wo))
pcolormesh(xo,po,Wo,vmin=-VMAX,vmax=VMAX,cmap="bwr")
xticks(4*pi*nvec-pi,["$"+str(n*4-2)+"\pi$" for n in nvec])
yticks(0.5*nvec+.25,["$"+str(2*n+1)+"/4$" for n in nvec])
grid(linestyle="--",linewidth=0.5)
ylim(-2,2)
xlim((-12*pi,12*pi))
colorbar()
# Plot probablity distribution in P and X (I think).
figure(1)
tvec = arange(nx)
xvec = phi
phig,tg = meshgrid(xvec,tvec)
pcolormesh(phig/(2*pi),tg,X[:nx,:])


title("X")
figure(8)
pcolormesh(phig/(2*pi),tg,P[:nx,:])

title("P")
colorbar()
xlabel("$\phi/(2*pi)$")

### Plot the measure for the quibt chernecce.
figure(221)
plot(tvec,WIlist[:nx])
plot(tvec,real(Clist[:nx])+3e-3)
plot(tvec,imag(Clist[:nx])+6e-3)
xlabel("periods")
ylabel("Interwell imbalance")
title("Decay of interwell imbalance (relaxation time)")
# plot(tvec,W2list[:np])
legend(["$\\langle \sigma_z\\rangle$","$\\langle \sigma_x\\rangle$","$\\langle\sigma_y\\rangle$"])



### Plot Wigner function for the whole ensemble agian (maybe with a different resolution?)
tic()
figure(3)
Xg,Pg = meshgrid(xo,po)
print(toc())
VMAX = amax(abs(Wo))/1
pcolormesh(Xg/(2*pi),Pg/(2*pi),Wo,vmin=-VMAX,vmax=VMAX,cmap="bwr")
xlabel("$\phi / 2\pi$")
ylabel("$\Pi /2\pi$")
colorbar()
xlim(-5,5)
ylim(-.2,.2)
# ax  =gca()
# arrow(2*pi+.2,5,4*pi,0)
# arrow(2*pi+.2,0,0,8*pi)

plot()
# ax.set_aspect("equal")
tstr = "Wigner function of steady state. $\sqrt{L/C}$"+f"={sqrt(inductance/C)/(planck_constant/e_charge**2):.2} $e^2/h$, \nJ={Josephson_energy/GHz} GHz. JJ interval = {dt_JJ/T} T, $\gamma=${gamma/GHz:.2} GHz"
title(tstr)# ylim(-5*pi,5*pi)
print(toc())
