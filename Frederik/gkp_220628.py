#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 13:46:29 2022

@author: frederiknathan

Simulations of GKP circuit.

Playing with initialization of GKP circuit 


System is a time crystal!!!

"""

# print("%"*80)
# print("Warning: shutting on and off system-bath coupling at timescale dt_JJ gives the bath an effective  temperature of 1/dt_JJ (c.f. the formula for the ULE jump operator. This is not reflected in the present simulation")
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

nsamples = 10            # Number of samples in SSE simulation
NT       = 120          # Number of cycles to include in simulation
D        = 1000         # Dimension of Hilbert space
N_wells  = 15
pvec     = linspace(-2,2,400)   # Momenta to smaple for wigner function

 
omega_c  = 0.5*THz         # Bath cutoff frequency
gamma    = 4e-3*meV      # Bare system-bath coupling. Translates to a resistance. 
Temp     = 0.001#2e-3*Kelvin   # Bath temperature 
omega0   = 1*meV         # Normalizing frequency that enters in ohmic spectral function

Josephson_energy = 100*GHz 
cavity_frequency = 5*GHz

# Period of the cycle
Tc = 2*pi/cavity_frequency

# Duration at which josephson junction is turned on
dt_JJ = 0.01*Tc

T = Tc/4+dt_JJ
Omega = 2*pi/T
# Accuracty of timing. (not implemented yet)
timing_accuracy=100

quantization_parameter  = 0.99

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

# Construct Floquert operator
U1 = expm(-1j*H0*Tc/4)
U2 = expm(-1j*H1*dt_JJ/2)
U  = U2@U1@U2 

[Ph,Psi] = eig(U)
QE = -imag(log(Ph))/T
Heff = Psi@diag(QE)@Psi.conj().T

# Shifting quasienergies to make micromotion as smooth as possible. (for 2-step protocols the quasienergy equals the average energy mod Omega). By shifting the quasienergy to equal the average energy, the micromotion will be very smooth.
E0av = sum(Psi.conj() * (H0@Psi),axis=0)
E1av = sum(Psi.conj() * (H1@Psi),axis=0)

Eav = real((Tc/4*E0av+dt_JJ*E1av)/T)

dn = ((Eav-QE)/Omega+0.5).astype(int)
QE = QE + dn*Omega

[E0,V0] = eigh(H0)
[E1,V1] = eigh(H1)

### Finding fourier coefficients of micromotin operator
# N0 = 6* T/dt_JJ
dt = dt_JJ/5
N0 = T/dt
assert (abs((N0+1)%2-1)<1e-10), "dt_JJ/2 must be a rational multiple of the driving period"
N0 = int(N0+0.5)

f_c = 20
zrange = arange(-f_c,f_c+1).reshape((2*f_c+1,1,1))

dU0 = expm(-1j*dt*H0)
dU1 = expm(-1j*dt*H1)

P = Psi.reshape((1,D,D))
DM = exp(1j*dt*QE).reshape(1,1,D)
D0 = exp(-1j*dt*E0)
D1 = exp(-1j*dt*E1)

Pout = zeros((2*f_c+1,D,D))

n0=0
print("-"*80)
print("Computing fourier transform of micromotion operator")
tic()
interval = 1 

for n in range(0,N0):
    t = n*dt+dt/2
    
    # print((t-dt/2)/dt_JJ)
    if n%(N0//20)==0:
        print(f"   at step {n}/{N0}. Time spent: {toc(disp=False):.2}s")
    if t<dt_JJ/2 or (T-t)<dt_JJ/2:
        dU = dU1
        n0+=1
        # print(n0)
        interval = 1 
    else:
        dU = dU0 
        
        interval = 0 
        
    # if interval == 1 :
        
    
    P = (dU@P)*DM
    
    Pout = Pout + P*exp(1j*Omega*zrange*t)*dt/T
    
Pout = Pout@(Psi.conj().T).reshape((1,D,D))
Pconj = Pout.swapaxes(1,2).conj()
dt = T/10

figure(1)
yvec = sqrt(sum(abs(Pout)**2,axis=(1,2)))
plot(zrange[:,0,0],yvec,'.')
plot([0,0],[0,amax(yvec)],":",color="gray")
xlabel("fourier index")
ylabel("Frobenius norm")
title("Norm of fourier coefficients of the micromotion operator")
show()

X  = 0 * Pout 

x0 = X1.reshape((1,D,D))

print("Computing fourier transform of operators coupled to bath")
XP = x0 @ Pout 
for z in arange(0,2*f_c+1):
    v1 = XP[z:]
    if z>0:
        
        v2 = Pconj[:-z]
    else:
        v2= Pconj
    X[z:z+1] =sum(v2@v1,axis=0)#tensordot(v2@x0@v1,axes=([0],[0]))
# raise ValueError

figure(2)
yvec = sum(abs(X)**2,axis=(1,2))
plot(arange(0,len(zrange)),yvec,'.')
# plot([0,0],[0,amax(yvec)],":",color="gray")
xlabel("fourier index")
ylabel("Frobenius norm")
title("Norm of fourier coefficients of the exp(phase) operator")
show()

 

# ----------------------------------------------------------------------------
# Get bath

# Spectral function
J = get_J_ohmic(Temp,omega_c,omega0=omega0)

wvec = linspace(-10*2*pi/dt_JJ,10*2*pi/dt_JJ,10000)
dw = wvec[1]-wvec[0]

# Bath object
B = bath(J,10*omega_c,10*omega_c/4000)

# Get jump operators and diagonalize the Hamiltoinan
X1 = (expm(1j*Phi))
X2 = expm(-1j*Phi)

# raise ValueError
[L1,L2],[E,U] = B.get_ule_jump_operator([X1,X2],H,return_ed=True)

# ----------------------------------------------------------------------------
# Get objects for time evolution

dt_0 = T/4
U0 = expm(-1j*H0*dt_0)

Heff = H - 0.5j*gamma * (L1.conj().T@L1+L2.conj().T@L2)

# 
n_it = 10

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
    # a = a+a
    
U1 = U1list[-1]
U12 = expm(-1j*Heff*dt_JJ_i/2)
nss= 0
Wo = 0

WIlist = zeros(NT)
W2list = zeros(NT)
Clist = zeros(NT,dtype=complex)
X = zeros((NT,D))
P = zeros((NT,D))
psilist = zeros((NT,D),dtype=complex)
# psi0            = U[:,0]+U[:,1]

E0,V0 = eigh(H0)
psi0 = 1*V0[:,0]+0.5*V0[:,1]

psi0 = psi0/(norm(psi0))

# raise ValueError

figure(37)
plot(phi/(2*pi),psi0)
ylabel("$\psi_0$")
xlabel("$\phi/(2*pi)$")
# raise ValueError
#%%
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
            # print(norm(psi))
        psi = U0@psi 
        psi1 = U1@psi 
  
        N = norm(psi1)**2
        
        if N<r :
            
            
            
            t = 0# 1/2**n_it
            f = 0 
            converged = False 
            n_j = 0
            while f<(2**n_it):
    
                # print("x")
                mlist = zeros(n_it)
                for m in range(n_it-1,-1,-1):
                    # print(f)
                    if flist[m]+f<(2**n_it):
                        
                        
                        M = U1list[m]
                        psi2 = M @ psi
                        
                        if norm(psi2)**2>r:
                            t = t+tlist[m]
                            f = f + flist[m]
                            psi = psi2
   
                    
                # print("Hej")
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
                    
                    # print("jump")
                        
                    r = rand()
                    psi = psi/norm(psi)
                    
                    n_j+=1 
                    
            assert n_j>0
            
                
                            
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
            # for m in range(0,n_it):
            np+=1 
            #     if mlist[m]==0:
            #         psi = U1list[m]@psi 
                    
            #         t = t+tlist[m]
                    
    
            # print(f"{n_j} jumps at step {n}!")
            # print(t)
        else:
            psi = psi1     
            
                    
                    
                
                
                
                
                
                
                
                
        # print(toc())
        

        U12 = expm(-1j*Heff*dt_JJ_i/2)   
        psi = U12@psi
        psi = psi/norm(psi)         
    rho = rho+outer(psi,psi.conj())
    W,xo,po = wigner(psi,phi,pvec=pvec)
    # print(toc())
    Wo += W 
        
    nss +=1 
Wo = Wo/nss
rho = rho/nss
#%%
figure(19)
plot(phi,abs(psi)**2)

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

figure(20)
VMAX = amax(abs(Wo))
pcolormesh(xo,po,Wo,vmin=-VMAX,vmax=VMAX,cmap="bwr")
xticks(4*pi*nvec-pi,["$"+str(n*4-2)+"\pi$" for n in nvec])
yticks(0.5*nvec+.25,["$"+str(2*n+1)+"/4$" for n in nvec])
grid(linestyle="--",linewidth=0.5)
ylim(-2,2)
xlim((-12*pi,12*pi))
colorbar()
#%%    
figure(1)
tvec = arange(nx)
xvec = phi
phig,tg = meshgrid(xvec,tvec)
pcolormesh(phig/(2*pi),tg,X[:nx,:])

title("X")
figure(8)
# tvec = arange(nx)
# xvec = phi
# phig,tg = meshgrid(xvec,tvec)
pcolormesh(phig/(2*pi),tg,P[:nx,:])

title("P")
colorbar()
xlabel("$\phi/(2*pi)$")
# figure(2)
# pcolormesh(P)
# title("P")
# colorbar()
figure(221)
plot(tvec,WIlist[:nx])
plot(tvec,real(Clist[:nx])+3e-3)
plot(tvec,imag(Clist[:nx])+6e-3)
xlabel("periods")
ylabel("Interwell imbalance")
title("Decay of interwell imbalance (relaxation time)")
# plot(tvec,W2list[:np])
legend(["$\\langle \sigma_z\\rangle$","$\\langle \sigma_x\\rangle$","$\\langle\sigma_y\\rangle$"])



# W,xo,po = wigner(psi,phi)
# %%
tic()
figure(3)
Xg,Pg = meshgrid(xo,po)
print(toc())
VMAX = amax(abs(Wo))/5
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
