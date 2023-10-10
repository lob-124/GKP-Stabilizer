#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 17:06:30 2021

@author: frederiknathan

Moudule generating spectral functions and jump operators from gaussian baths. 
Core object is Bath.
"""

from matplotlib.pyplot import *
from numpy import *
import numpy.fft as fft
from numpy.linalg import *
import warnings
from basic import tic,toc,SX,SY,SZ#,derivative
# import vectorization as vect 
warnings.filterwarnings('ignore')
from scipy.interpolate import RectBivariateSpline
import sys

import numpy.random as npr 

RESOLUTION_FOR_INTERPOLATOR = 50
OUTPUT_STEPS = 100
def window_function(omega,E1,E2):
    """return window function which is 1 for E1 < omega < E2, and scale of smoothness set by E_softness
    its a gaussian, with center (E1+E2)/2, and width (E1-E2)
    """
    
    Esigma = 0.5*(E1-E2)
    Eav = 0.5*(E1+E2)
    X = exp(-(omega-Eav)**2/(2*Esigma**2))
    if sum(isnan(X))>0:
        raise ValueError
        
    return X
    
def S0_colored(omega,E1,E2,omega0=1):
    """
    Spectral density of colored noise (in our model)
    """
    A1 = window_function(omega, E1, E2)
    A2 = window_function(-omega, E1, E2)
    
    return (A1+A2)*abs(omega)/omega0

def get_ohmic_spectral_function(Lambda,omega0=1,symmetrized=True):
    """
    generate spectral function

    S(\omega) = |\omega| * e^{-\omega^2/2\Lambda^2}/\omega_0
    
    Parameters
    ----------
    Lambda : float
        Cutoff frequency.
    omega0 : float, optional
        Normalization. The default is 1.
    symmetrized : bool, optional
        indicate if the spectral function shoud be symmetric or antisymmetric. If False, |\omega| -> \omega in the definition of S. 
        The default is True.

    Returns
    -------
    S : method
        spectral function S,

    """
    if symmetrized :
            
        def f(omega):
            return abs(omega)*exp(-(omega**2)/(2*Lambda**2))/omega0
    
    else:
        def f(omega):
            return (omega)*exp(-(omega**2)/(2*Lambda**2))/omega0
        
    return f

def S0_ohmic(omega,Lambda,omega0=1):
    """
    Spectral density of ohmic bath
    
        S(omega) = |omega|*e^(-omega^2/(2*Lambda**2))
         
    """
    
    Out = abs(omega)*exp(-(omega**2)/(2*Lambda**2))/omega0
    
    
    return Out
def BE(omega,Temp=1):
    """
    Return bose-einstein distribution function at temperature temp. 
    """
    return (1)/(1-exp(-omega/Temp))*sign(omega)

def get_J_colored(E1,E2,Temp,omega0=1):
    """
    generate spectral function of colored bath at given values of E0,E1,Temp 
    
    Returns spectral function as a function/method
    """
    dw = 1e-12
    nan_value = S0_colored(dw,E1,E2,omega0=omega0)*BE(dw,Temp=Temp)
    def J(omega):
        
        return nan_to_num(S0_colored(omega,E1,E2,omega0=omega0)*BE(omega,Temp=Temp),nan=nan_value)
    
    return J

def get_J_ohmic(Temp,Lambda,omega0=1):
    """
    generate spectral function of ohmic bath, modified with gausian as cutoff,
    
    S(omega) = |omega|*e^(-omega^2/(2*Lambda**2))

    """
    def J(omega):
        out  = nan_to_num(S0_ohmic(omega,Lambda,omega0=omega0)*BE(omega,Temp=Temp),nan=Temp/omega0)
        if len(shape(omega))>0:
            out[where(abs(omega)<1e-14)] = Temp/omega0
        return out 
    
    return J 

def get_J_from_S(S,temperature,zv):
    """
    Get bath spectral function from bare spectral function at a given temperature. Zv specifies what value to give at zero (Where BE diverges)"""
    

    def out(energy):
        return nan_to_num(BE(energy,Temp = temperature)*S(energy)*sign(energy),nan=zv)
    
    return out

def get_g(J):
    """
    Get jump spectral function from given bath spectral function, J
    input:method
    output: method
    """
    
    def g(omega):
        return sqrt(abs(J(omega))/(2*pi))
    
    return g


def get_ft_vector(f,cutoff,dw):
    """
    Return fourier transform of function as vector, with frequency cutoff <cutoff> and frequency resolution <dw>
    Fourier transform, \int dw e^{-iwt} J(w)
    """
    
    omrange = linspace(-cutoff,cutoff,2*int(cutoff/dw)+1)[:-1]
    n_om  = len(omrange)
    omrange = concatenate((omrange[n_om//2:],omrange[:n_om//2]))
    
    vec    = fft.fft(f(omrange))*dw 
    # Jvec    = (-1)**(arange(0,n_om))*Jvec
    times   = 2*pi*fft.fftfreq(n_om,d=dw)
    AS = argsort(times)
    times = times[AS]
    vec = vec[AS]
    
    return times,vec
        
        
        
class bath():
    """
    bath object. Takes as input a spectral function. Computes jump correlator 
    and ULE timescales automatically. 
    Can plot correlation functions and spectral functions as well as generate 
    jump operators and Lamb shfit
    
    Parameters
    ----------
        J : callable.     
            Spectral function of bath. Must be real-valued
        cutoff : float, >0.    
            Cutoff frequency used to compute time-domain functions (used to 
            compute Lamb shift and ULE timescales, and for plotting correlation 
            functions).
        dw : float, >0.  
            Frequency resolution to compute time-domain functions (see above)
        
    Properties
    ----------
        J : callable.  
            Spectral function of bath. Same as input variable J
        g : callable.  
            Fourier transform of jump correlator (sqrt of spectral function)
        cutoff : float.    
            Same as input variable cutoff
        dw : float.     
            Same as input variable dw
        dt : floeat.    
            Time resoution in time-domain functions. Given by pi/cutoff
        omrange : ndarray(NW)    
            Frequency array used as input for computation of time-domain 
            observables (see above). Frequencies are in range (-cutoff,cutoff)
            and evenly spaced by dw. Here NW is the length of the resulting 
            array.
        times : ndarray(NW)     
            times corresponding to time-domain functions
        correlation_function : ndarray(NW), complex
            Correlation function at times specified in times. 
            Defined such that correlation_function[z] = J(times[z]).
        jump_correlator  :ndarray(NW), complex    
            Jump correlator at times specified in times 
        Gamma0 : float, positive.    
            'bare' Gamma energy scale. The ULE Gamma energy scale is given by 
            gamma*||X||*Gamma0, where gamma and ||X|| are properties of the 
            system-bath coupling (see ULE paper), and not the bath itself. 
            I.e. gamma, ||X|| along with Gamma0 can be used to compute Gamma.
        tau : float, positive.      
            Correlation time of the bath, as defined in the ULE paper.
        

    """
    
    def __init__(self,J,cutoff,dw):
        self.J = J
        self.g = get_g(J)
        
        self.cutoff = cutoff
        self.dw     = dw
        self.dt     = 2*pi/(2*cutoff)
        
        # Range of frequencies 
        self.omrange = linspace(-cutoff,cutoff,2*int(cutoff/dw)+1)[:-1]+dw/2

        self.times,self.correlation_function = self.get_ft_vector(self.J)
        Null,self.jump_correlator = self.get_ft_vector(self.g)
                
        
        

        self.g_int = sum(abs(self.jump_correlator))*self.dt 
        
        
        self.K_vec  = cumsum(self.correlation_function[::-1])[::-1]*self.dt
        K_int  = sum(abs(self.K_vec[self.times>=0]))*self.dt
        self.lambda_const = 4*K_int
        self.Gamma0 = 4*self.g_int**2
        self.tau = sum(abs(self.jump_correlator*self.times))*self.dt/self.g_int 
        self.dephasing_speed = 4*pi*self.J(0)
        self.GammaJtau = 4*sum((self.times*abs(self.correlation_function))[self.times>=0])*self.dt

    def plot_correlation_function(self,nfig=1):
        plot(self.times,abs(self.correlation_function))
        title(f"Correlation function (abs)")
        xlabel("Time")
        
    def plot_jump_correlator(self,nfig=1):
        """Plot jump correlator as a function of time, evaluated at times in self.times.
        """
        
        figure(nfig)
        
        plot(self.times,abs(self.jump_correlator))
        title(f"Jump correlator (abs)")
        xlabel("Time")
    def plot_K(self,nfig=1):
        """Plot jump correlator as a function of time, evaluated at times in self.times.
        """
        
        figure(nfig)
        
        plot(self.times,abs(self.K_vec))
        title(f"Antiderivative of correlation function (abs)")
        xlabel("Time")     
        
    def plot_spectral_function(self,nfig=2):
        """
        Plot spectral function, evaluated at frequencies in self.omrange.
        """
        figure(nfig)
 
        plot(self.omrange,self.J(self.omrange))
        title(f"Spectral function")
        xlabel("$\omega$")


    def get_ft_vector(self,f) :
        """
        Return fourier transform of function as vector, with frequency cutoff <cutoff> and frequency resolution <dw>
        Fourier transform, \int dw e^{-iwt} J(w)
        """
        cutoff= self.cutoff
        dw    = self.dw
        omrange = linspace(-cutoff,cutoff,2*int(cutoff/dw)+1)[:-1]
        n_om  = len(omrange)
        omrange = concatenate((omrange[n_om//2:],omrange[:n_om//2]))
        
        
        vec    = fft.fft(f(omrange))*dw 
        # Jvec    = (-1)**(arange(0,n_om))*Jvec
        times   = 2*pi*fft.fftfreq(n_om,d=dw)
        AS = argsort(times)
        times = times[AS]
        vec = vec[AS]
        if sum(isnan(vec))>0:
            self.vec = vec
            self.arg = f(omrange)
            # raise ValueError
        
        return times,vec
    
    def get_ule_jump_operator(self,X,H,return_ed=False):
        """
        Get jump operator for bath, associated with operator X and Hamiltonian H
        (all must be arrays)      

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        H : TYPE
            DESCRIPTION.

        Returns
        -------
        L : TYPE
            DESCRIPTION.


        """

        global Emat,Q 
        [E,V]=eigh(H)
        ND = len(E)
        Emat = outer(E,ones(ND))
        Emat = Emat.T-Emat
        self.Emat = Emat
         
        X_eb = V.conj().T @ X @ V
        
        # print(self.g(Emat))
        self.xyz = self.g(Emat)
        L = 2*pi*V@(X_eb *self.xyz )@(V.conj().T)
        L = L * (abs(L)>1e-13)
        
        if not return_ed:
            
            return L
        
        else:
            
            return L,[E,V]
        
        

        
        
        
    def get_cpv(self,f,real_valued=True):
        """ 
        Return Cauchy principal value of integral \int dw f(w)/(w-w0) 
        
        The integral is defined as
        
        Re ( \int dw f(w)Re ( 1/(w-w0-i0^+)))
        
        This is the same as 
        
        i/2 *  \int_-\infty^\infty dt f(t)e^{-0^+ |t|} sgn(t)
            
        where  f(t) =     \int d\omega f(\omega)e^{-i\omega t} 
        
        (i.e. get_time_domain_function(f))  
        
        """
        S0 = shape(f(0))
        nd = len(S0)
        Sw = (len(self.omrange),)+(1,)*(nd-1)
        wrange = self.omrange.reshape(Sw)
        # self.Sw = Sw
        vec1 = f(wrange)
        vec2 = f(-wrange)
        # self.vec1 = vec1
        # raise ValueError
        
        vec = 0.5*(vec1-vec2)/(wrange)
        dw = 1e-10
        self.vec12 = vec
        self.vec1 = vec1 
        self.vec2 = vec2
        
        # if 0 in wrange:
            

        if amin(abs(wrange))<1e-12:
            ind = where(abs(wrange)<1e-12)[0]
            
            vec[ind] = 0.5*(f(dw)-f(-dw))/dw
            
        
        # if sum(isnan(vec))+sum(isinf(vec)) or amax(abs(vec))>1e12:
        #     # self.f=f
        #     # self.vec  = vec
        #     # raise ValueError
        #     self.W = where(isnan(vec))
        #     # self.W2 = where(isinf(vec))
            
        #     assert prod(self.W[1]==self.W[2])
            # indices = concatenate((where(isnan(vec))[0],where(isinf(vec))[0]))
            # samples = concatenate((where(isnan(vec))[1],where(isinf(vec))[1:]))
            # self.indices = indices 
            # # self.samples = samples
            # for n in range(0,len(indices)):
                
            #     sample = samples[n]
            #     ind = indices[n]
            #     if ind>0 and ind<len(self.omrange):
                    
            #         vec[ind,sample] = (vec[ind+1,sample]+vec[ind-1,sample])/2
            #     elif ind==0:
            #         vec[ind,sample] = vec[ind+1,sample]
                
    
            #     else:
            #         vec[ind,sample] = vec[ind-1,sample]
                    
            # raise ValueError("Nan value encountered")
        
        out = sum(vec,axis=0)*self.dw 
 
        return out 
    
        # # times,F = get_time_domain_function(g,freqvec) 
        # Null,F = self.get_ft_vector(f)
        
        # # self.F = F
        
        # N = len(F)
        
        # F[N//2:]=-F[N//2:]
        # F[0]=0
        # # dt = tvec[1]-tvec[0]
        # S =-0.5j*sum(F)*self.dt
        
        # if real_valued:
        #     S=real(S)
            
        # if isnan(S):
        #     self.F=F
            
        #     # raise ValueError
        # return S
    # def get_amplitude_array(self,wmin,wmax,nw):
        
    #     def f(x):
    #         return self.g(x)*self.g(x+arg2)
    #     self.f = f
    #     return -2*pi*self.get_cpv(f,w0=arg1)                          
            
    def get_lamb_shift_amplitudes(self,q1list,q2list):
        """
        Get amplitude of lamb shift F_{\alpha \beta }(q1,q2) (see L)
        
        q1 and q2 must be 1d arrays of the same length
        """
        
        nq = len(q1list)
        assert len(q2list)==nq
        
        q1list = q1list.reshape(1,nq)
        q2list = q2list.reshape(1,nq)
        def f(x):
            return self.g(x-q1list)*self.g(x+q2list)
        self.f = f
        return -2*pi*self.get_cpv(f)    
    def create_lamb_shift_amplitude_interpolator(self,cutoff,resolution):
        assert(type(resolution)==int)
        
        global amplitudes,Evec,E1,E2
        Evec = linspace(-cutoff,cutoff,resolution)
        E1,E2 = meshgrid(Evec,Evec)
        amplitudes = self.get_lamb_shift_amplitudes(E2.flatten(),E1.flatten()).reshape(shape(E1))
        
        interpolator_r = RectBivariateSpline(Evec,Evec,real(amplitudes))
        interpolator_i = RectBivariateSpline(Evec,Evec,imag(amplitudes))        
        return Evec,amplitudes,interpolator_r,interpolator_i
       
    def get_lambda0(self):
        """
        Compute \Lambda_0 = \mathcal P \int dw J(w)/w
        """
        
        return self.get_cpv(self.J)
    
    def get_ule_lamb_shift_static_old(self,X,H):
        """Get ULE Lamb shift for a static Hamiltonian, using self.get_ft_vector to calculate cauchy p.v.
        
        The cpv calculation can definitely be parallelized for speedup.
        
        
        With a modified M operator (The one that makes the ULE calculation simpler), 
        the lamb shift calculation can be improved quite a lot!"""
             
        
        
        [E,U]=eigh(H) 
        D  = shape(H)[0]
    
        X_b = U.conj().T.dot(X).dot(U)
     
        LS_b = zeros((D,D),dtype=complex)
        
        
        # g = get_jump_correlator(J)
        print("Computing Lamb shift")
        n_it = 0
        n_output = max(1,D**2//OUTPUT_STEPS)
        for m in range(0,D):
            # if m%(max(1,D//10))==0:
            #     print(f"    At column {m}/{D}")
            tic()
            for n in range(0,D):
                n_it+=1
                if n_it %n_output == 0:
                    print(f"at step {n_it}/{D**2}. Time spent: {toc():.2f}")
                    sys.stdout.flush()

                E_mn = E[m]-E[n]
                E_nl_list = E[n]-E
                E_mn_list= E_mn*ones(len(E))
                # for l in range(0,D):
                #     E_nl = E[n]-E[l]
                    
                Amplitudes = self.get_lamb_shift_amplitudes(E_mn_list,E_nl_list)
                self.Amplitudes=Amplitudes
                LS_b[m]+=Amplitudes*X_b[m,n]*X_b[n,:]
        
        LS = U.dot(LS_b).dot(U.conj().T)
       
        
        return LS 
    
    def get_ule_lamb_shift_static(self,X,H):
        """Get ULE Lamb shift for a static Hamiltonian, using self.get_ft_vector to calculate cauchy p.v.
        
        The cpv calculation can definitely be parallelized for speedup.
        
        
        With a modified M operator (The one that makes the ULE calculation simpler), 
        the lamb shift calculation can be improved quite a lot!"""
             
        
        
        [E,U]=eigh(H) 
        D  = shape(H)[0]
    
        X_b = U.conj().T.dot(X).dot(U)
     
        LS_b = zeros((D,D),dtype=complex)
          
        Emin = amin(E)
        Emax = amax(E)
        
        cutoff = 1.05*(Emax-Emin)
        resolution = RESOLUTION_FOR_INTERPOLATOR
        # print("set resolution to 150")
        global Q_r,Q_i,Eab,Ev,Evec
        Evec,Values,Q_r,Q_i = self.create_lamb_shift_amplitude_interpolator(cutoff, resolution)      
        
        # g = get_jump_correlator(J)
        print("Computing Lamb shift")
        n_it = 0
        n_output=max(1,D**2//OUTPUT_STEPS)
         
        for m in range(0,D):
            # if m%(max(1,D//10))==0:
            #     print(f"    At column {m}/{D}")
            for n in range(0,D):
                n_it+=1
                if n_it %n_output == 0:
                    print(f"at step {n_it}/{D**2}")
                E_mn = E[m]-E[n]
                E_nl_list = E[n]-E
                E_mn_list= E_mn*ones(len(E))
                # for l in range(0,D):
                #     E_nl = E[n]-E[l]
                    
                # Amplitudes = self.get_lamb_shift_amplitudes(E_mn_list,E_nl_list)
                Amplitudes = Q_r(E_mn_list,E_nl_list,grid=False)+1j*Q_i(E_mn_list,E_nl_list,grid=False)

                self.Amplitudes=Amplitudes
                LS_b[m]+=Amplitudes*X_b[m,n]*X_b[n,:]
        
        LS = U.dot(LS_b).dot(U.conj().T)
       
        
        return LS 
    
    def get_c_amplitude(self,E1,E2):
        """
        E1 = p_m
        E2 = p_n

        Parameters
        ----------
        E1 : TYPE
            DESCRIPTION.
        E2 : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        N1,N2 = len(E1),len(E2)
        
        E1 = E1.reshape(N1,1)
        E2 = E2.reshape(1,N2)
    
        DE = E1-E2
        
        # print("measuring")
        # print(shape(E1))
        # print(shape(E2))
        tic()
        Im = 2*pi**2 * self.g(E2)*(self.g(E2)-self.g(E1))/(E1-E2)
        Ind = where(isnan(Im) | isinf(Im))
        self.Ind = Ind 
        self.Im = Im
        # print(toc())
        self.E1d = E1[Ind[0]]
        Im[Ind] =-2*pi**2*self.g(E1[Ind[0],0])*derivative(self.g)(E1[Ind[0],0])
        self.I0 = where(isnan(Im[Ind]))
        self.E1 = E1 
        self.E2 = E2 
        
        # print(toc())        
        
        assert not sum(isnan(Im[Ind]))
        E1w = E1.reshape(1,N1,1)
        E2w = E2.reshape(1,1,N2)        
        
        def f(omega):
            return  self.g(omega+E2w)*(self.g(omega+E2w)-self.g(omega+(E1w)))/(E1w-E2w)
        
        self.f=  f
        # print(shape(f(0)))
        R = self.get_cpv(f)
        # print(toc())      
        Ed = E1[Ind[0]].reshape(1,len(Ind[0]))
        self.Ed = Ed 
        def fd(omega):
            return -self.g(omega+Ed)*derivative(self.g)(omega+Ed)
        self.fd=fd
        
        R_d = self.get_cpv(fd)
        
        
        self.out_d = R_d
        R[Ind[0],Ind[1]] = R_d
        
        R = 2*pi*R   
        
        self.R= R
        self.Im = Im
        out = R+1j*Im
        # print(toc())
        # print("Done\n")
        assert not sum(isnan(out))
        return out
    
    def create_c_amplitude_interpolator(self,cutoff,resolution):
        assert(type(resolution)==int)
        
        Evec = linspace(-cutoff,cutoff,resolution)
        
        X = self.get_c_amplitude(Evec,Evec)
        
        interpolator_r = RectBivariateSpline(Evec,Evec,real(X))
        interpolator_i = RectBivariateSpline(Evec,Evec,imag(X))        
        return Evec,X,interpolator_r,interpolator_i
    
        
        
     
    def apply_M_operator(self,X,H,rho,adjoint=False):
        """ 
        get M[rho]. Assuming rho is hermitian. 
        """ 

        [E,U]=eigh(H)
        assert amax(abs(rho-rho.conj().T))<1e-8
        # UU = vect.lm(U)@vect.rm(U.conj().T)
        X = U.conj().T@X@U
        rho = U.conj().T@rho@U 
        
        Emin = amin(E)
        Emax = amax(E)
        
        cutoff = 1.05*(Emax-Emin)
        resolution = RESOLUTION_FOR_INTERPOLATOR
        global Q_r,Q_i,Eab,Ev,Evec
        Evec,Values,Q_r,Q_i = self.create_c_amplitude_interpolator(cutoff, resolution)
        
        
        
        # self.X = X 
        # self.Xb = Xb
        dim = len(E)

        
        E=-E
        
        Earray = E.reshape(dim,1)
        Earray = Earray - Earray.T
        
        Ev  = Earray.flatten()

        # Ev[a,b] = E_a-E_b 
        # c_mat = zeros((dim,)*4,dtype=complex)
        # E_mat = zeros((dim,)*4,dtype=complex)
        
        drho = 0*rho 
        
        n_it = 0
        n_output = max(1,dim**2//OUTPUT_STEPS)
        tic()
        for a in range(0,dim):
            for b in range(0,dim):
                n_it +=1 
                
                if n_it%n_output==0:
                    print(f"at step {n_it}/{dim**2}. Time spent: {toc():.2f} s")
                Eab = array([E[a]-E[b]])
                c_vec = Q_r(Eab,-Ev,grid=False)+1j*Q_i(Eab,-Ev,grid=False)
    
                cmat2  = 1*c_vec.reshape((dim,)*2)
                Xr = cmat2*X.conj().T
                
                if not adjoint:
                        
                    drho[a,:] += X[a,b]*rho[b,:]@Xr 
                    drho[:,b] -= rho@Xr[:,a]*X[a,b]
                
                else:
                    drho[b,:] += (X[b,a]*rho[a,:]@Xr.conj().T)
                    drho      -= outer(rho[:,b]*X[b,a],((Xr.conj().T)[a,:]))
        drho = drho + drho.conj().T 
        
        drho = U@drho@U.conj().T
        return drho 
    
                
                
        """ 
        c_mat[m,n,k,l] = 2*pi*c_{mn;kl}
        """ 
        
        # self.c_mat = c_mat
        # self.E = E 
        # for i in range(0,20):
        #     a,b,c,d = [npr.randint(dim) for n in range(0,4)]
            
        #     self.a = a 
        #     self.b = b
        #     self.c = c
        #     self.d = d
        #     self.x = c_mat[a,b,c,d]
        #     self.y = self.get_c_amplitude(array([E[a]-E[b]]),array([E[d]-E[c]]))
        #     self.diff = self.x-self.y
        #     assert abs(self.diff)<1e-10
            
            
        # Xsq    = tensordot(Xb,Xb,axes=0)
        
        # for i in range(0,20):
        #     a,b,c,d = [npr.randint(dim) for n in range(0,4)]
        #     assert Xsq[a,b,c,d] == Xb[a,b]*Xb[c,d]
        #     # self.a = a 
        #     # self.b = b
        #     # self.c = c
        #     # self.d = d
        #     # self.diff = c_mat[a,b,c,d]-self.get_c_amplitude(array([E[a]-E[b]]),array([E[d]-E[c]]))
        #     # assert abs(c_mat[a,b,c,d]-self.get_c_amplitude(array([E[a]-E[b]]),array([E[d]-E[c]])))<1e-10
        
        # print("cmat set to 1 here!!")
        # self.cc = c_mat
        # # c_mat = 1 
        
        # Q      = c_mat * tensordot(Xb,Xb,axes=0)
        
        # self.Earray = Earray
        # self.E = E 
        # self.c_mat = c_mat 
        
        # self.Q=Q
        
        # M      = Q.swapaxes(1,2).swapaxes(1,3)-tensordot(eye(dim),trace(Q,axis1=0,axis2=3),axes=0).reshape((dim,)*4).swapaxes(1,2)
        # M      = M + M.conj().swapaxes(0,1).swapaxes(2,3)
        # for i in range(0,100):
        #     r1,r2,s1,s2 = [npr.randint(dim) for n in range(0,4)]
            
        #     diff  = abs(M[r1,r2,s1,s2] -( Q[r1,s1,s2,r2]-trace(Q[:,r2,s2,:])*(r1==s1)+Q[r2,s2,s1,r1].conj()-trace(Q[:,r1,s1,:]).conj()*(r2==s2)))
        #     # self.i=i
            
        #     # diff = M[r1,r2,s1,s2] -  (Q[r1,s1,s2,r2]-trace(Q,0,3)[r2,s2]*(r1==s1)+Q[r2,s2,s1,r1].conj()-trace(Q,0,3)[r1,s1].conj()*(r2==s2))
        #     self.diff = diff
            
        #     assert abs(diff)<1e-10
        # M = M.reshape((dim**2,dim**2))
        
        # M = UU@M@UU.conj().T 

        
        # return M        
    def get_M_operator(self,X,H):
    
        [E,U]=eigh(H)
        UU = vect.lm(U)@vect.rm(U.conj().T)
        Xb = U.conj().T@X@U
        self.X = X 
        self.Xb = Xb
        dim = len(E)

        
        E=-E
        
        Earray = E.reshape(dim,1)
        Earray = Earray - Earray.T
        
        Ev  = Earray.flatten()

        # Ev[a,b] = E_a-E_b 
        c_mat = zeros((dim,)*4,dtype=complex)
        E_mat = zeros((dim,)*4,dtype=complex)
        for a in range(0,dim):
            for b in range(0,dim):
            
                Eab = array([E[a]-E[b]])

                ## E1 = E_{ab}
                ## E2 = E_{cd}

                c_vec  = self.get_c_amplitude(Eab,-Ev)
            
                c_mat[a,b,:,:] = 1*c_vec.reshape((dim,)*2)
        self.c_mat = c_mat
        self.E = E 
        for i in range(0,20):
            a,b,c,d = [npr.randint(dim) for n in range(0,4)]
            
            self.a = a 
            self.b = b
            self.c = c
            self.d = d
            self.x = c_mat[a,b,c,d]
            self.y = self.get_c_amplitude(array([E[a]-E[b]]),array([E[d]-E[c]]))
            self.diff = self.x-self.y
            assert abs(self.diff)<1e-10
            
            
        Xsq    = tensordot(Xb,Xb,axes=0)
        
        for i in range(0,20):
            a,b,c,d = [npr.randint(dim) for n in range(0,4)]
            assert Xsq[a,b,c,d] == Xb[a,b]*Xb[c,d]
            # self.a = a 
            # self.b = b
            # self.c = c
            # self.d = d
            # self.diff = c_mat[a,b,c,d]-self.get_c_amplitude(array([E[a]-E[b]]),array([E[d]-E[c]]))
            # assert abs(c_mat[a,b,c,d]-self.get_c_amplitude(array([E[a]-E[b]]),array([E[d]-E[c]])))<1e-10
        
        print("cmat set to 1 here!!")
        self.cc = c_mat
        # c_mat = 1 
        
        Q      = c_mat * tensordot(Xb,Xb,axes=0)
        
        self.Earray = Earray
        self.E = E 
        self.c_mat = c_mat 
        
        self.Q=Q
        
        M      = Q.swapaxes(1,2).swapaxes(1,3)-tensordot(eye(dim),trace(Q,axis1=0,axis2=3),axes=0).reshape((dim,)*4).swapaxes(1,2)
        M      = M + M.conj().swapaxes(0,1).swapaxes(2,3)
        for i in range(0,100):
            r1,r2,s1,s2 = [npr.randint(dim) for n in range(0,4)]
            
            diff  = abs(M[r1,r2,s1,s2] -( Q[r1,s1,s2,r2]-trace(Q[:,r2,s2,:])*(r1==s1)+Q[r2,s2,s1,r1].conj()-trace(Q[:,r1,s1,:]).conj()*(r2==s2)))
            # self.i=i
            
            # diff = M[r1,r2,s1,s2] -  (Q[r1,s1,s2,r2]-trace(Q,0,3)[r2,s2]*(r1==s1)+Q[r2,s2,s1,r1].conj()-trace(Q,0,3)[r1,s1].conj()*(r2==s2))
            self.diff = diff
            
            assert abs(diff)<1e-10
        M = M.reshape((dim**2,dim**2))
        
        M = UU@M@UU.conj().T 

        
        return M
        
        





if __name__ == "__main__":
    import numpy.random as npr 
    import units as u
    
    Temp = 1
    omega_c=  50*Temp
    # def J0(omega):
    #     return omega**2*(omega>0)

    
    J0 = get_J_ohmic(Temp,omega_c)
    
    B0 = bath(J0,10*omega_c,10*omega_c/4000)
    
    H = SZ 
    X = SX+0.2*SY
    Y = SX+0.5*SZ+0.3*SY
    rho = eye(2,dtype=complex)/2+0.2*SZ
    M = B0.get_M_operator(X,H)
    
    # d1 =vect.vec_to_mat(M@vect.mat_to_vec(rho))
    d2 = B0.apply_M_operator(X, H, rho)       
    d1 = B0.apply_M_operator(X,H,Y,adjoint=True)
    # L1 = B0.get_ule_lamb_shift_static_old(X, H)
    # L2 = B0.get_ule_lamb_shift_static(X, H)    
    # Evec,X,Q_r,Q_i  = B0.create_c_amplitude_interpolator(100,100)
    
    # Z = arange(-10,10.,1)
    # Z1,Z2 = meshgrid(Z,Z)
    
    # Qmat = Q(Z1.flatten(),Z2.flatten())#.reshape(shape(Z1))