#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 15:21:46 2023

Try to make clever basis for GKP protocol
@author: qxn582

Convention: Psi is already converted to matrix. psi[m,k] gives well m, rung k. 

As a function of \phi, the state |m,k> has wavefunction 

    \langle \phi|m,k>  =  \psi_k(\phi-(1/2+m)*\well_spacing)

The states {|m,k>} are not orthogonal. We therefore consider the overlap matrix 

    V_{ab}^{k-l} = <k,a|l,b>

(note that V by construction is symmetric under translations in well index)

If the states were orthogonal we would have V_{ab}^k = \delta_{ab}\delta_{k0}

For a,b<<1/\sigma^2,  states |k1,b>,|k2,a>  are well confined in wells extremely close to being orthogonal for k1\neq k2.
I.e, 

    V_{ab}^k \approx  \delta_{ab}\delta_{k0},   for a,b<<1/\sigma^2,
    
the residual correction is extremely small, of order e^{1/\sigma^2}. 

However, we might climb up the ladder, and want to capture the physics of states beginning to transistion between wells.
Therefore, we let the code take the non-orthogonality of the states |l,b> into account. 

We want to find an orthogonal basis spanning the space spanned by the well states we include in the sijulations. 
This would make it much simpler to implement the ULE etc. 

To this end, we apply a useful trick for finiding an orthogonal basis of states:
For now, let us neglect the multi-index notation, to keep notation simple

For a set of non-orthogonal states {|x>}, we have that the overlap matrix 

    M_{xy} = <x|y> 
    
is nonzero. 

M is Hermitian and positive-semidefinite. 

We can therefore take any real power of M without any ambiguity. 

Consider the states 

    |x>> \equiv \sum_{x}(M^{-1/2})_{yx} |y>.

We have 

    <<x|y>>     = \sum_{c,d} (M^{-1/2})^*_{cx}<c|d> (M^{-1/2})_{dy}
                = \sum_{c,d} (M^{-1/2})^*_{cx}M_{cd} (M^{-1/2})_{dy}
                = \sum_{c,d} (M^{-1/2})_{xc}M_{cd}M^{-1/2})_{dy}
                = [M^{-1/2} * M * M^{-1/2}]_{xy}
                = \delta_{xy}
            
In the third line we exploited the hermiticity of M^{-1/2}.

In other words, the collumns of the matrix G = 1/\sqrt{M^T} correspond to orthogonal vectors. 


In our simulations, we want to work in an orthonormal basis), we use the basis states

|n,k>> to represent our matrices and vectors.

The basis states can be obtained as follows: 
    
    |m,a>> = \sum_{l} G^{m-l}_{a,b} |l,b>.
    
where \sum_{k,l} [G^{-k}_{ba} * G^{-l}_{cb} * V_{cd}^(m-k-l) = \delta_{bd}* \delta_{m0}).
             
I.e., G = [V^{-1/2}]^T

where V[nrungs*k:nrungs*(k+1),nrungs*l:nrungs*(l+1)] = V^{(k-l)}.
and   G[nrungs*k:nrungs*(k+1),nrungs*l:nrungs*(l+1)] = G^{(k-l)}.

We can construct the matrix elements <m,a| Mat |n,b> (for Hamiltonian and Fourier transform matrix) easily, either using analytical
expressions, or numerical integrations. 


The matrix elements of the Hamiltonian in the new basis is given by 

<<m,a|H|n,d>> = G^{m-l}_{ab}Hmat_{bc}^{l,k} G^{k-n}_{cd} (*)

The current version of the code can find the states {|m,a>>} (the .get_orthonormal_state(a) function does the job)
It can also find the matrix elements of the Hamiltonian in the original non-orthogonal well basis, <m,a|H|n,d>. 
However, I have not yet implemented the function returning the matrix elements of the Hamiltoinan in the orthonormal basis, <<m,a|H|n,d>>.
I hope to do so after the sep 3rd deadline. 

Until then, we can simply use the matrix elemetns <m,a|H|n,d> to represent the Hamiltonian, as the well states are any way very close to orthogonal for small a,d (which is where we want to find ourselves anyway)

Alternatively @Liam can implement the orthogonal-basis represeentation of the Hamiltonian, using the formula (*) above

under __main__ plots the wavefunction of the orthogonal states (as a function of phi), and returns the blocks of the Hamiltonian in the well state basis. 


"""
from basic import *
from numpy import *
from numpy.linalg import *
from matplotlib.pyplot import *
from scipy.special import factorial,hermite
from units import *
from gaussian_bath import bath,get_J_ohmic
# from wigner import wigner 
from basic import tic,toc
from scipy.interpolate import interp1d
from numpy.random import rand
import scipy.sparse as sp
import scipy.sparse.linalg as spla 
from scipy.special import hermite
from numpy.random import randint 
from numpy.random import rand

# ----------------------------------------------------------------------------
# 1. Parameters 

quantization_parameter  = 1
D        = 100       # Dimension of Hilbert space for computing matrix elements
n_wells  = 10
Dmax     = 1600

assert n_wells%2 == 0 
 
Josephson_energy =200*GHz 
cavity_frequency = 2*GHz

omega_c  = 0.1*THz         # Bath cutoff frequency
gamma    = 5e-3*meV      # Bare system-bath coupling. Translates to a resistance. 
Temp     = 1*GHz#2e-3*Kelvin   # Bath temperature 
omega0   = 1*meV         # Normalizing frequency that enters in ohmic spectral function


DT = abs(30000*2*pi/Josephson_energy)

# Derive physical quantities from parameters
impedance   = planck_constant/(2*quantization_parameter*e_charge**2)
L           = impedance/cavity_frequency
C           = 1/(cavity_frequency*impedance)
omega_jj = 1/sqrt(abs(Josephson_energy)/(flux_quantum/(4*pi))**2*C)
inductance  = 1*L

### Phi grid 

phi         = linspace(-pi,pi,D)
dphi        = phi[1]-phi[0]

### canonically conjugate operators 

# dimensionless phase 
Phi     = diag(phi)
X1 = (expm(1j*Phi))
X2 = expm(-1j*Phi)

# canonically conjugate momentum to Phi
Tmat    = get_tmat(D,dtype=complex)
Tmat[0,-1]=1
Id = eye(D,dtype=complex)
Pi0     = 1j*(Tmat-eye(D))/dphi

# squared momentum
Pi_2 = 0.5*(Pi0@Pi0.conj().T + Pi0.conj().T@Pi0)

# Squared charge operator 
Q2 = ((2*e_charge)/hbar)**2 * Pi_2 

class LCJ_circuit():
    def __init__(self,Inductance,Capacitance,Josephson_energy,grid_shape,phi_resolution = 300):
 
        self.C = Capacitance
        self.L = Inductance
        self.J = Josephson_energy
        self.omega = 1/sqrt(self.L*self.C)
        self.phi_resolution = phi_resolution
        
        
        self.r = 1 + flux_quantum**2/(4*pi**2*self.L*self.J) # well spacing ratio
        self.well_spacing = 2*pi/self.r 
        self.sigma = (4*pi*hbar*self.omega/self.J)**(1/4)
     
        ### Phi grid 
        self.phi         = linspace(0,self.well_spacing,self.phi_resolution+1)[:-1]
        self.dphi        = self.phi[1]-self.phi[0]
        self.phi         = self.phi + self.dphi/2
        self.shape  = grid_shape
        self.nwells = grid_shape[0]
        self.nrungs = grid_shape[1]
        
        # dimensionless phase 
        self.Phi     = diag(self.phi)
        self.X1 = (expm(1j*self.Phi))
        self.X2 = expm(-1j*self.Phi)
    
        # canonically conjugate momentum to Phi
        self.Tmat    = get_tmat(self.phi_resolution,dtype=complex)
        self.Tmat[0,-1]=1
        self.Id = eye(self.phi_resolution,dtype=complex)
        self.Pi0     = 1j*(self.Tmat-self.Id)/self.dphi
    
        # squared momentum
        self.Pi_2 = 0.5*(self.Pi0@self.Pi0.conj().T + self.Pi0.conj().T@self.Pi0)
    
        # Squared charge operator 
        self.Q2 = ((2*e_charge)/hbar)**2 * self.Pi_2 
        
        self.D = self.nwells*self.nrungs    
       

        assert self.nwells%2 == 0,"nwells must be an even integer"
        assert self.nrungs <= self.phi_resolution, f"nrungs must be smaller than phi_resolution used to compute matrix elements ({self.phi_resolution})"
               
        
        # Compute the matrices {V^{(k)}_{ab}}
        self.overlap_matrices = self.compute_overlap_matrices()
        assert self.overlap_matrices.shape[0]%2 == 1
        
        # Compute the range of k for which V^{(k)}_{ab} is effectively nonzero. 
        # The n0th element of self.overlap_matrices corresponds to V^0.
        
        self.n0 = (shape(self.overlap_matrices)[0]-1)//2 
        
        self.G_matrices = self.compute_inverse_squareroot_transpose_of_overlap()
        
        self.n0g = (shape(self.G_matrices)[0]-1)//2 
        self.well_annihilation_operator = get_annihilation_operator(self.nrungs)
        
        
    def compute_inverse_squareroot_transpose_of_overlap(self):
        """
        Compute block collumns of X = V^{-1/2} 
        
        where V[nrungs*k:nrungs*(k+1),nrungs*l:nrungs*(l+1)] = V^{(k-l)}.
        
        

        Returns
        -------
        mat : ndarray(nblocks,nrungs,nrungs)
            Array of matrices encoding X, such that mat[n0+k,:,:] gives the kth off diagonal block of X, i.e. mat[n0+k-l,:,:]X[nrungs*k:nrungs*(k+1),nrungs*l:nrungs*(l+1)]
            here n0 = (nblocks-1)/2, and nblocks is guaranteed to be odd. 

        """
        mat = get_matrix_list_power(self.overlap_matrices, -1/2, 171)
        
        # Test that the computation was succesful
        Y = apply_matrix_list(mat,apply_matrix_list(mat,self.overlap_matrices,expand_shape=True),expand_shape=True)
        n0 = (shape(Y)[0]-1)//2
        Y[n0] -= eye(shape(Y)[1])
        
        assert(norm(Y)<1e-7),"Inverse square root algorithm did not converge. Try to decrease number of rungs. "
        
        mat = transpose_matrix_list(mat)
        return mat
    
    
    def get_G_matrix(self,k):
        """
        Return the kth G matrix, G^k_{ab} (see documentation of module)

        Parameters
        ----------
        k : int
            integer k of the matrix.

        Returns
        -------
        G: ndarray(nrungs,nrungs)
            matrix  G^k_{ab}.

        """
        if k>self.n0g:
            return zeros((nrungs,nrungs))
        else:
            
            return 1*self.G_matrices[self.n0g+k]
    

    def get_V_matrix(self,k):
        """
        Return the kth overlap matrix, V^k_{ab} (see documentation of module)

        Parameters
        ----------
        k : int
            integer k of the matrix.

        Returns
        -------
        V: ndarray(nrungs,nrungs)
            matrix  V^k_{ab}.

        """
        if abs(k)<= self.n0:
            
            return 1*self.overlap_matrices[self.n0+k]
        else:
            return zeros((self.nrungs,self.nrungs))
        
    def get_orthonormal_state(self,a):
        """
        Return the orthonormal state |0,a>>, represented in the well basis
        
        (other orthonormal states with same a
                                             are related by translations)

        Parameters
        ----------
        a : int
            index of state in dressed well basis.

        Returns
        -------
        psi : ndarray(nblocks,nrungs)
              state psi in well basis:
                
        |m,a>> = \sum_k psi[n0+k,a,b]|m+k,b>
        
        with n0 = (shape(psi)[0]-1)/2 '
        n0 is guaranteed to be odd.
        
        """
        psi = 1*self.G_matrices[:,a,:]
        return psi
    
        
    def compute_H_LC_block(self,k,l):
        """
        Generate Hamiltonian matrix of the HO in well k
        
        returns mat, where 
        
        <a,k|H_LC|b,l> = mat[a,b] 
         
        with 
        
            H_LC = (q^2/2C) + (1/2L) (phi \varphi_0/2\pi)^2
        
        with \varphi_0 the flux quantum. Our computation uses that
        
            H_LC = -E_C\partial_\phi^2  + E_L (phi/2\pi)^2 
        
        With E_C = 2e^2/C, E_J = \varphi_0^2/2L. 
        
        Now, the annihilation operator of well k is given by 
        
        b = 1/sqrt(2)*((\phi-[k+1/2]*Delta phi])/sigma + \sigma*\partial_\phi)
        
        so 
        
        phi            =  (sigma/sqrt(2))*(b+b^\dagger) + k*well_index*(k+1/2)*Delta phi 
        \partial_phi   =  1/(sqrt(2)*sigma)*(b-b^dagger)

        
        Thus, 
        
        phi|b,k> = [(sigma/sqrt(2))*(b+b^\dagger) + k*well_index]^2}|b,k>
                  
        with b|k,a> = sqrt(a)|k,a-1>.
        
        and so on.
        
        
        ----------
        k : int
            Well index .


        Returns
        -------
        mat : ndarray(nrungs,nrungs)
            Matrix mat.
        
        """
        well_annihilation_operator = get_annihilation_operator(self.nrungs+2)
        
        # phi observable represented through creation/anihilation operators of well k 
        Phi = (k+1/2)*eye(self.nrungs+2)*self.well_spacing + self.sigma/sqrt(2)*(well_annihilation_operator+well_annihilation_operator.T)
      
        # d/dphi observable represented through creation/anihilation operators of well k '
        dPhi = 1/sqrt(2*self.sigma)*(well_annihilation_operator-well_annihilation_operator.T)
        
        H0  =  -e_charge**2/(2*self.C)*(dPhi@dPhi) + 1/(2*self.L)*(flux_quantum/(2*pi))**2 * (Phi@Phi)
        
    
        assert(norm((H0-H0.T))<1e-8)
        """ 
            \hat{H}_LC |k,b> = \sum_a [H0]_{ab}|k,a>
        
        Thus 
        
            mat[a,b] \equiv <k,a|hat{H}_LC |l,b> 
                     =      \sum_c [H0]_{cb}<k,a|l,c>
                     =      \sum_c [H0]_{cb}V^{k-l}_{a-c}
                     =      [V^{k-l} * H0]_{ab}
            
        so 
        
            mat = V^{k-l} * H0 
        """ 
    
        mat = self.compute_overlap_matrix(k-l,nmax=self.nrungs+2)@H0
        mat = mat[:self.nrungs,:self.nrungs]    
        return mat
        
    def compute_H_JJ_block(self,k,l,dphi=1e-4,phi_max_seed = 2*pi):
        """
        Generate Hamiltonian matrix of the JJ in well k
        
        returns mat_c, where 
        
         <a,k|H_JJ|b,l> = \sum_{a} mat_c^(k-l)[a,b] 
         
        with 
        
        H_JJ = E_J * cos(phi)
        
        w
        ----------
        k : int
            Well index .
  
  
        Returns
        -------
        mat: ndarray(nrungs,nrungs), float
            matrix mat_c, as defined above.
        
        """
     
        phimax = 1*phi_max_seed 
        shift = (k-l) * self.well_spacing
        
        # iterate over phi cutoff and see if it converges. If not increase phi ctuoff 
        nit = 0 
        while True:
                 
            phivec = arange(-phimax,phimax,dphi)
            NP = len(phivec)
            
            phivec = phivec-phivec[NP//2]+self.well_spacing*(l+1/2)
            
            dphi = phivec[1]-phivec[0]
            
            Psi1Mat = zeros((self.nrungs,NP))
            Psi2Mat = zeros((self.nrungs,NP))
            
            for n in range(0,self.nrungs):
                f = self.get_well_eigenfunction(n)
                Psi1Mat[n] = f(phivec-self.well_spacing*(l+1/2))
                Psi2Mat[n] = self.J*cos(phivec)*f(phivec-self.well_spacing*(k+1/2))
                
            if sum(Psi1Mat[:,int(NP*0.9):]**2)*dphi < 1e-8:
                out = Psi1Mat@Psi2Mat.T*dphi
                return out  
            
            else:
                # increasing phimax to {phimax*1.2}
                phimax = phimax * 1.2
                nit +=1 
                
            if nit>20:
                raise ValueError("Did not converge")

        
    def get_H_block(self,k,l,dphi=1e-4,phi_max_seed =2*pi):
        """
        Generate Hamiltonian matrix of the JJ in well k
        
        returns mat, where 
        
         <a,k|H|b,l> = \sum_{a} mat[a,b] 
         
        with 
        
        H = H_JJ + H_LC 
    
        ----------
        k,l : int
            Well indices .
  
  
        Returns
        -------
        mat: ndarray(nrungs,nrungs), float
            matrix mat, as defined above.
        
        """       
        
        H_LC = self.compute_H_LC_block(k,l)
        H_JJ = self.compute_H_JJ_block(k,l,dphi=1e-4,phi_max_seed =2*pi)
        
        H = H_LC + H_JJ 
        return H

    def get_well_eigenfunction(self,n):
        """
        Get wavefunction of nth eigenstate of well harmonic oscillator (with 
        vacuum fluctuation length self.sigma).

        Parameters
        ----------
        n : int
            index of eigenstate.

        Returns
        -------
        f : callable
            wavefunction of eigenstate.

        """
        sig = self.sigma 
        def f(phi):
            out = ho_psi(n,phi/sig)/sqrt(sig)
            return out 
        return f 
    
    def get_wavefunction(self,psi):
        sig = 1*self.sigma 
        ws  = 1*self.well_spacing
        
        nwells = shape(psi)[0]
        n0     = (nwells-1)//2 
        nr = 1*self.nrungs

        well_list = arange(nwells)-n0
        def f(phi):
            global phimat 
            phimat = array([phi - (k+1/2)*ws for k in well_list])      
            out = zeros(shape(phi))
            for n in range(0,nr):
                
                vec = ho_psi(n,phimat/sig)/sqrt(sig)
                # print(shape(vec))
                # print(shape(psi))
                out += psi[:,n].T@vec
                
            return out 
        return f  
    def get_orthonormal_state_wavefunction(self,n):
        psi = self.get_orthonormal_state(n)
        f = self.get_wavefunction(psi)
        return f 
    
    
    def compute_overlap_matrices(self,max_overlap_range = 100, threshold=1e-10):
        """ 
        Return list of overlap matrices {V^k_{ab}} (see function get_overlap_matrix)
        """ 
        out = zeros((2*max_overlap_range+1,self.nrungs,self.nrungs))
        n0 = max_overlap_range
        
        for n in range(0,max_overlap_range):
            mat1 = self.compute_overlap_matrix(n)
            mat2 = self.compute_overlap_matrix(-n)
            
            if norm(mat1)>threshold:
                
                out[n0+n]  = mat1 
                out[n0-n]  = mat2 
            else:
                overlap_range = n-1
                break
    
        out = out[n0-overlap_range:n0+overlap_range+1]
        return out 
         
    def compute_overlap_matrix(self,k,nmax=0,dphi=1e-4,phimax=2*pi):
        """
        Get overlap matrix 
        
        V_{mn}^(k) = <a+k,m|a,n> 
                   = \int d\phi \psi_m(\phi+k*lambda)*\psi_n(\phi)
        
        with k = self.well_spacing the well spacing. 
        I.e., compute the overlap
        
        V_{mn}^{(k)} 
        
        where \{|a,n>\} denote the grid basis states. 
    
        The overlap is computed through direct numerical integration.

        Parameters
        ----------
        k : int
            displacement of wavefunction.
        dphi : float, optional
            phi discretizatin used for calculation. The default is 1e-4.
        phimax : float, optional
            seed phi range used for calculation. The default is 2*pi. Automatically
            adjusted by algorithm to ensure convergence. 

        Returns
        -------
        out : ndarray(N,N), float, where N = self.nrungs
            overlap matrix, such that out[m,n] = V_{mn}^{(k)}

        """
        if nmax == 0 :
            nmax = self.nrungs
        global PsiMat,Nshift,NP,N0,phivec
        if k ==0 :
            return eye(nmax)
        
        assert(type(k)==int)
        shift = k * self.well_spacing
        while True:
                    
            Nshift = int(shift//dphi+0.1)
            
            dphi = shift/Nshift
            
            phivec = arange(-phimax,phimax,dphi)
            NP = len(phivec)
            phivec = phivec-phivec[NP//2]
            N0 = NP//2 
            # Nshift = 
            
            dp = phivec[1]-phivec[0]
            
            PsiMat = zeros((nmax,NP))
            
            for n in range(0,nmax):
                f = self.get_well_eigenfunction(n)
                PsiMat[n] = f(phivec)
                
            if sum(PsiMat[:,int(NP*0.9):]**2)*dp < 1e-8:
                if  Nshift>0:
                    
                    out = PsiMat[:,Nshift:]@PsiMat[:,:-Nshift].T*dp
                else:
                        
                    out = PsiMat[:,:Nshift]@PsiMat[:,-Nshift:].T*dp
                    
          
                return out
            
            else:
                    
                phimax = phimax * 1.2
                
    # def well_to_orth
        

    
def apply_matrix_list(matrix_list,psi,expand_shape = False ):
    """
    Compute the matrix 
    
    X_{a,l}[psi] = \sum_{b,m} M^{l-m}_{ab}psi_{bm}

    For instance, this can be used for calculating overlaps:
    
    <phi|psi> = \sum_{a,a',k,k'} <a,k|a',k'>    phi^*_{a,k}psi_{a',k'}
              = \sum_{a,a',k,k'} V^{k-k'}_{aa'} phi^*_{a,k}psi_{a',k'}
              = \sum_{a',k} phi^*_{a,k}X_{ak}[psi]
    Parameters
    ----------
    matrix_list : list of ndarrays (nrungs,nrungs)
        Matrix list. matrix_list[n0+k] = M^k_{ab}
        with n0 = shape(matrix_list)-1/2 
    psi : ndarray of floats (nrungs,nwells) or (nrungs,nwells,nstates)
        state(s) to be multiplied with.


    Returns
    -------
    X : ndarray of floats, same dims as psi
             (nrungs,nwells) or (nrungs,nwells,nstates)
        Output: X[a,l] = X_{a,l}.

    """
    # global X,mat
    NL = shape(matrix_list)[0]
    # print(NL)
    assert NL%2 == 1,"Matrix list must have odd length"
    
    n0 = (NL-1)//2 
    
    X = 0 
    
    """ 
    X_{a,l}[psi] = \sum_{b,m} M^{l-m}_{ab}psi_{bm}
                 = \sum_{b,k} M^{k}_{ab}psi_{b(l-k)}
    """ 
    Npsi = shape(psi)[0]
    if expand_shape==False:
            
        X = 0 * psi
        for k in range(-n0,n0+1):
            mat = matrix_list[n0+k]
            
            if k>0:
                X[k:] += mat@psi[:-k]
            elif k<0:
                X[:k] += mat@psi[-k:]        
            else:
                X       += mat@psi
                
                
        return X 

    else:

        S0 = shape(psi)
        S  = (Npsi +NL-1,)+S0[1:]
        
        X = zeros(S,dtype=psi.dtype)
        for k in range(0,2*n0+1):
            mat = matrix_list[k]
            
        
            X[k:k+Npsi] += mat@psi
            # elif k<0:
            #     X[:k] += mat@psi[-k:]        
            # else:
            #     X       += mat@psi
                
                
        return X       
        


        
        
        
    
def get_matrix_list_power(matrix_list,p,nblocks,threshold=1e-9,max_it = 1000,renormalize=True):
    """
    Compute the pth power of A, X = (A)^p; 
    
    where A[nrungs*k:nrungs*(k+1),nrungs*l:nrungs*(l+1)] = Alist[k-l]}.
                  
    I.e., the pth power of Id + V with respect to the multiplication operation 
    
        [A o B]^k_{ab} = \sum_{l} A^B{k-l}_{ab} X^l_{bc}
        
    with Id^k_{ab} = \delta_{k0} * \delta_{ab}
    
    We compute the power using the Taylor series for A = (1+delta A), (1+delta A)^p:
            
            1 + p A + p(p-1)/2 A^2 + p (p-1) (p-2) A^3/6 + ... 
            
            = \sum_k \prod_{n=0}^k (p-n)/(n+1) A^k 
            
            = \sum_k C_k
        
        where 
        
            C_0 = 1 
            
            C_{k+1} = A o C_k * (p-k)/(k+1) 
    

    Returns
    -------
    Xlist, where Xlist[k-l] = X[nrungs*k:nrungs*(k+1),nrungs*l:nrungs*(l+1)] 

    """
    Alist = 1*matrix_list 
    if renormalize :
        
        val = get_matrix_list_norm(Alist)
        q = 1.3*val
        Alist = Alist/q
        
    else:
        val = 1 
        q   = 1 
        
        
    assert nblocks % 2 == 1
    dim        = shape(Alist)[1]
    C0         = zeros((nblocks,dim,dim),dtype=Alist.dtype)
    n0_out     = nblocks//2+1
    n0_a       = (shape(Alist)[0]-1)//2
    C0[n0_out] = eye(dim,dtype=Alist.dtype)
   
    dAlist = 1*Alist
    dAlist[n0_a]  = dAlist[n0_a] - eye(dim)
    
    C      = 1 * C0 
    
    out    = zeros((nblocks,dim,dim),dtype=Alist.dtype)
    
    converged = False 
    
    block_range = 0 

    nclist = []
    for k in range(0,max_it):
        out += C
        # print(out[n0_out-2:n0_out+2])
        C = (p-k)/(k+1)*apply_matrix_list(dAlist,C)
        
        nc  =norm(C)
        nclist.append(nc)
        if nc  < threshold:
            
            converged = True 
            # print(f"power {p} converged at order {k}")
            break 
        
        
        block_range += 1 
        
    # if block_range > n0_out:
      
    #     print(f"Warning: block_range chosen too small. Use range {block_range}")
    
    if converged:
        if block_range <= n0_out:
            out = out[n0_out-block_range:n0_out+1+block_range:]
        else:
      
            out = out[2:]
        global n0
        n0 = (shape(out)[0]-1)//2
        I0 = where(norm(out,axis=(1,2))>1e-9)[0]
        a0 = amin(I0)
        a1 = amax(I0)
        
        dmax = max(abs(a0-n0),abs(a1-n0))
        
        if a0>0:
           ind = arange(n0-dmax-1,n0+dmax)+1
            
           out = out[ind]
       
        if renormalize:
            
            return out*(q**p)
    
    else:
        figure(1)
        plot(log10(array(nclist)))
        raise AssertionError("Taylor series failed to converge. Spectrum of matrix too spread out")
        
             
def get_matrix_list_norm(matrix_list):
    global vec
    S = shape(matrix_list)[1]
    vec = zeros((101,S,1))
    vec[50] +=1 
    # print(vec)
    vec = vec/norm(vec)
    
    val_old = 1e9
    for k in range(0,1000):
        vecnew = apply_matrix_list(matrix_list, vec,expand_shape = False )
        # print(vecnew)
        s_old = shape(vec)[0]
        s_new = shape(vecnew)[0]
        ds = s_new-s_old 
        assert ds%2 == 0 
        
        val    = sum(vecnew.conj()*vec)
        
        # print(val)
        if abs(val/val_old-1)<1e-8:
            return val
        val_old = 1 * val
        vec = 1*vecnew
        vec = vec/norm(vec)
        
    return val 

def transpose_matrix_list(matrix_list):
    matlist = matrix_list 
    
    out = matlist[::-1,:,:]
    out = out.swapaxes(1,2)
    
    return out 

def ho_psi(a,phi):
    """
    return dimensionless HO wavefunction

    Parameters
    ----------
    a : TYPE
        DESCRIPTION.
    phi : TYPE
        DESCRIPTION.

    Returns
    -------
    Ha : TYPE
        DESCRIPTION.

    """
    Ha = hermite(a)
    out =  (2**a*factorial(a)*sqrt(pi))**(-1/2)*exp(-phi**2/2)*Ha(phi)
    return out 




if __name__=="__main__":
    # Shape of grid (wells,rungs)
    S = (30,16)
    
    # Create grid object    
    LCJ =   LCJ_circuit(L,C,Josephson_energy,S)
    
    # Get matrix elements of Hamiltonian in (non-orthogonal) well basis
    b1  = 1
    b2  = 1
    
    HLC = LCJ.compute_H_LC_block(b1,b2)
    HJJ = LCJ.compute_H_JJ_block(b1,b2)
    
    
    # Plot wavefunctions of orthogonal states 
    figure(1)
    # subplot(223)
    pcolormesh(HLC/GHz,cmap="bwr",vmin=-100,vmax=100)
    title(f"Plot of LC Hamiltonian <{b1},a|"+"$H_{\\rm LC}$"+f"|{b2},b>, in GHz")
    colorbar()
    xlabel("index a")
    ylabel("index b")
    # subplot(224)
    figure(2)
    title(f"Plot of JJ Hamiltonian <{b1},a|"+"$H_{\\rm JJ}$"+f"|{b2},b>, in GHz")
    xlabel("index a")
    ylabel("index b")
    pcolormesh(HJJ/GHz,cmap="bwr",vmin=-100,vmax=100)
    colorbar()

    a,b = 11,14
    phi = arange(-5*pi,5*pi,.001)
    dphi = phi[1]-phi[0]
    figure(3)
    # pcolormesh(abs(LCJ.get_orthonormal_state(8)).T)'
    Psi1 = LCJ.get_orthonormal_state(a)
    Psi2 = LCJ.get_orthonormal_state(b)
    dk = 1
    if dk>0:
            
        Psi2 = vstack((Psi2[dk:,:],0*Psi2[-dk:,:]))
    
    f1 = LCJ.get_wavefunction(Psi1)
    f2 = LCJ.get_wavefunction(Psi2)
    
    print(f"Double-checked overlap of orthonormal states\n    <<{0},{a}|{dk},{b}>>  =  {sum(f1(phi)*f2(phi))*dphi}")
    

    figure(8)
    y1 = f1(phi)
    y2 = f2(phi)
    plot(phi/pi,y1)
    plot(phi/pi,y2)
    title(f"Wavefunctions of orthonormal states |0,{a}$\\rangle\!\\rangle$ and |{dk},{b}$\\rangle\!\\rangle$")
    xlabel("$\phi/\pi$")
    show()
