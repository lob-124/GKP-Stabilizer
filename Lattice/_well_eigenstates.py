#Add directory with Frederik's code to path
from sys import path
path.insert(0,"../Frederik/")

from units import *
from basic import get_tmat

from numpy import diag,cos,exp,sqrt,ceil,interp,zeros,pi,complex64
from numpy.linalg import eigh,norm
from numpy.fft import fft,fftshift,fftfreq

from time import perf_counter


def quarter_cycle_phi(psi,dphi,phi_min,alpha=4*pi):
    """
    Given a wavefunction phi defined over a range of phi, outputs the wavefunction resulting from
        1/4-cycle evolution under the LC Hamiltonian.

    This is essentially an FFT, with a rescaling:
                U_{1/4}psi(x) = Psi(-x/alpha)/sqrt(i*alpha)
        where
                Psi(k) = (1/sqrt{2*pi})*int_{-infty}^{infty}psi(x)exp(ikx)dx

        and alpha defined below.

    Params:
        psi: ndarray
            The wavefunction, in the phi basis
        dphi: float
            The spacing of the phi grid used to define psi
        phi_min: float
            The minimum value in the phi grid used to define psi
        alpha: float, optional
            The value of (4e^2/hbar)sqrt{L/C}, as dictated by the stabilizer condition. Defaults 
                to 4pi
    """
    #Do the FFT
    psi_tilde = fftshift(fft(psi,norm="ortho"))
    phi_tilde = 2*pi*fftshift(fftfreq(len(psi),dphi))   #NB: numpy.fft defines the DFT with exp(-2pi*i*k*x), hence the factor of 2*pi

    #The DFT in numpy.fft assumes the signal starts at phi=0, whereas the underlying phi grid starts at
    #   phi_min. This means we need to include a shift of exp(i*phi_min*k) to get the correct Fourier
    #   transform of psi  
    psi_tilde = exp(1j*phi_min*phi_tilde)*psi_tilde

    #Return the results, rescaling phi_tilde by alpha
    #NB: we don't reverse the order of psi_tilde since numpy.fft uses exp(-2pi*i*kx) in the forward transform
    return alpha*phi_tilde , psi_tilde/sqrt(1j)




class Wells():

    def __init__(self,omega,E_J,D,quantization_param=1.0):
        self.omega = omega
        self.E_J = E_J
        self.D = D
        self.quantization_param = quantization_param

        self.phi_0    = linspace(-pi,pi,D)
        self.dphi     = self.phi_0[1]-self.phi_0[0]

        self.Z  = planck_constant/(2*self.quantization_param*e_charge**2)
        self.L = self.Z/self.omega
        self.C = 1/(self.Z*self.omega)

        self.Tmat    = get_tmat(self.D,dtype=complex64)
        self.Tmat[0,-1]=0
        self.Pi0     = 1j*(self.Tmat-eye(self.D))/self.dphi     #conjugate momentum to phi
        self.Pi_2 = 0.5*(self.Pi0@self.Pi0.conj().T + self.Pi0.conj().T@self.Pi0)    #squared momentum
        self.Q2 = ((2*e_charge)/hbar)**2 * self.Pi_2  #Squared charge

        self.k_half = int(ceil(self.D/pi))//2

    def diagonalize_well(self,well_num,fraction=1.7):
        phi = self.phi_0 + 2*pi*well_num

        #Relevant operators in this well
        H_LC = (hbar**2/(4*e_charge**2))*diag(phi**2)/(2*self.L) + self.Q2/(2*self.C)
        #H_LC = self.Q2/(2*self.C)
        H = H_LC - self.E_J*diag(cos(self.phi_0))
        
        #Eigenvalues and Eigenvectors
        _eigvals , _eigvecs = eigh(H)


        #Find the highest energy level with excitation energy less than a specificed fraction of E_J
        max_n = 0
        while _eigvals[max_n] - _eigvals[0] < fraction*self.E_J:
            max_n += 1
        #NB: max_n is one MORE than the maximal index satisfying E_n < fraction*E_J



        #Compute the mean square displacements from the center of the well
        ms_vals = zeros((max_n,max_n))
        for i in range(max_n):
            for j in range(max_n):
                ms_vals[i,j] = sum(_eigvecs[:,i].conj()*(self.phi_0**2)*_eigvecs[:,j])


        #Compute the matrix elements of H_LC in this well for the max_n lowest-lying levels
        H_LC_well = zeros((max_n,max_n))
        for i in range(max_n):
            for j in range(i,max_n):
                H_LC_well[i,j] = _eigvecs[:,i].conj().T @ H_LC @ _eigvecs[:,j]
                H_LC_well[j,i] = H_LC_well[i,j].conj()

        #Compute the matrix elements of X1, X2 in this well for the max_n lowest-lying levels
        X1_well = zeros((max_n,max_n),dtype=complex64)
        X2_well = zeros((max_n,max_n),dtype=complex64)
        for i in range(max_n):
            for j in range(i,max_n):
                X1_well[i,j] = sum(_eigvecs[:,i].conj()*exp(1j*phi)*_eigvecs[:,j])
                X2_well[i,j] = sum(_eigvecs[:,i].conj()*exp(-1j*phi)*_eigvecs[:,j])

                if i != j:
                    X1_well[j,i] = X2_well[i,j].conj()
                    X2_well[j,i] = X1_well[i,j].conj()


        return _eigvals[:max_n] , ms_vals, H_LC_well, X1_well, X2_well




    def translation_matrix(self,well_num,fraction=1.7):
        """
        Compute the matrix elements <k+2,n'|T+|k,n> for a given well number k, where T+ is the
            operator that translates by 4pi (<-> 2 wells)
        """

        phi_plus = self.phi_0 + 2*pi*(well_num+2)
        phi = self.phi_0 + 2*pi*well_num

        #Hamiltonians in the wells
        H_plus = (hbar**2/(4*e_charge**2))*diag(phi_plus**2)/(2*self.L) + self.Q2/(2*self.C) - self.E_J*diag(cos(self.phi_0))
        H = (hbar**2/(4*e_charge**2))*diag(phi**2)/(2*self.L) + self.Q2/(2*self.C) - self.E_J*diag(cos(self.phi_0))
        #H_plus = self.Q2/(2*self.C) - self.E_J*diag(cos(self.phi_0))
        #H = self.Q2/(2*self.C) - self.E_J*diag(cos(self.phi_0))
        
        
        #Eigenvalues and Eigenvectors
        _eigvals_plus , _eigvecs_plus = eigh(H_plus)
        _eigvals , _eigvecs = eigh(H)


        # #Find the highest energy level with energy less than a sepcificed fraction of E_J in each well
        max_n_plus, max_n = 0 , 0
        still_looking_plus , still_looking = True, True
        while still_looking_plus or still_looking:
            if still_looking_plus: 
                if _eigvals_plus[max_n_plus] - _eigvals_plus[0] < fraction*self.E_J:
                    max_n_plus += 1
                else:
                    still_looking_plus = False
            if still_looking: 
                if _eigvals[max_n] -_eigvals[0] < fraction*self.E_J:
                    max_n += 1
                else:
                    still_looking = False

        
        #The truncated Hilbert spaces
        basis_plus = _eigvecs_plus[:,:max_n_plus]
        basis = _eigvecs[:,:max_n]

        #Return the overlaps of the eigenstates in the adjacent wells
        return basis_plus.conj().T @ basis


    def quarter_cycle_wells(self,max_wells,fraction=1.7):
        """
        Compute the matrices of the 1/4-cycle evolution in the bound state basis.

        Params:
            max_wells: int
                The maximum well index to include - i.e., include wells from -max_well
                    to max_well
            fraction: float, optional
                The fraction f defining the bound states in each well.
                Specifically, if E is a list of the energies in each well, we keep only 
                    states i whose energies satisfy 
                                        E[i] - E[0] <= fraction*E_J
                    where E[0] is the ground state of each well. Defaults to 1.7

        Returns:
            quarter_cycle_evo: dict
                A dictionary of dictionaries containing the matrices of the quarter-cycle evolution in the 
                    bound state basis.
                More specifically, quarter_cycle_evo[w1][w2][i,j] is the matrix element
                                        <w2,i|U_{1/4}|w1,j>
                    where U_{1/4} the 1/4-cycle evolution operator, and |w,i> is the i'th bound state in well
                    number w.
        """

        #The part of the Hamiltonian that is invariant between wells
        _H = self.Q2/(2*self.C) - self.E_J*diag(cos(self.phi_0))

        #Get the grid of bound states
        well_vecs = {}
        for w in range(-max_wells,max_wells+1):
            phi = self.phi_0 + 2*pi*w
            H = (hbar**2/(4*e_charge**2))*diag(phi**2)/(2*self.L) + _H
            
            #Eigenvalues and Eigenvectors of H
            _eigvals , _eigvecs = eigh(H)

            #Keep only the "bound states" satisfying E[i] - E[0] <= fraction*E_J
            well_vecs[w] = _eigvecs[:,_eigvals - _eigvals[0] <= fraction*self.E_J]


        #Loop through the wells, and evolve each bound state in each using quarter_cycle_phi().
        #   Then project each of those results onto the bound states of all wells
        quarter_cycle_evo = {}
        for w1 in range(-max_wells,max_wells+1):
            print(w1)
            d1 = well_vecs[w1].shape[1]
            k_left = self.k_half + w1
            k_right = self.k_half - w1

            quarter_cycle_evo[w1] = {}

            for i in range(d1):
                #Get the current vector
                psi = well_vecs[w1][:,i]

                #Pad the vector with zeros left and right as needed, to ensure the phi grid 
                #   returned by quarter_cycle_phi has (approximately) the same spacing as the 
                #   original
                if k_left > 0:
                    psi_left = zeros((self.D-1)*k_left+1)
                    psi = concatenate((psi_left,psi))
                    phi_min = -pi*(2*self.k_half+1)
                else:
                    phi_min = -pi*(2*w1+1)
                if k_right > 0:
                    psi_right = zeros((self.D-1)*k_left+1)
                    psi = concatenate((psi,psi_right))

                #Evolve the vector
                phi_tilde , psi_tilde = quarter_cycle_phi(psi,self.dphi,phi_min)

                #We want to take the inner product of the evolved vector with the well vectors. 
                #   To do so correctly, both vectors need to be defined at the same phi points. We 
                #   thus interpolate for psi_tilde in each well at the points in the original phi 
                #   grid.
                interpolations = {}
                for w in range(-max_wells,max_wells+1):
                    phi_diag = self.phi_0 + 2*pi*w
                    mask_array = abs(phi_tilde-2*pi*w) <= pi
                    interpolations[w] = interp(phi_diag,phi_tilde[mask_array],psi_tilde[mask_array])

                #Compute the norm of the interpolated wavefunction
                interp_norm = sqrt(sum([norm(vec)**2 for vec in interpolations.values()]))


                #Loop over all wells, and project onto each bound state in each well
                for w2 in range(-max_wells,max_wells+1):
                    d2 = well_vecs[w2].shape[1]
                    if quarter_cycle_evo[w1].get(w2) is None:
                        quarter_cycle_evo[w1][w2] = zeros((d2,d1),dtype=complex64)

                    for j in range(d2):
                        quarter_cycle_evo[w1][w2][j,i] = (well_vecs[w2][:,j].conj() @ interpolations[w2])/interp_norm

            #end loop over bound states of well w1
        #end loop over wells

        return quarter_cycle_evo , well_vecs



#end class