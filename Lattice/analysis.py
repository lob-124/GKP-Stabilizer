#Add directory with Frederik's code to path
from sys import path
path.insert(0,"../Frederik/")

from units import *
from basic import get_tmat

from numpy import diag,cos,array,eye,complex64,zeros,pi,concatenate,linspace,exp,convolve
from numpy.linalg import norm,eigh 
from numpy.fft import fft, fftfreq, fftshift

from time import perf_counter

def my_norm(psi):
	return sum(list(map(lambda v: norm(v)**2,psi.values())))


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


def wigner(psi,x_in,pvec = None,print_progress=False ):
	x = x_in
	L = len(x_in)
	dx = x_in[1]-x_in[0]
	x_out = linspace(x[0]-x[-1],x[-1]-x[0],2*L-1)/2
	
	if type(pvec) ==type(None):
		
		prange = x_out
		p_out = prange
	else:
		prange = pvec
		p_out = prange 
		
	W = zeros((len(p_out),2*L-1))
	
	num_p = 0
	for p in prange:
		if print_progress and (num_p % 250 == 0):
			print(num_p)
		v= psi * exp(1j*x_in*p)
		# v2 = psi * exp()
		W[num_p] = convolve(v,v.conj())*dx/pi
		num_p+=1 
	
	
	# W = W.T 
	return W,x_out,p_out


def upper_state_norm(psi,num):
	"""
	Return the norm squared of the wavefunction in the num uppermost levels of each well 
	"""
	_norm = 0.0
	for v in psi.values():
		_norm += norm(v[-num:])**2

	return _norm


def outer_well_norm(psi,num,max_well):
	"""
	Return the norm squared of the wavefunction in the 2*num outermost wells (num on each side) 
	"""
	
	_norm = 0.0
	for i in range(num):
		_norm += norm(psi.get(-max_well+i,0.0))**2
		_norm += norm(psi.get(max_well-i,0.0))**2
		
	return _norm


def expectation_sz(psi):
	"""
	Return the S_z expectation value of the spin, defined as the total weight in even wells minus the total 
		weight in odd wells
	"""
	res = 0.0
	for well_ind,vec in psi.items():
		res += (-1)**well_ind*norm(vec)**2
	return res


class Analysis():

	def __init__(self,omega,E_J,D,overlaps,max_wells,f,alpha=1):
		self.omega = omega
		self.E_J = E_J
		
		self.D        = D        # Dimension of Hilbert space
		self.phi_0    = linspace(-pi,pi,self.D)
		self.dphi     = self.phi_0[1]-self.phi_0[0]
		self.max_wells = max_wells

		self.overlaps = overlaps
		
		self.alpha  = alpha
		self.Z  = planck_constant/(2*self.alpha*e_charge**2)
		self.L = self.Z/self.omega
		self.C = 1/(self.Z*self.omega)

		self.Tmat    = get_tmat(self.D,dtype=complex)
		self.Tmat[0,-1]=0
		self.Pi0     = 1j*(self.Tmat-eye(self.D))/self.dphi
		self.Pi_2 = 0.5*(self.Pi0@self.Pi0.conj().T + self.Pi0.conj().T@self.Pi0)    #squared momentum
		self.Q2 = ((2*e_charge)/hbar)**2 * self.Pi_2  #Squared charge
		self.V = diag(cos(self.phi_0))


		#Compute and store the eigenvectors in each well
		self.eigvals = {}
		self.eigvecs = {}
		for well_num in range(-max_wells,max_wells+1):
			phi = self.phi_0 + 2*pi*well_num
			
			#Relevant operators in this well
			H_LC = (hbar**2/(4*e_charge**2))*diag(phi**2)/(2*self.L) + self.Q2/(2*self.C)
			H = H_LC - self.E_J*self.V
		
			#Eigenvalues and Eigenvectors
			_eigvals , _eigvecs = eigh(H)

			#Extract the "bound states"
			#	i.e., find the index i such that 
			#		E_j - E_0 <= f*E_J
			#	for all j < i
			i = 1
			while ((_eigvals[i] - _eigvals[0]) <= f*E_J):
				i += 1

			self.eigvals[well_num] = _eigvals[:i]
			self.eigvecs[well_num] = _eigvecs[:,:i]
	



	def reconstruct_wavefunction(self,psi):
		phis , psi_recon = [] , []
	
		for well_num , well_vec in psi.items():
			# phi = self.phi_0 + 2*pi*well_num

			# #Relevant operators in this well
			# H_LC = (hbar**2/(4*e_charge**2))*diag(phi**2)/(2*self.L) + self.Q2/(2*self.C)
			# H = H_LC - self.E_J*self.V
		
			# #Eigenvalues and Eigenvectors
			# _eigvals , _eigvecs = eigh(H)
			
			# phis.append(phi)
			# psi_recon.append(_eigvecs[:,:len(well_vec)]@well_vec)
			psi_recon.append(self.eigvecs[well_num]@well_vec)
		
		return phis,psi_recon


	def reconstruct_wavefunction2(self,psi):
		phis , psi_recon = array([]) , array([],dtype=complex64)
		
		#max_well = max(psi.keys())
		zero_vec = zeros(self.D,dtype=complex64)
		
		for i in range(-self.max_wells,self.max_wells+1):
			phi = self.phi_0 + 2*pi*i
			# well_vec = psi.get(i)
			# if well_vec is not None:
			# 	#Relevant operators in this well
			# 	H_LC = (hbar**2/(4*e_charge**2))*diag(phi**2)/(2*self.L) + self.Q2/(2*self.C)
			# 	H = H_LC - self.E_J*self.V
		
			# 	#Eigenvalues and Eigenvectors
			# 	_eigvals , _eigvecs = eigh(H)
			# 	psi_recon = concatenate((psi_recon,_eigvecs[:,:len(well_vec)]@well_vec))
			# else:
			# 	psi_recon = concatenate((psi_recon,zero_vec))
			well_vec = psi.get(i)
			if well_vec is not None:
				psi_recon = concatenate((psi_recon,self.eigvecs[i]@well_vec))
			else:
				psi_recon = concatenate((psi_recon,zero_vec))
		
			phis = concatenate((phis,phi))
			
		return phis,psi_recon


	def stabilizers(self,psi):
		"""
		Return the expectation of the two stabilizer oeprators for the given state.
		"""
		stab_1 = 0.0
		stab_2 = 0.0
		#print("---- NEW ATTEMPT ----")
		for well_num, well_vec in psi.items():
			phi = self.phi_0 + 2*pi*well_num

			
			#t1 = perf_counter()
			
			psi_recon = self.eigvecs[well_num]@well_vec

			#Add to the expected value of the first stabilizer
			stab_1 += sum(abs(psi_recon)**2*exp(-1j*self.alpha*phi))
			#t2 = perf_counter()
			#print("Stabilizer 1: {}".format(t2-t1))

			#t1 = perf_counter()
			next_well = psi.get(well_num+2)
			if next_well is not None:
				stab_2 += next_well.conj() @ self.overlaps[well_num] @ well_vec
			#t2 = perf_counter()
			#print("Stabilizer 2: {}".format(t2-t1))
		

		_norm = my_norm(psi)
		return stab_1/_norm , stab_2/_norm


	def quarter_cycle(self,psi):
		"""
		Given an input vector in the bound state basis, outputs the projections of the 1/4-cycle 
		    evolution under the LC Hamiltonian onto the various bound states in each well.

		Args:
		    psi: dict
		        The input wavefunction, in the bound state basis, to be transformed. Should be a dict, with psi[i][j] the 
		            projection of psi onto state j in well i
		    
		Returns:
		    projections: dict
		        The projections onto the various bound well states, as a dictionary.
		        Specifically, projections[w] is an array containing the projections of the evolution of vec onto the
		            bound states in well w.
		"""
		#Construct the wavefunction in the phi basis
		psi_vec = array([],dtype=complex64)
		min_well,max_well = min(psi.keys()) , max(psi.keys())
		for w in range(min_well,max_well+1):
		    psi_this_well = self.eigvecs[w] @ psi.get(w,zeros(self.eigvecs[w].shape[1])) 
		    psi_vec = concatenate((psi_vec,psi_this_well))

		#Compute the norm of the wavefunction
		norm_psi = sqrt(my_norm(psi))


		#Determine the amount of phi points we need to have (approximately) the same spacing in the phi-grid after the FFT
		k_half = self.D    
		k_left = min_well+k_half
		k_right = k_half-max_well
		    

		#Pad the wavefunction with zeros to the left and right of the wells it's defined in (if necessary) 
		if k_left > 0:
		    psi_left = zeros((self.D-1)*k_left+1)
		    psi_vec = concatenate((psi_left,psi_vec))
		    phi_min = -pi*(2*k_half+1)
		else:
		    phi_min = -pi*(2*min_well+1)
		    
		if k_right > 0:
		    psi_right = zeros((self.D-1)*k_right+1)
		    psi_vec = concatenate((psi_vec,psi_right))
		    

		phi_tilde , psi_tilde = quarter_cycle_phi(psi_vec,self.dphi,phi_min)

		interpolations = {}
		max_well_index = min(self.max_wells,k_half)
		for i in range(-max_well_index,max_well_index+1):
		    mask_array = abs(phi_tilde-2*pi*i) <= pi
		    phi_well = phi_tilde[mask_array]
		    psi_well = psi_tilde[mask_array]
		    
		    #Interpolate for psi at the phi points we used in the ED to construct the bound states
		    phi_diag = self.phi_0 + 2*pi*i
		    interpolations[i] = interp(phi_diag,phi_well,psi_well)

		#Get the norm of the interpolated vector
		interp_norm = sqrt(sum([norm(vec)**2 for vec in interpolations.values()]))
		    
		projections = {}
		for i in range(-max_well_index,max_well_index+1):
		    #Take the inner product of the interpolated psi with the bound state wavefunctions obtained earlier
		    projections_this_well = []
		    num_this_well = self.eigvecs[i].shape[1]
		    for j in range(num_this_well):
		        projections_this_well.append((self.eigvecs[i][:,j].conj().T @ interpolations[i])*norm_psi/interp_norm)
		    
		    projections[i] = array(projections_this_well)
		        
		    
		return projections


	def S_evolve(self,psi):
		"""
		Perform the evolution under the full (LC + JJ) Hamiltonian for a duration 2\pi/\omega - effectively implementing 
			an S gate. 
		"""
		res = {}
		for w,vec in psi.items():
			res[w] = exp(-1j*self.eigvals[w]/self.omega)*vec

		return res


	def expectation_sz(self,psi):
		"""
		Return the S_z expectation value of the spin, defined as the total weight in even wells minus the total 
			weight in odd wells
		"""
		res , _norm = 0.0 , 0.0
		for well_ind,vec in psi.items():
			res += (-1)**well_ind*norm(vec)**2
			_norm += norm(vec)**2
		return res/_norm

	def expectation_sx(self,psi):
		"""
		Compute the S_x expectation, obtained by doing a quarter-cycle evolution (~ Hadamard gate)
			and then an S_z expectation
		"""
		psi_rot = self.quarter_cycle(psi)
		return self.expectation_sz(psi_rot)

	def expectation_sy(self,psi):
		"""
		Compute the S_x expectation, obtained by doing a phase gate (evolution under the LC Hamiltonian), followed
			by a quarter-cycle evolution (~ Hadamard gate), and then an S_z expectation
		"""
		psi_phase = self.S_evolve(psi)
		psi_rot = self.quarter_cycle(psi_phase)
		return self.expectation_sz(psi_rot)

