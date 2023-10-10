#Add directory with Frederik's code to path
from sys import path
path.append("../Frederik/")
from units import *

from numpy import sqrt,pi,zeros,linspace,interp,array,complex64,ceil
from numpy.random import default_rng
from numpy.linalg import norm
from numpy.fft import fft, fftfreq, fftshift


#Function to apply operators to psi (where the operator is assumed to act only WITHIN each well)
def apply_diagonal_operator(psi,O):
	"""
	Given a wavefunction and operator O, represented in our grid structure, apply the 
		unitary to psi. We assume U does not couple between wells
	"""
	return dict(map(lambda tup: (tup[0],O[tup[0]] @ tup[1]), psi.items()))


#Function to compute the norm squared of psi
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



#### ****
#### ****
##  Another version of the class, that is in the lab frame (so there is only one window instead of two)
#### ****
#### ****
class SSE_lab():

	#We represent the wavefunctions as a dictionary, with psi[k] being a vector
	#	of the components of psi in the basis of eigenfunctions of well k
	#More precisely:
	#			psi[k][i] = <k,i|psi>
	#	where |k,i> is the ith level in well k.
	#We also only keep track of the wells where psi has support, so psi.items() gives
	#	a list of the well indices where components of psi are non-zero 

	def __init__(self,E_J,max_wells,well_vecs,Us,dUs,L1s,L2s,delta_t,num_tpoints,tol=1e-10):
		#System parameters and bound state basis
		self.E_J = E_J	
		self.max_wells = max_wells
		self.well_vecs = well_vecs
		self.D = well_vecs[0].shape[0]
		self._phi = linspace(-pi,pi,self.D)
		self.dphi = self._phi[1] - self._phi[0]

		#(Block) matrices for the SSE evolution
		self.Us = Us
		self.dUs = dUs
		self.L1s = L1s
		self.L2s = L2s
		self.delta_t = delta_t
		
		#The time points in the window
		self.num_tpoints = num_tpoints
		self.t_vals_window = linspace(0,delta_t,num_tpoints+1)

		#Tolerance threshold (of the L_2 norm) for keeping a well vector
		self.tol = tol
		

	# def quarter_cycle(self,psi):
	# 	"""
	# 	Apply the quarter cycle evolution to the given wavefunction.
	# 	"""
	# 	psi_mapped = {}
	# 	for w1 in psi.keys():
	# 		matrices_this_well = self.quarter_cycle_matrices[w1] 
	# 		psi_this_well = psi[w1]

	# 		for w2 in range(-self.max_wells,self.max_wells+1):
	# 			new_vec = matrices_this_well[w2] @ psi_this_well
	# 			if norm(new_vec) > self.tol:
	# 				psi_mapped[w2] = psi_mapped.get(w2,0.0) + new_vec

	# 	return psi_mapped

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
		    psi_this_well = self.well_vecs[w] @ psi.get(w,zeros(self.well_vecs[w].shape[1])) 
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
		    phi_diag = self._phi + 2*pi*i
		    interpolations[i] = interp(phi_diag,phi_well,psi_well)

		#Get the norm of the interpolated vector
		interp_norm = sqrt(sum([norm(vec)**2 for vec in interpolations.values()]))
		    
		projections = {}
		for i in range(-max_well_index,max_well_index+1):
		    #Take the inner product of the interpolated psi with the bound state wavefunctions obtained earlier
		    projections_this_well = []
		    num_this_well = self.well_vecs[i].shape[1]
		    for j in range(num_this_well):
		        projections_this_well.append((self.well_vecs[i][:,j].conj().T @ interpolations[i])*norm_psi/interp_norm)
		    
		    projections[i] = array(projections_this_well)
		        
		    
		return projections





	def SSE_evolution(self,psi_0,seed,num_periods):
		"""
		Perform one trajectory of the SSE evolution.

		Params:
			psi_0: dict
				Initial wavefunction. Should be a dictionary, whose keys are the well numbers and items
					are the wavefunction in the basis of eigenstates of each well
			seed: int
				The seed for the RNG of this run
			num_periods: int
				The number of driving periods to simulate for

		Returns:
			A list of the wavefunctions (each one a dict as described above) after each driving period
		"""

		rng = default_rng(seed)
		r = rng.random()

		psi = dict(psi_0)
		
		psis_mid , psis_end =  [] , []
		pre_jumps , post_jumps = [] , []
		jump_times = []
		for i in range(num_periods):
			if i % 25 == 0:
				print("Now on period {} of {} for seed {}".format(i,num_periods,seed))
			#Indices used by binary search within the window
			start_index = 0
			ind_left = 0
			ind_right = self.num_tpoints

			
			#Perform the evolution over the first window
			psi_new = apply_diagonal_operator(psi,self.Us[-1])

			#Check if there was a quantum jump within this window
			while (1 - my_norm(psi_new)) > r:
				#print("JUMP!")
				#Binary search for the time at which the jump occurred
				while (ind_right - ind_left) > 1:
					ind_mid = round((ind_right+ind_left)/2)

					#Evolve the wavefunction to the time given by ind_mid
					if start_index != 0:
						#If we've already had a jump this period, evolve from the time
						#	of the last jump
						#NB: psi is the wavefunction at the previous jump time (see ****)
						psi_temp = dict(psi)
						for index in range(start_index,ind_mid):
							psi_temp = apply_diagonal_operator(psi_temp,self.dUs[index])
					else:
						#Otherwise, evolve from the beginning of the period
						psi_temp = apply_diagonal_operator(psi,self.Us[ind_mid])

					#Check if the jump has happened yet
					if (1 - my_norm(psi_temp)) >= r:
						ind_right = ind_mid
					else:
						ind_left = ind_mid

				#end binary search while loop

				#We've now found the time t at which the jump occurred
				ind = ind_right
				jump_times.append(i*self.delta_t + self.t_vals_window[ind])

				#Advance the wavefunction to time t (****)
				if start_index != 0:
					#Evolve from prior jump (if needed)
					for index in range(start_index,ind):
						psi = apply_diagonal_operator(psi,self.dUs[index])
				else:
					#Otherwise, evolve from beginning of period
					psi = apply_diagonal_operator(psi,self.Us[ind])

				#Record the wavefunction pre-jump (for debugging purposes)
				pre_jumps.append(psi)

				#Determine the type of jump, and jump the wavefunction
				_psi1 = apply_diagonal_operator(psi,self.L1s)
				_psi2 = apply_diagonal_operator(psi,self.L2s)
				p_1 , p_2 = my_norm(_psi1) , my_norm(_psi2)
				if rng.random() < p_1/(p_1+p_2):
					psi = dict([(well,vec/sqrt(p_1)) for well,vec in _psi1.items()])
				else:
					psi = dict([(well,vec/sqrt(p_2)) for well,vec in _psi2.items()])

				#Record the wavefunction post-jump (for debugging purposes)
				post_jumps.append(psi)

				#Reset the random variable r
				r = rng.random()

				#Evolve wavefunction to the end of the period, and reset the binary search markers
				psi_new = dict(psi)
				for index in range(ind,self.num_tpoints):
					psi_new = apply_diagonal_operator(psi_new,self.dUs[index])
				start_index = ind #Record the time of the last jump!
				ind_left = ind
				ind_right = self.num_tpoints

			#end jump while block


			#Update psi, now that we are done with the window in this driving period
			psi = dict(psi_new)
			psis_mid.append(psi)

			#Apply the quarter-cycle evolution
			psi = self.quarter_cycle(psi)
			psis_end.append(psi)

		#end loop over periods

		return psis_mid, psis_end, jump_times, pre_jumps, post_jumps