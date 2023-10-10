from units import *

from numpy import sqrt,pi,zeros
from numpy.random import default_rng
from numpy.linalg import norm,svd
from math import factorial



#Function to apply operators to psi (where the operator is assumed to act only WITHIN each well)
def apply_diagonal_operator(psi,O):
	"""
	Given a wavefunction and operator O, represented in our grid structure, apply the 
		unitary to psi. We assume U does not couple between wells
	"""
	return dict(map(lambda tup: (tup[0],O[tup[0]] @ tup[1]), psi.items()))


#Function to apply operators to psi (where the operator is assumed to couple wells differing by n)
def apply_interwell_operator(psi,O,n):
	"""
	Given a wavefunction and operator O, represented in our grid structure, apply the 
		unitary to psi. We assume U couples only between wells separated by n
	"""
	min_well , max_well = min(psi.keys()) , max(psi.keys())
	to_return = {}
	for well_num,v in psi.items():
		if (well_num + n > max_well) or (well_num + n < min_well):
			continue
		else:
			to_return[well_num+n] = O[well_num] @ psi[well_num]

	return to_return



#Function to compute the norm squared of psi
def my_norm(psi):
	return sum(list(map(lambda v: norm(v)**2,psi.values())))




class SSE():

	#We represent the wavefunctions as a dictionary, with psi[k] being a vector
	#	of the components of psi in the basis of eigenfunctions of well k
	#More precisely:
	#			psi[k][i] = <k,i|psi>
	#	where |k,i> is the ith level in well k.
	#We also only keep track of the wells where psi has support, so psi.items() gives
	#	a list of the well indices where components of psi are non-zero 

	def __init__(self,E_J,num_periods,max_wells,Us_window1,dUs_window1,L1s_window1,L2s_window1,H_LC_window2,overlaps_window2,num_tpoints1,dt_2,dt_window2):
		self.E_J = E_J
		self.num_periods = num_periods
		self.max_wells = max_wells
		self.Us_window1 = Us_window1
		self.dUs_window1 = dUs_window1
		self.L1s_window1 = L1s_window1
		self.L2s_window1 = L2s_window1
		self.H_LC_window2 = H_LC_window2
		self.overlaps_window2 = overlaps_window2
		self.num_tpoints1 = num_tpoints1
		self.dt_2 = dt_2
		self.dt_window2 = dt_window2

		self.delta = 2	#The number of wells to translate by (fixed by the choice of L,C)
		self.tol = self.dt_window2/100



	#Function to compute the evolution (t -> t+dt) during the second window
	def evolution_window_2(self,psi,dt,order=5):
		if order < 1:
			print("Error: order must be at least one.")
			exit(1)

		psi_tot = dict(psi)
		dpsi_prev = dict(psi)
		for i in range(1,order+1):
			#Apply H_LC to the previous order's correction
			dpsi_curr = apply_diagonal_operator(dpsi_prev,self.H_LC_window2)

			#Apply H_JJ (<-> well translation) to the previous order's correction
			for well_num, v in dpsi_prev.items():
				#print("On well {}".format(well_num))
				if well_num >= -self.max_wells + self.delta:
					#Translation backward by two wells
					dpsi_curr[well_num-self.delta] = dpsi_curr.get(well_num-self.delta,0.0+0j) - (self.E_J/2)*(self.overlaps_window2[well_num-self.delta].conj().T @ v)
				if well_num <= self.max_wells - self.delta:
					#Translation forward by two wells
					dpsi_curr[well_num+self.delta] = dpsi_curr.get(well_num+self.delta,0.0+0j) - (self.E_J/2)*(self.overlaps_window2[well_num] @ v)

			for well_num,dv in dpsi_curr.items():
				psi_tot[well_num] = psi_tot.get(well_num, 0.0+0j) + ((-1j*dt/hbar)**i/factorial(i))*dv

			dpsi_prev = dpsi_curr

		return psi_tot




	def SSE_evolution(self,psi_0,seed):
		"""
		Perform one trajectory of the SSE evolution.

		Params:
			psi_0: dict
				Initial wavefunction. Should be a dictionary, whose keys are the well numbers and items
					are the wavefunction in the basis of eigenstates of each well
			num_periods: int
				The number of driving periods to simulate for
			seed: int
				The seed for the RNG of this run

		Returns:
			A list of the wavefunctions (each one a dict as described above) after each driving period
		"""

		rng = default_rng(seed)
		r = rng.random()

		psi = dict(psi_0)
		
		psis_mid , psis_end = [] , []
		for i in range(self.num_periods):
			if i % 50 == 0:
				print("Currently on period {} of {}".format(i,self.num_periods))
			#Indices used by binary search within the window
			start_index = 0
			ind_left = 0
			ind_right = self.num_tpoints1

			
			#Perform the evolution over the first window
			psi_new = apply_diagonal_operator(psi,self.Us_window1[-1])

			#Check if there was a quantum jump within this window
			while (1 - my_norm(psi_new)) > r:
				#Binary search for the time at which the jump occurred
				while (ind_right - ind_left) > 1:
					ind_mid = round((ind_right+ind_left)/2)

					#Evolve the wavefunction to the time given by ind_mid
					if start_index != 0:
						#If we've already had a jump thios period, evolve from the time
						#	of the last jump
						#NB: psi is the wavefunction at the previous jump time (see ****)
						psi_temp = dict(psi)
						for index in range(start_index,ind_mid):
							psi_temp = apply_diagonal_operator(psi_temp,self.dUs_window1[index])
					else:
						#Otherwise, evolve from the beginning of the period
						psi_temp = apply_diagonal_operator(psi,self.Us_window1[ind_mid])

					#Check if the jump has happened yet
					if (1 - my_norm(psi_temp)) >= r:
						ind_right = ind_mid
					else:
						ind_left = ind_mid

				#end binary search while loop

				#We've now found the time t at which the jump occurred
				ind = ind_right
				#jump_time = i*T + t_vals_window1[ind]

				#Advance the wavefunction to time t (****)
				if start_index != 0:
					#Evolve from prior jump (if needed)
					for index in range(start_index,ind):
						psi = apply_diagonal_operator(psi,self.dUs_window1[index])
				else:
					#Otherwise, evolve from beginning of period
					psi = apply_diagonal_operator(psi,self.Us_window1[ind])

				#Determine the type of jump, and jump the wavefunction
				_psi1 = apply_diagonal_operator(psi,self.L1s_window1)
				_psi2 = apply_diagonal_operator(psi,self.L2s_window1)
				p_1 , p_2 = my_norm(_psi1) , my_norm(_psi2)
				if rng.random() < p_1/(p_1+p_2):
					psi = dict([(well,vec/sqrt(p_1)) for well,vec in _psi1.items()])
				else:
					psi = dict([(well,vec/sqrt(p_2)) for well,vec in _psi2.items()])

				#Reset the random variable r
				r = rng.random()

				#Evolve psi to the end of the period, and reset the binary search markers
				for index in range(ind,self.num_tpoints1):
					psi_new = apply_diagonal_operator(psi,self.dUs_window1[index])
				ind_left = ind
				ind_right = self.num_tpoints1

			#end jump while block


			#Update psi, now that we are done with this window driving period
			psi = dict(psi_new)
			psis_mid.append(psi)

			#Now we evolve over the second window. Note the resistor is not on here, so this is 
			#	unitary evolution
			_t = 0
			while _t + self.dt_window2 < self.dt_2:
				psi = self.evolution_window_2(psi,self.dt_window2)
				_t += self.dt_window2

			if self.dt_2-_t < self.tol:
				psi = self.evolution_window_2(psi,self.dt_2-_t)

			#Store the wavefunction at the end of this driving period
			psis_end.append(psi)


		#end loop over periods

		return psis_mid , psis_end




#### ****
#### ****
##  Another version of the class, that allows for the resistor to be on in window 2
#### ****
#### ****
class SSE2():

	#We represent the wavefunctions as a dictionary, with psi[k] being a vector
	#	of the components of psi in the basis of eigenfunctions of well k
	#More precisely:
	#			psi[k][i] = <k,i|psi>
	#	where |k,i> is the ith level in well k.
	#We also only keep track of the wells where psi has support, so psi.items() gives
	#	a list of the well indices where components of psi are non-zero 

	def __init__(self,E_J,num_periods,max_wells,Us_window1,dUs_window1,L1s_window1,L2s_window1,L1s_window2,L2s_window2,H_LC_window2,overlaps_window2,dt_1,num_tpoints1,dt_2,dt_window2,T_LC,W):
		self.E_J = E_J
		self.num_periods = num_periods
		self.max_wells = max_wells
		self.Us_window1 = Us_window1
		self.dUs_window1 = dUs_window1
		self.L1s_window1 = L1s_window1
		self.L2s_window1 = L2s_window1
		self.L1s_window2 = L1s_window2
		self.L2s_window2 = L2s_window2

		self.delta = 2	#The number of wells to translate by (fixed by the choice of L,C)

		#Construct the non-hermitian part of the Effective SSE Hamiltonian in window 2
		#NB: we assume L1 is translation FORWARD by two wells, and L2 BACKWARD by two wells
		self.H_eff_NH_window2 = {}
		for well_num , L in self.L1s_window2.items():
			L_plus = L1s_window2.get(well_num+self.delta)
			if L_plus is not None:
				self.H_eff_NH_window2[well_num] = -0.5j*L_plus.conj().T @ L
			else:
				self.H_eff_NH_window2[well_num] = zeros(L.shape,dtype=complex64)
		
		for well_num , L in self.L2s_window2.items():
			L_minus = L2s_window2.get(well_num-self.delta)
			if L_minus is not None:
				self.H_eff_NH_window2[well_num] = self.H_eff_NH_window2.get(well_num,0.0+0.0j) -0.5j*L_minus.conj().T @ L
		
		self.H_LC_window2 = H_LC_window2
		self.overlaps_window2 = overlaps_window2
		self.dt_1 = dt_1
		self.num_tpoints1 = num_tpoints1
		self.t_vals_window1 = linspace(0,dt_1,num_tpoints1+1)
		self.dt_2 = dt_2

		self.dt_window2 = dt_window2
		for w in range(-max_wells,max_wells+1):
			_,S,_ = svd(self.H_LC_window2[w] + self.H_eff_NH_window2.get(w,0.0))
			self.dt_window2 = min(self.dt_window2,hbar/(16*S[0]))

		#The LC period and driving period
		self.T_LC = T_LC
		self.T = T_LC/2 + dt_2 + dt_1

		#The window function
		self.W = W

	

	#Function to compute the evolution (t -> t+dt) during the second window
	def evolution_window_2(self,psi,t,dt,order=5):
		if order < 1:
			print("Error: order must be at least one.")
			exit(1)

		psi_tot = dict(psi)
		dpsi_prev = dict(psi)
		for i in range(1,order+1):
			#Apply H_LC and H_eff_NH to the previous order's correction
			dpsi_curr = apply_diagonal_operator(dpsi_prev,self.H_LC_window2)
			dpsi_curr2 = apply_diagonal_operator(dpsi_prev,self.H_eff_NH_window2)

			# for well_num in psi.keys():
			# 	H_eff = self.H_LC_window2[well_num] + self.H_eff_NH_window2[well_num]
			# 	print("Well {}: Norm of H_eff = {}".format(well_num,norm(H_eff)))
			# 	print("\tAssociated timescale: {} ps".format((hbar/norm(H_eff))/picosecond))

			#Add the two above together
			for well_num,v in dpsi_curr2.items():
				#NB: this is the only place we need the window function W
				dpsi_curr[well_num] = dpsi_curr.get(well_num,0.0+0.0j) + self.W(t)**2*v

			#Apply H_JJ (<-> well translation) and  to the previous order's correction
			for well_num, v in dpsi_prev.items():
				#print("On well {}".format(well_num))
				if well_num >= -self.max_wells + self.delta:
					#Translation backward by two wells
					dpsi_curr[well_num-self.delta] = dpsi_curr.get(well_num-self.delta,0.0+0j) - (self.E_J/2)*(self.overlaps_window2[well_num-self.delta].conj().T @ v)
				if well_num <= self.max_wells - self.delta:
					#Translation forward by two wells
					dpsi_curr[well_num+self.delta] = dpsi_curr.get(well_num+self.delta,0.0+0j) - (self.E_J/2)*(self.overlaps_window2[well_num] @ v)

			for well_num,dv in dpsi_curr.items():
				psi_tot[well_num] = psi_tot.get(well_num, 0.0+0j) + ((-1j*dt/hbar)**i/factorial(i))*dv

			dpsi_prev = dpsi_curr

		return psi_tot




	def SSE_evolution(self,psi_0,seed):
		"""
		Perform one trajectory of the SSE evolution.

		Params:
			psi_0: dict
				Initial wavefunction. Should be a dictionary, whose keys are the well numbers and items
					are the wavefunction in the basis of eigenstates of each well
			num_periods: int
				The number of driving periods to simulate for
			seed: int
				The seed for the RNG of this run

		Returns:
			A list of the wavefunctions (each one a dict as described above) after each driving period
		"""

		rng = default_rng(seed)
		r = rng.random()

		psi = dict(psi_0)
		
		psis_mid , psis_end = [] , []
		jump_times = []
		for i in range(self.num_periods):
			if i % 25 == 0:
				print("Now on period {} of {} for seed {}".format(i,self.num_periods,seed))
			#Indices used by binary search within the window
			start_index = 0
			ind_left = 0
			ind_right = self.num_tpoints1

			
			#Perform the evolution over the first window
			psi_new = apply_diagonal_operator(psi,self.Us_window1[-1])

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
							psi_temp = apply_diagonal_operator(psi_temp,self.dUs_window1[index])
					else:
						#Otherwise, evolve from the beginning of the period
						psi_temp = apply_diagonal_operator(psi,self.Us_window1[ind_mid])

					#Check if the jump has happened yet
					if (1 - my_norm(psi_temp)) >= r:
						ind_right = ind_mid
					else:
						ind_left = ind_mid

				#end binary search while loop

				#We've now found the time t at which the jump occurred
				ind = ind_right
				jump_times.append(i*self.T + self.t_vals_window1[ind])

				#Advance the wavefunction to time t (****)
				if start_index != 0:
					#Evolve from prior jump (if needed)
					for index in range(start_index,ind):
						psi = apply_diagonal_operator(psi,self.dUs_window1[index])
				else:
					#Otherwise, evolve from beginning of period
					psi = apply_diagonal_operator(psi,self.Us_window1[ind])

				#Determine the type of jump, and jump the wavefunction
				_psi1 = apply_diagonal_operator(psi,self.L1s_window1)
				_psi2 = apply_diagonal_operator(psi,self.L2s_window1)
				p_1 , p_2 = my_norm(_psi1) , my_norm(_psi2)
				if rng.random() < p_1/(p_1+p_2):
					psi = dict([(well,vec/sqrt(p_1)) for well,vec in _psi1.items()])
				else:
					psi = dict([(well,vec/sqrt(p_2)) for well,vec in _psi2.items()])

				#Reset the random variable r
				r = rng.random()

				#Evolve psi to the end of the period, and reset the binary search markers
				for index in range(ind,self.num_tpoints1):
					psi_new = apply_diagonal_operator(psi,self.dUs_window1[index])
				ind_left = ind
				ind_right = self.num_tpoints1

			#end jump while block


			#Update psi, now that we are done with the first window in this driving period
			psi = dict(psi_new)
			psis_mid.append(psi)

			#Now we evolve over the second window. Note we do not construct the evolution operators
			#	here, so we simply evolve step-by-step, checking for jumps as we go
			_t = 0
			while _t + self.dt_window2 < self.dt_2:
				psi = self.evolution_window_2(psi,_t,self.dt_window2)
				_t += self.dt_window2

				#If we detect a jump, sample and apply the appropriate jump operator
				if (1 - my_norm(psi_new)) > r:
					_psi1 = apply_interwell_operator(psi,self.L1s_window2,self.delta)
					_psi2 = apply_interwell_operator(psi,self.L2s_window2,-self.delta)
					p_1 , p_2 = my_norm(_psi1) , my_norm(_psi2)
					if rng.random() < p_1/(p_1+p_2):
						psi = dict([(well,vec/sqrt(p_1)) for well,vec in _psi1.items()])
					else:
						psi = dict([(well,vec/sqrt(p_2)) for well,vec in _psi2.items()])

					#Reset the random variable r
					r = rng.random()

					#Record the jump time
					jump_times.append(i*self.T + self.dt_1 + self.T_LC/4 + _t)


			#Evolve to the end of the period (in case dt_window2 doesn't divide dt_2)
			#rint("and the other thing...")
			if _t < self.dt_2:
				psi = self.evolution_window_2(psi,_t,self.dt_2-_t)

				#Check for a jump here at the end of the period
				if (1 - my_norm(psi_new)) > r:
					_psi1 = apply_interwell_operator(psi,self.L1s_window2,self.delta)
					_psi2 = apply_interwell_operator(psi,self.L2s_window2,-self.delta)
					p_1 , p_2 = my_norm(_psi1) , my_norm(_psi2)
					if rng.random() < p_1/(p_1+p_2):
						psi = dict([(well,vec/sqrt(p_1)) for well,vec in _psi1.items()])
					else:
						psi = dict([(well,vec/sqrt(p_2)) for well,vec in _psi2.items()])

					#Reset the random variable r
					r = rng.random()

					#Record the jump time
					jump_times.append(i*self.T + self.dt_1 + self.T_LC/4 + _t)

			#We're now done with this driving period
			#Store the wavefunction at the end of this driving period
			psis_end.append(psi)
			#rint(my_norm(psi))


		#end loop over periods

		return psis_mid , psis_end, jump_times

#end SSE2