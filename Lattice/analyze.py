import numpy as np
import matplotlib.pyplot as plt
from well_params import *

from scipy.fft import fft,fftfreq

def my_norm(psi):
    return sum(list(map(lambda v: np.linalg.norm(v)**2,psi.values())))

def reconstruct_wavefunction(psi):
    phis , psi_recon = [] , []
    
    for well_num , well_vec in psi.items():
        phi = phi_0 + 2*np.pi*well_num

        #Relevant operators in this well
        H_LC = (hbar**2/(4*e_charge**2))*diag(phi**2)/(2*L) + Q2/(2*C)
        H = H_LC - E_J*np.diag(np.cos(phi_0))
    
        #Eigenvalues and Eigenvectors
        _eigvals , _eigvecs = np.linalg.eigh(H)
        
        phis.append(phi)
        psi_recon.append(_eigvecs[:,:len(well_vec)]@well_vec)
    
    return phis,psi_recon

def reconstruct_wavefunction2(psi):
    phis , psi_recon = np.array([]) , np.array([],dtype=np.complex64)
    
    max_well = max(psi.keys())
    zero_vec = np.zeros(D,dtype=np.complex64)
    
    for i in range(-max_well,max_well+1):
        phi = phi_0 + 2*np.pi*i
        well_vec = psi.get(i)
        if well_vec is not None:
            #Relevant operators in this well
            H_LC = (hbar**2/(4*e_charge**2))*diag(phi**2)/(2*L) + Q2/(2*C)
            H = H_LC - E_J*np.diag(np.cos(phi_0))
    
            #Eigenvalues and Eigenvectors
            _eigvals , _eigvecs = np.linalg.eigh(H)
            psi_recon = np.concatenate((psi_recon,_eigvecs[:,:len(well_vec)]@well_vec))
        else:
            psi_recon = np.concatenate((psi_recon,zero_vec))
    
        phis = np.concatenate((phis,phi))
        
    return phis,psi_recon


def wigner(psi,x_in,pvec = None,print_progress=False ):
    x = x_in
    L = len(x_in)
    dx = x_in[1]-x_in[0]
    x_out = np.linspace(x[0]-x[-1],x[-1]-x[0],2*L-1)/2
    
    if type(pvec) ==type(None):
        
        prange = x_out
        p_out = prange
    else:
        prange = pvec
        p_out = prange 
        
    W = np.zeros((len(p_out),2*L-1))
    
    num_p = 0
    for p in prange:
        if print_progress and (num_p % 250 == 0):
            print(num_p)
        v= psi * np.exp(1j*x_in*p)
        # v2 = psi * exp()
        W[num_p] = np.convolve(v,v.conj())*dx/pi
        num_p+=1 
    
    
    # W = W.T 
    return W,x_out,p_out


def upper_state_norm(psi,num):
	"""
	Return the norm squared of the wavefunction in the num uppermost levels of each well 
	"""
	_norm = 0.0
	for v in psi.values():
		_norm += np.linalg.norm(v[-num:])**2

	return _norm


def outer_well_norm(psi,num):
	"""
	Return the norm squared of the wavefunction in the 2*num outermost wells (num on each side) 
	"""
	max_well = max(psi.keys())

	_norm = 0.0
	for i in range(num):
		_norm += np.linalg.norm(psi.get(-max_well+i,0.0))**2
		_norm += np.linalg.norm(psi.get(max_well-i,0.0))**2

	return _norm


def stabilizers(psi,overlaps,alpha=1.0):
	"""
	Return the expectation of the two stabilizer oeprators for the given state.
	"""
	stab_1 = 0.0
	stab_2 = 0.0

	for well_num, well_vec in psi.items():
		phi = phi_0 + 2*np.pi*well_num

		#Relevant operators in this well
		H_LC = (hbar**2/(4*e_charge**2))*diag(phi**2)/(2*L) + Q2/(2*C)
		H = H_LC - E_J*np.diag(np.cos(phi_0))
    
		#Eigenvalues and Eigenvectors
		_eigvals , _eigvecs = np.linalg.eigh(H)
        
		psi_recon = _eigvecs[:,:len(well_vec)]@well_vec

		#Add to the expected value of the first stabilizer
		stab_1 += sum(abs(psi_recon)**2*np.exp(-1j*alpha*phi))

		next_well = psi.get(well_num+2)
		if next_well is not None:
			stab_2 += next_well.conj() @ overlaps[well_num] @ well_vec

	_norm = my_norm(psi)
	return stab_1/_norm , stab_2/_norm


if __name__ =="__main__":
	from sys import argv

	if len(argv) not in [7,9]:
		print("Usage: stride1 stride 2 <data_file> <data_save_file> <overlap_file> <plot_path> [num_level_norms num_well_norms (op)]")
		exit(0)


	stride_1 = int(argv[1])
	stride_2 = int(argv[2])
	data_file = argv[3]
	data_save_file = argv[4]
	overlap_file = argv[5]
	plot_path = argv[6]

	if len(argv) == 9:
		num_level_norms = int(argv[7])
		num_well_norms = int(argv[8])
	else:
		num_level_norms = 5
		num_well_norms = 3


	data = np.load(data_file,allow_pickle=True)

	print("loaded!")
	print(data[0]["dt_1"]/picosecond)
	print(data[0]["dt_2"]/picosecond)

	num_samples = len(data[1])
	num_periods = len(data[1][0])


	wvfn_mids = []
	wvfn_ends = []

	for i in range(num_samples):
		print("Now on {} of {}".format(i,num_samples))
		phis , v_mid = reconstruct_wavefunction2(data[1][i][-1])
		norm_sq_mid = my_norm(data[1][i][-1])

		plt.figure(figsize=(10,10))
		plt.plot(phis/np.pi,abs(v_mid)**2/(dphi*norm_sq_mid))
		
		plt.xlabel(r"$\varphi/\pi$",size=20)
		plt.ylabel(r"$|\psi(\varphi)|^2$",size=20)
		plt.savefig(plot_path+"-mid-wavefunction-{}.png".format(i))
		plt.close()

		wvfn_mids.append(v_mid/np.sqrt(dphi*norm_sq_mid))

		#Take the FFT and plot it
		v_q = fft(v_mid)
		q = fftfreq(len(v_mid),d=dphi)
		norm_q , dq = np.linalg.norm(v_q) , q[1]-q[0]

		plt.figure(figsize=(10,10))
		plt.plot(q,abs(v_q)**2/(dq*norm_q**2))
		
		plt.xlabel(r"$Q$",size=20)
		plt.ylabel(r"$|\psi(Q)|^2$",size=20)
		plt.savefig(plot_path+"-mid-FT-wavefunction-{}.png".format(i))
		plt.close()


		#### **** REAPEAT FOR THE END OF THE LAST CYCLE **** ####


		phis, v_end = reconstruct_wavefunction2(data[2][i][-1])
		norm_sq_end = my_norm(data[1][i][-1])

		wvfn_ends.append(v_end/np.sqrt(dphi*norm_sq_mid))

		plt.figure(figsize=(10,10))
		plt.plot(phis/np.pi,abs(v_end)**2/norm_sq_end)
		
		plt.xlabel(r"$\varphi/\pi$",size=20)
		plt.ylabel(r"$|\psi(\varphi)|^2$",size=20)
		plt.savefig(plot_path+"-end-wavefunction-{}.png".format(i))
		plt.close()

		#Take the FFT and plot it
		v_q = fft(v_end)
		q = fftfreq(len(v_end),d=dphi)
		norm_q , dq = np.linalg.norm(v_q) , q[1]-q[0]

		inds = np.nonzero(abs(q) < 5)

		plt.figure(figsize=(10,10))
		plt.plot(q[inds],abs(v_q[inds])**2/(dq*norm_q**2))
		
		plt.xlabel(r"$Q$",size=20)
		plt.ylabel(r"$|\psi(Q)|^2$",size=20)
		plt.savefig(plot_path+"-end-FT-wavefunction-{}.png".format(i))
		plt.close()

	# 	if wvfn_end_total is not None:
	# 		wvfn_end_total += v_end/np.sqrt(dphi*norm_sq_end)
	# 	else:
	# 		wvfn_end_total = v_end/np.sqrt(dphi*norm_sq_end)



	# plt.figure(figsize=(10,10))
	# plt.plot(phis/np.pi,abs(wvfn_mid_total)**2/num_samples**2)
	
	# plt.xlabel(r"$\varphi/\pi$",size=20)
	# plt.ylabel(r"$|\psi(\varphi)|^2$",size=20)
	# plt.savefig(plot_path+"-mid-avg-wavefunction.png")
	# plt.close()


	# plt.figure(figsize=(10,10))
	# plt.plot(phis/np.pi,abs(wvfn_end_total)**2/num_samples**2)
	
	# plt.xlabel(r"$\varphi/\pi$",size=20)
	# plt.ylabel(r"$|\psi(\varphi)|^2$",size=20)
	# plt.savefig(plot_path+"-end-avg-wavefunction.png")
	# plt.close()


	print("done!")
	
	# f_upper = plt.figure(figsize=(10,10))
	# f_outer = plt.figure(figsize=(10,10))
	# for i,psi_list in enumerate(data[1]):
	# 	upper_norms_this_sample = []
	# 	outer_norms_this_sample = []
	# 	for psi in psi_list[::stride_1]:
	# 		full_norm = my_norm(psi)
	# 		upper_norm = upper_state_norm(psi,num_level_norms)
	# 		outer_norm = outer_well_norm(psi,num_well_norms)

	# 		upper_norms_this_sample.append(upper_norm/full_norm)
	# 		outer_norms_this_sample.append(outer_norm/full_norm)


	# 	plt.figure(f_upper)
	# 	plt.plot(list(range(1,num_periods+1,stride_1)),upper_norms_this_sample,'.')
	# 	plt.figure(f_outer)
	# 	plt.plot(list(range(1,num_periods+1,stride_1)),outer_norms_this_sample,'.')

	# plt.figure(f_upper)
	# plt.xlabel("Number of driving periods",size=20)
	# plt.ylabel(r"Projection onto top {} well levels".format(num_level_norms),size=20)
	# plt.tick_params(which="both",labelsize=20)
	# plt.savefig(plot_path+"-mid-upper-norms.png",bbox_inches="tight")
	# plt.close()


	# plt.figure(f_outer)
	# #plt.ylim(0.0,.01)
	# plt.xlabel("Number of driving periods",size=20)
	# plt.ylabel(r"Projection onto outermost {} wells".format(num_well_norms),size=20)
	# plt.tick_params(which="both",labelsize=20)
	# plt.savefig(plot_path+"-mid-outer-norms.png",bbox_inches="tight")
	# plt.close()


	# f_upper = plt.figure(figsize=(10,10))
	# f_outer = plt.figure(figsize=(10,10))
	# for i,psi_list in enumerate(data[2]):
	# 	upper_norms_this_sample = []
	# 	outer_norms_this_sample = []
	# 	for psi in psi_list[::stride_1]:
	# 		full_norm = my_norm(psi)
	# 		upper_norm = upper_state_norm(psi,num_level_norms)
	# 		outer_norm = outer_well_norm(psi,num_well_norms)

	# 		upper_norms_this_sample.append(upper_norm/full_norm)
	# 		outer_norms_this_sample.append(outer_norm/full_norm)


	# 	plt.figure(f_upper)
	# 	plt.plot(list(range(1,num_periods+1,stride_1)),upper_norms_this_sample,'-')
	# 	plt.figure(f_outer)
	# 	plt.plot(list(range(1,num_periods+1,stride_1)),outer_norms_this_sample,'-')

	# plt.figure(f_upper)
	# plt.xlabel("Number of driving periods",size=20)
	# plt.ylabel(r"Projection onto top {} well levels".format(num_level_norms),size=20)
	# plt.tick_params(which="both",labelsize=20)
	# plt.savefig(plot_path+"-end-upper-norms.png",bbox_inches="tight")
	# plt.close()


	# plt.figure(f_outer)
	# plt.xlabel("Number of driving periods",size=20)
	# plt.ylabel(r"Projection onto outermost {} wells".format(num_well_norms),size=20)
	# plt.tick_params(which="both",labelsize=20)
	# plt.savefig(plot_path+"-end-outer-norms.png",bbox_inches="tight")
	# plt.close()


	# print("stabilizer time!")

	# #Load in the overlap matrices (used for computing Q stabilizer)
	# overlap_data = np.load(overlap_file,allow_pickle=True)
	# overlap_matrices = dict(zip(overlap_data[0],overlap_data[1])) 


	# fig1 = plt.figure(figsize=(10,10))
	# fig2 = plt.figure(figsize=(10,10))
	# for i,psi_list in enumerate(data[1]):
	# 	print("Now on sample {} of {}".format(i+1,num_samples))
		
	# 	stab_1_this_sample = []
	# 	stab_2_this_sample = []
	# 	#for psi in psi_list:
	# 	for j in range(0,num_periods,stride_1):
	# 		if j %1000 == 0:
	# 			print("On period number {} of {}".format(j+1,num_periods))
	# 		stab_1 , stab_2 = stabilizers(data[1][i][j],overlap_matrices)
	# 		stab_1_this_sample.append(stab_1)
	# 		stab_2_this_sample.append(stab_2)

	# 	plt.figure(fig1)
	# 	plt.plot(list(range(1,num_periods+1,stride_1)),real(stab_1_this_sample),'-')
	# 	plt.figure(fig2)
	# 	plt.plot(list(range(1,num_periods+1,stride_1)),real(stab_2_this_sample),'-')

	# plt.figure(fig1)
	# plt.xlabel("Number of driving periods",size=20)
	# plt.ylabel(r"Real part of expected value of stabilizer 1",size=20)
	# plt.tick_params(which="both",labelsize=20)
	# plt.savefig(plot_path+"-mid-stab-1-lines.png",bbox_inches="tight")
	# plt.close()


	# plt.figure(fig2)
	# plt.xlabel("Number of driving periods",size=20)
	# plt.ylabel(r"Real part of expected value of stabilizer 2",size=20)
	# plt.tick_params(which="both",labelsize=20)
	# plt.savefig(plot_path+"-mid-stab-2-lines.png",bbox_inches="tight")
	# plt.close()


	# fig1 = plt.figure(figsize=(10,10))
	# fig2 = plt.figure(figsize=(10,10))
	# for i,psi_list in enumerate(data[1]):
	# 	print("Now on sample {} of {}".format(i+1,num_samples))
		
	# 	stab_1_this_sample = []
	# 	stab_2_this_sample = []
	# 	#for psi in psi_list:
	# 	for j in range(0,num_periods,stride_1):
	# 		if j %1000 == 0:
	# 			print("On period number {} of {}".format(j+1,num_periods))
	# 		stab_1 , stab_2 = stabilizers(data[2][i][j],overlap_matrices)
	# 		stab_1_this_sample.append(stab_1)
	# 		stab_2_this_sample.append(stab_2)

	# 	plt.figure(fig1)
	# 	plt.plot(list(range(1,num_periods+1,stride_1)),real(stab_1_this_sample),'-')
	# 	plt.figure(fig2)
	# 	plt.plot(list(range(1,num_periods+1,stride_1)),real(stab_2_this_sample),'-')

	# plt.figure(fig1)
	# plt.xlabel("Number of driving periods",size=20)
	# plt.ylabel(r"Real part of expected value of stabilizer 1",size=20)
	# plt.tick_params(which="both",labelsize=20)
	# plt.savefig(plot_path+"-end-stab-1-lines.png",bbox_inches="tight")
	# plt.close()


	# plt.figure(fig2)
	# plt.xlabel("Number of driving periods",size=20)
	# plt.ylabel(r"Real part of expected value of stabilizer 2",size=20)
	# plt.tick_params(which="both",labelsize=20)
	# plt.savefig(plot_path+"-end-stab-2-lines.png",bbox_inches="tight")
	# plt.close()

	# print("Onto wigner function")
	# wig_mid, wig_end = [],[]
	# q_vec = np.linspace(-5,5,num=1000)
	# for i in range(num_samples):
	# 	print("Doing sample {} of {}".format(i+1,num_samples))
	# 	#Wigner function
	# 	wig_calc , phi, q = wigner(wvfn_mids[i][::stride_1],phis[::stride_1],pvec=q_vec,print_progress=False)
	# 	wig_reduced = wig_calc[::stride_2,::stride_2]
	# 	phi_reduced = phi[::stride_2]
	# 	q_reduced = q[::stride_2]

	# 	wig_mid.append(wig_calc)
	# 	#np.save(data_save_file,{"phi":phi,"q":q,"wig":wig_calc})

	# 	#Wigner function
	# 	wig_calc , phi, q = wigner(wvfn_ends[i][::stride_1],phis[::stride_1],pvec=q_vec,print_progress=False)
	# 	wig_reduced = wig_calc[::stride_2,::stride_2]
	# 	phi_reduced = phi[::stride_2]
	# 	q_reduced = q[::stride_2]

	# 	wig_end.append(wig_calc)


	# fig = plt.figure(figsize=(10,10))
	# ax = plt.gca()

	# pcm = ax.pcolormesh(phi_reduced/np.pi,q_reduced,np.mean(wig_mid,axis=0))
	# ax.set_xlabel(r"$\varphi/\pi$",size=20)
	# ax.set_ylabel(r"$q$",size=20)
	# ax.tick_params(which="both",labelsize=12)
	# cbar = fig.colorbar(pcm,ax=ax)
	# cbar.ax.tick_params(labelsize=12)
	# cbar.set_label(r"$W(\varphi,q)$",size=15)

	# plt.savefig(plot_path+"-wig-mid.png")
	# plt.close()


	# fig = plt.figure(figsize=(10,10))
	# ax = plt.gca()

	# pcm = ax.pcolormesh(phi_reduced/np.pi,q_reduced,np.mean(wig_end,axis=0))
	# ax.set_xlabel(r"$\varphi/\pi$",size=20)
	# ax.set_ylabel(r"$q$",size=20)
	# ax.tick_params(which="both",labelsize=12)
	# cbar = fig.colorbar(pcm,ax=ax)
	# cbar.ax.tick_params(labelsize=12)
	# cbar.set_label(r"$W(\varphi,q)$",size=15)

	# plt.savefig(plot_path+"-wig-end.png")
	# plt.close()