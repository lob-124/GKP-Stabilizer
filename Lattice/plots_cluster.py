from analysis import *
from numpy import load,array,pi,real,imag,floor,sqrt

import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex":True,"text.latex.preamble":r"\usepackage{amsmath}"})

if __name__ =="__main__":
	#Add directory with Frederik's code to path
	from sys import path,argv
	path.insert(0,"../Frederik/")


	if len(argv) != 10:
		print("Usage: omega dt E_J D max_wells f <data_file> <save_path> <overlap_file>")
		exit(0)

	omega = float(argv[1])*(1e-3*GHz)
	dt = float(argv[2])*picosecond
	E_J = float(argv[3])*GHz*hbar
	D = int(argv[4])
	max_wells = int(argv[5])
	f = float(argv[6])
	data_file = argv[7]
	save_path = argv[8]
	overlap_file = argv[9]

	data = load(data_file,allow_pickle=True)
	#Load in the overlap matrices (used for computing Q stabilizer)
	overlap_data = load(overlap_file,allow_pickle=True)
	overlap_matrices = dict(zip(overlap_data[0],overlap_data[1])) 

	anal_obj = Analysis(omega,E_J,D,overlap_matrices,max_wells,f)

	num_samples = len(data[1])
	to_do = min(num_samples,2)
	num_periods = len(data[1][0])

	num_level_norms = 13
	num_well_norms = 3

	
	stride = 20

	T_LC = 0#(2*pi/omega)/picosecond
	T_tot = dt#T_LC/2 + dt#2*dt

	for i in range(to_do):
		# from matplotlib import animation

		# fig,ax = plt.subplots()
		# phi , wvfn = anal_obj.reconstruct_wavefunction2(data[1][i][0])
		# dphi = phi[1]-phi[0]
		# _norm = my_norm(data[1][i][0])

		# ax.plot(phi/pi,wvfn/sqrt(_norm*dphi))[0]
		# ax.set_xlabel(r"$\phi/\pi$",size=15)
		# ax.set_ylabel(r"$\psi$",size=15)

		# def animate(j):
		# 	print(j)
		# 	phi , wvfn = anal_obj.reconstruct_wavefunction2(data[1][i][j])
		# 	dphi = phi[1]-phi[0]
		# 	_norm = my_norm(data[1][i][0])

		# 	ax.clear()
		# 	ax.set_title("Period number {} of {}".format(j,num_periods))
		# 	ax.plot(phi,wvfn/sqrt(_norm*dphi))
			

		# anim = animation.FuncAnimation(fig,animate,list(range(num_periods)))
		# _writer = animation.FFMpegWriter(fps=2)
		# anim.save(save_path+"wvfn-animated-{}.gif".format(i),fps=2)
		# plt.close()

		psi = data[1][i][-1]


		phi , wvfn = anal_obj.reconstruct_wavefunction2(psi)


		#Plot the real/imag parts of the wavefunction
		# plt.figure()

		# plt.plot(phi/pi,real(wvfn),'r',label="Re")
		# plt.plot(phi/pi,imag(wvfn),'b',label="Im")

		# plt.xlabel(r"$\phi/\pi$",size=15)
		# plt.ylabel(r"Re[$\psi$] , Im[$\psi$]",size=15)
		# plt.legend()
		# plt.tick_params(labelsize=15)

		# plt.savefig(save_path+"wavfns-{}.png".format(i),bbox_inches="tight")
		# plt.close()

		# print("Wigner time!")

		# #Plot the wigner function
		# wig, phi_wig, q_wig = wigner(wvfn[::stride],phi[::stride])

		# plt.figure()

		# plt.pcolormesh(phi_wig/pi,q_wig,wig)

		# plt.xlabel(r"$\phi/\pi$",size=15)
		# plt.ylabel(r"$q$",size=15)
		# plt.tick_params(labelsize=15)

		# cbar = plt.colorbar()
		# cbar.ax.tick_params(labelsize=15)
		# cbar.set_label(r"$W(\phi,q)$",size=20)

		# plt.savefig(save_path+"wigner-{}.png".format(i))

		#Record in which periods quantum jumps occurred
		jumps = data[3][i]
		has_jump_1 = {}
		has_jump_2 = {}
		for jump in jumps:
			ind = int(floor(jump/T_tot))
			rem = jump % T_tot

			if rem <= dt:
				has_jump_1[ind] = True
			else:
				has_jump_2[ind] = True

		jumps_window_1 = []
		jumps_window_2 = []
		for j in range(num_periods):
			if has_jump_1.get(j):
				jumps_window_1.append(j)
			if has_jump_2.get(j):
				jumps_window_2.append(j)

		#print(jumps_window_1)
		#print(jumps_window_2)
		print(len(jumps))
		#print("Periods per jump: {}".format(num_periods/len(jumps_window_1)))

		print("Norm time!")

		#Plot the upper and outer norms as a function of time
		_norm_up_mid , _norm_out_mid = [],[]
		_norm_up_end , _norm_out_end = [],[]
		norms_mid,  norms_end = [] , []
		for j in range(num_periods):
			psi_mid = data[1][i][j]
			psi_end = data[2][i][j]

			_norm = my_norm(psi_mid)
			_norm_up_mid.append(upper_state_norm(psi_mid,num_level_norms)/_norm)
			_norm_out_mid.append(outer_well_norm(psi_mid,num_well_norms,max_wells)/_norm) 
			norms_mid.append(_norm)


			_norm = my_norm(psi_end)
			_norm_up_end.append(upper_state_norm(psi_end,num_level_norms)/_norm)
			_norm_out_end.append(outer_well_norm(psi_end,num_well_norms,max_wells)/_norm) 
			norms_end.append(_norm)


		plt.figure()

		plt.semilogy(list(range(1,num_periods+1)),_norm_up_mid,label="After resistor")
		plt.semilogy(list(range(1,num_periods+1)),_norm_up_end,label="After 1/4-cycle")
		#plt.vlines(jumps_window_1,0.0,1.0,colors='k',linestyles="dashed")
		#plt.vlines(jumps_window_2,0.0,1.0,colors='r',linestyles="dotted")

		plt.ylim((1e-6,1))
		plt.xlabel("Period Number",size=15)
		plt.ylabel("Norm fraction in upper {} wells".format(num_level_norms),size=15)
		plt.tick_params(labelsize=15)
		plt.legend()

		plt.savefig(save_path+"upper-norms-{}.png".format(i),bbox_inches="tight")
		plt.close()


		plt.figure()

		plt.plot(list(range(1,num_periods+1)),_norm_out_mid,label="After resistor")
		plt.plot(list(range(1,num_periods+1)),_norm_out_end,label="After 1/4-cycle")
		#plt.vlines(jumps_window_1,0.0,1.0,colors='k',linestyles="dashed")
		#plt.vlines(jumps_window_2,0.0,1.0,colors='r',linestyles="dotted")

		plt.xlabel("Period Number",size=15)
		plt.ylabel("Norm fraction in outer {} wells".format(num_well_norms),size=15)
		plt.tick_params(labelsize=15)
		plt.legend()

		plt.savefig(save_path+"outer-norms-{}.png".format(i),bbox_inches="tight")
		plt.close()


		plt.figure()

		# plt.plot(list(range(1,num_periods+1))[0:20],norms_mid[0:20],label="After resistor")
		# plt.plot(list(range(1,num_periods+1))[0:20],norms_end[0:20],label="After 1/4-cycle")
		# plt.vlines(jumps_window_1,0.0,1.0,colors='b',linestyles="dashed")
		# plt.vlines(jumps_window_2,0.0,1.0,colors='r',linestyles="dotted")

		plt.xlabel("Period Number",size=15)
		plt.ylabel(r"$|\psi|^2$".format(num_well_norms),size=15)
		plt.tick_params(labelsize=15)
		plt.legend()

		plt.savefig(save_path+"-norms-{}.png".format(i),bbox_inches="tight")
		plt.close()


		print("Stabilizer time!")

		#Plot the stabilizer epxectations as a function of time
		_stab_1_mid , _stab_2_mid = [],[]
		_stab_1_end , _stab_2_end = [],[]
		for j in range(num_periods):
			psi_mid = data[1][i][j]
			psi_end = data[2][i][j]
			
			stab_1 , stab_2 = anal_obj.stabilizers(psi_mid)
			_stab_1_mid.append(stab_1)
			_stab_2_mid.append(stab_2) 

			stab_1 , stab_2 = anal_obj.stabilizers(psi_end)
			_stab_1_end.append(stab_1)
			_stab_2_end.append(stab_2) 

		plt.figure()

		#plt.plot(list(range(1,num_periods+1)),_stab_1_mid,'+',label=r"$S_1$ odd 1/4-cycle")
		#plt.plot(list(range(1,num_periods+1)),_stab_2_end,'+',label=r"$S_2$ even 1/4-cycle")
		plt.plot(list(range(1,num_periods+1)),_stab_1_end,'.')
		#plt.vlines(jumps_window_1,0.0,1.0,colors='k',linestyles="dashed")
		#plt.vlines(jumps_window_2,0.0,1.0,colors='r',linestyles="dotted")

		plt.xlabel("Period Number",size=15)
		plt.ylabel(r"$\langle \psi | S^1 | \psi \rangle/\langle \psi|\psi \rangle$",size=15)
		plt.tick_params(labelsize=15)
		#plt.legend()

		plt.savefig(save_path+"stab-1-{}.png".format(i),bbox_inches="tight")
		plt.close()


		plt.figure()

		# plt.plot(list(range(1,num_periods+1)),_stab_2_mid,'+',label=r"$S_2$ odd 1/4-cycle")
		# plt.plot(list(range(1,num_periods+1)),_stab_1_end,'+',label=r"$S_1$ even 1/4-cycle")
		plt.plot(list(range(1,num_periods+1)),_stab_2_end,'.')
		#plt.vlines(jumps_window_1,0.0,1.0,colors='k',linestyles="dashed")
		#plt.vlines(jumps_window_2,0.0,1.0,colors='r',linestyles="dotted")

		plt.xlabel("Period Number",size=15)
		plt.ylabel(r"$\langle \psi | S^2 | \psi \rangle/\langle \psi|\psi \rangle$",size=15)
		plt.tick_params(labelsize=15)
		#plt.legend()

		plt.savefig(save_path+"stab-2-{}.png".format(i),bbox_inches="tight")
		plt.close()




		print("Fidelity time!")

		#Plot the upper and outer norms as a function of time
		# psi_0 = data[4]
		# def fidelity(psi):
		# 	return sum([psi_0.get(well,zeros(len(well_vec))).conj() @ well_vec for well,well_vec in psi.items()])

		# fid_mid , fid_end = [],[]
		# for j in range(num_periods):
		# 	psi_mid = data[1][i][j]
		# 	psi_end = data[2][i][j]

		# 	_norm = sqrt(my_norm(psi_mid))
		# 	fid_mid.append(fidelity(psi_mid)/_norm)
			
		# 	_norm = my_norm(psi_end)
		# 	fid_end.append(fidelity(psi_end)/_norm)


		# plt.figure()

		# plt.plot(list(range(1,num_periods+1)),fid_mid,label="After Resistor")
		# plt.plot(list(range(1,num_periods+1)),fid_end,label="After 1/4-cycle")
		# #plt.vlines(jumps_window_1,0.0,1.0,colors='b',linestyles="dashed")
		# #plt.vlines(jumps_window_2,0.0,1.0,colors='r',linestyles="dotted")

		# plt.xlabel("Period Number",size=15)
		# plt.ylabel("Fidelity with initial state",size=15)
		# plt.tick_params(labelsize=15)
		# plt.legend()

		# plt.savefig(save_path+"fidelity-{}.png".format(i),bbox_inches="tight")
		# plt.close()



		# #Find the peaks of the (absolute square of the) wavefunction within each well
		# peak_phis = []
		# peak_psis = []
		# for w in range(0,2*max_wells+1,2):
		# 	curr_well_start,curr_well_end = w*D , (w+1)*D 

		# 	max_ind = abs(wvfn[curr_well_start:curr_well_end]).argmax()

		# 	peak_psis.append(abs(wvfn[curr_well_start+max_ind])**2)
		# 	peak_phis.append(phi[curr_well_start+max_ind])


		# #Fit the peaks vs phi to estimate the envelop
		# from scipy.optimize import curve_fit
		
		# #First, a Gaussian 
		# def gaussian(x,x_0,sigma,C):
		# 	return C*exp(-((x-x_0)/sigma)**2)

		# #Estimate of the decay length: 4\pi/\Delta, where \Delta = (E_C/E_J)^1/4 is decay length 
		# #	of QHO in each well
		# E_C = (2*e_charge)**2/(2*anal_obj.C)
		# Delta = (E_C/E_J)**0.25
		# popt, _ = curve_fit(gaussian,peak_phis,peak_psis,p0=[0.0,4*pi/Delta,max(peak_psis)])

		# print(popt)


		# #Plot ansolute value of the wavefunction, with fitted Gaussian envelope overlaid
		# plt.figure()

		# plt.plot(phi/pi,abs(wvfn)**2,'b-',label="Wavefunction")
		# plt.plot(phi/pi,gaussian(phi,*popt),'r-',label="Fit")

		# plt.xlabel(r"$\phi/\pi$",size=15)
		# plt.ylabel(r"$|\psi|^2$",size=15)
		# plt.legend()
		# plt.tick_params(labelsize=15)

		# plt.savefig(save_path+"wavfn-fit-gaussian-{}.png".format(i),bbox_inches="tight")
		# plt.close()


		# #Second, a trig function 
		# def cosine(x,x_0,x_c,C):
		# 	return C*cos((x-x_0)/x_c)**2

		# popt, _ = curve_fit(cosine,peak_phis,peak_psis,p0=[0.0,4*max_wells,max(peak_psis)])


		# #Plot absolute value of the wavefunction, with fitted Gaussian envelope overlaid
		# plt.figure()

		# plt.plot(phi/pi,abs(wvfn)**2,'b-',label="Wavefunction")
		# plt.plot(phi/pi,cosine(phi,*popt),'r-',label="Fit")

		# plt.xlabel(r"$\phi/\pi$",size=15)
		# plt.ylabel(r"$|\psi|^2$",size=15)
		# plt.legend()
		# plt.tick_params(labelsize=15)

		# plt.savefig(save_path+"wavfn-fit-cosine-{}.png".format(i),bbox_inches="tight")
		# plt.close()

