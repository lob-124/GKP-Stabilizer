from analysis import *
from numpy import load,array,pi,real,imag,floor,sqrt,amin,amax

import matplotlib.pyplot as plt
import matplotlib.colors as cl

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
	
	stride = 20
	plot_min_wells = 1e-4
	plot_min_rungs = 1e-5
	cmap = "hot"

	for i in range(to_do):

		#Compute the norm in each well and rung over time
		norms_wells_mid , norms_wells_end = [] , []
		norms_rungs_mid , norms_rungs_end = [] , []
		for j in range(num_periods):
			if j %100 == 0:
				print("Now on period {} of {}".format(j,num_periods))
			psi_mid = data[1][i][j]
			psi_end = data[2][i][j]

			norm_mid , norm_end = my_norm(psi_mid) , my_norm(psi_end)


			##
			## First, in each well
			##
			norms_mid = []
			for w in range(-max_wells,max_wells+1): 
				norms_mid.append(norm(psi_mid.get(w,0.0))**2)
			norms_wells_mid.append(array(norms_mid)/norm_mid)

			norms_end = []
			for w in range(-max_wells,max_wells+1): 
				norms_end.append(norm(psi_end.get(w,0.0))**2)
			norms_wells_end.append(array(norms_end)/norm_end)

			##
			## Next, in each well level
			##
			norms_mid = []
			for well_vec in psi_mid.values():
				for k,elem in enumerate(well_vec):
					if len(norms_mid) >= k+1:
						norms_mid[k] += abs(elem)**2
					else:
						norms_mid.append(abs(elem)**2)
			norms_rungs_mid.append(array(norms_mid)/norm_mid)

			norms_end = []
			for well_vec in psi_end.values():
				for k,elem in enumerate(well_vec):
					if len(norms_end) >= k+1:
						norms_end[k] += abs(elem)**2
					else:
						norms_end.append(abs(elem)**2)
			norms_rungs_end.append(array(norms_end)/norm_end)



	# 	#Plot the norms after the resistor segment
	# 	#fig = plt.figure(figsize=(8,8))
	# 	#ax = fig.gca()
		fig,ax = plt.subplots(nrows=2,ncols=1,sharex=True)

		vmin_wells = max(amin(norms_wells_mid),plot_min_wells)
		vmax_wells = amax(norms_wells_mid)
		norm_wells = cl.LogNorm(vmin=vmin_wells,vmax=vmax_wells)

		im_wells = ax[0].pcolormesh(array(range(num_periods)),array(range(-max_wells,max_wells+1)),array(norms_wells_mid).T,
			norm=norm_wells,cmap=cmap)
		cbar_wells = fig.colorbar(im_wells,ax=ax[0])
		cbar_wells.ax.tick_params(labelsize=20)
		cbar_wells.set_label(label=r"\hspace{-15pt}Norm fraction in\\ well $w$",size=15)

		#plt.xlabel("Period Number",size=15)
		ax[0].set_ylabel(r"Well index $w$",size=15)
		ax[0].tick_params(which="both",labelsize=20)


		vmin_rungs = max(amin(norms_rungs_mid),plot_min_rungs)
		vmax_rungs = amax(norms_rungs_mid)
		norm_rungs = cl.LogNorm(vmin=vmin_rungs,vmax=vmax_rungs)

		num_rungs = len(norms_rungs_mid[0])

		im_rungs = ax[1].pcolormesh(array(range(num_periods)),array(range(num_rungs)),array(norms_rungs_mid).T,
			norm=norm_rungs,cmap=cmap)
		cbar_rungs = fig.colorbar(im_rungs,ax=ax[1])
		cbar_rungs.ax.tick_params(labelsize=20)
		cbar_rungs.set_label(label=r"\hspace{-15pt}Norm fraction in\\ well level $n$",size=15)

		ax[1].set_xlabel("Period Number",size=15)
		ax[1].set_ylabel(r"Well level $n$",size=15)
		ax[1].tick_params(which="both",labelsize=20)



		plt.savefig(save_path+"-norms-mid-{}.png".format(i),bbox_inches="tight")
		plt.close()




		fig,ax = plt.subplots(nrows=2,ncols=1,sharex=True)

		vmin_wells = max(amin(norms_wells_end),plot_min_wells)
		vmax_wells = amax(norms_wells_end)
		norm_wells = cl.LogNorm(vmin=vmin_wells,vmax=vmax_wells)

		im_wells = ax[0].pcolormesh(array(range(num_periods)),array(range(-max_wells,max_wells+1)),array(norms_wells_end).T,
			norm=norm_wells,cmap=cmap)
		cbar_wells = fig.colorbar(im_wells,ax=ax[0])
		cbar_wells.ax.tick_params(labelsize=20)
		cbar_wells.set_label(label=r"\hspace{-15pt}Norm fraction in\\ well $w$",size=15)

		#plt.xlabel("Period Number",size=15)
		ax[0].set_ylabel(r"Well index $w$",size=15)
		ax[0].tick_params(which="both",labelsize=20)


		vmin_rungs = max(amin(norms_rungs_end),plot_min_rungs)
		vmax_rungs = amax(norms_rungs_end)
		norm_rungs = cl.LogNorm(vmin=vmin_rungs,vmax=vmax_rungs)

		num_rungs = len(norms_rungs_mid[0])

		im_rungs = ax[1].pcolormesh(array(range(num_periods)),array(range(num_rungs)),array(norms_rungs_end).T,
			norm=norm_rungs,cmap=cmap)
		# im_rungs = ax[1].pcolormesh(array(range(num_periods)),array(range(num_rungs)),array(norms_rungs_end).T,
		# 	cmap=cmap)
		cbar_rungs = fig.colorbar(im_rungs,ax=ax[1])
		cbar_rungs.ax.tick_params(labelsize=20)
		cbar_rungs.set_label(label=r"\hspace{-15pt}Norm fraction in\\ well level $n$",size=15)

		ax[1].set_xlabel("Period Number",size=15)
		ax[1].set_ylabel(r"Well level $n$",size=15)
		ax[1].tick_params(which="both",labelsize=20)



		plt.savefig(save_path+"-norms-end-{}.png".format(i),bbox_inches="tight")
		plt.close()


		# #Now for the well norms after 1/4-cycle
		# fig = plt.figure(figsize=(8,8))
		# ax = fig.gca()

		# vmin = max(amin(norms_wells_end),plot_min)
		# vmax = amax(norms_wells_end)
		# color_norm = cl.LogNorm(vmin=vmin,vmax=vmax)

		# im = ax.pcolormesh(array(range(num_periods)),array(range(-max_wells,max_wells+1)),array(norms_wells_end).T,
		# 	norm=color_norm,cmap="Reds")
		# cbar = fig.colorbar(im,ax=ax)
		# cbar.ax.tick_params(labelsize=20)
		# cbar.set_label(label=r"Norm fraction in well $w$",size=15)

		# plt.xlabel("Period Number",size=15)
		# plt.ylabel(r"Well index $w$",size=15)
		# plt.tick_params(which="both",labelsize=20)

		# plt.savefig(save_path+"-well-norms-end-{}.png".format(i))
		# plt.close()


		# #Now the rung norms after the resistor segment
		# fig = plt.figure(figsize=(8,8))
		# ax = fig.gca()

		# vmin = max(amin(norms_rungs_mid),plot_min)
		# vmax = amax(norms_rungs_mid)
		# color_norm = cl.LogNorm(vmin=vmin,vmax=vmax)

		# num_rungs = len(norms_rungs_mid[0])

		# im = ax.pcolormesh(array(range(num_periods)),array(range(num_rungs)),array(norms_rungs_mid).T,
		# 	norm=color_norm,cmap="Reds")
		# cbar = fig.colorbar(im,ax=ax)
		# cbar.ax.tick_params(labelsize=20)
		# cbar.set_label(label=r"Norm fraction in well level $n$",size=15)

		# plt.xlabel("Period Number",size=15)
		# plt.ylabel(r"Well level $n$",size=15)
		# plt.tick_params(which="both",labelsize=20)

		# plt.savefig(save_path+"-well-rungs-mid-{}.png".format(i))
		# plt.close()



		# #Now the rung norms after the 1/4-cycle
		# fig = plt.figure(figsize=(8,8))
		# ax = fig.gca()

		# vmin = max(amin(norms_rungs_end),plot_min)
		# vmax = amax(norms_rungs_mid)
		# color_norm = cl.LogNorm(vmin=vmin,vmax=vmax)

		# num_rungs = len(norms_rungs_end[0])

		# im = ax.pcolormesh(array(range(num_periods)),array(range(num_rungs)),array(norms_rungs_end).T,
		# 	norm=color_norm,cmap="Reds")
		# cbar = fig.colorbar(im,ax=ax)
		# cbar.ax.tick_params(labelsize=20)
		# cbar.set_label(label=r"Norm fraction in well level $n$",size=15)

		# plt.xlabel("Period Number",size=15)
		# plt.ylabel(r"Well level $n$",size=15)
		# plt.tick_params(which="both",labelsize=20)

		# plt.savefig(save_path+"-well-rungs-end-{}.png".format(i))
		# plt.close()

	S_mid , S_end = [] , []
	for i in range(to_do):

		#Compute the spin expectations over time
		S_z_mid , S_x_mid , S_y_mid = [] , [] , []
		S_z_end , S_x_end , S_y_end = [] , [] , []
		for j in range(2300,2500):#range(num_periods):
			if j %10 == 0:
				print("Now on period {} of {}".format(j,num_periods))
			#psi_mid = data[1][i][j]
			psi_end = data[2][i][j]

			# S_z_mid.append(anal_obj.expectation_sz(psi_mid))
			# S_x_mid.append(anal_obj.expectation_sx(psi_mid))
			# S_y_mid.append(anal_obj.expectation_sy(psi_mid))


			S_z_end.append(anal_obj.expectation_sz(psi_end))
			S_x_end.append(anal_obj.expectation_sx(psi_end))
			S_y_end.append(anal_obj.expectation_sy(psi_end))

		
		# plt.figure(figsize=(8,8))

		# plt.plot(array(range(num_periods)),S_x_mid,label=r"$\langle \psi | S_x | \psi\rangle$")
		# plt.plot(array(range(num_periods)),S_y_mid,label=r"$\langle \psi | S_y | \psi\rangle$")
		# plt.plot(array(range(num_periods)),S_z_mid,label=r"$\langle \psi | S_z | \psi\rangle$")

		# plt.xlabel("Period Number",size=15)
		# plt.ylabel("Spin expectation",size=15)
		# plt.tick_params(which="both",labelsize=15)
		# plt.legend()

		# plt.savefig(save_path+"-spins-corrected-mid-{}.png".format(i))
		# plt.close()


		#plt.figure(figsize=(8,8))
		fig,ax = plt.subplots(nrows=3,ncols=1,sharex=True)
		#range(num_periods)
		ax[0].plot(array(range(2300,2500)),S_x_end,'b.',label=r"$\langle \psi | S_x | \psi\rangle$")
		ax[0].set_ylabel(r"$\langle S_x \rangle$",size=15)
		ax[0].tick_params(which="both",labelsize=20)
		ax[1].plot(array(range(2300,2500)),S_y_end,'g.',label=r"$\langle \psi | S_y | \psi\rangle$")
		ax[1].set_ylabel(r"$\langle S_y \rangle$",size=15)
		ax[1].tick_params(which="both",labelsize=20)
		ax[2].plot(array(range(2300,2500)),S_z_end,'r.',label=r"$\langle \psi | S_z | \psi\rangle$")
		ax[2].set_ylabel(r"$\langle S_z \rangle$",size=15)
		ax[2].set_xlabel("Period number",size=15)
		ax[2].tick_params(which="both",labelsize=20)


		# plt.xlabel("Period Number",size=15)
		# plt.ylabel("Spin expectation",size=15)
		# plt.tick_params(which="both",labelsize=15)
		# plt.legend()

		plt.savefig(save_path+"-spins-end-{}.png".format(i),bbox_inches="tight")
		plt.close()