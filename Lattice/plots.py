from numpy import array, mean, std,sqrt

from struct import unpack

import matplotlib.pyplot as plt
import matplotlib.colors as colors


if __name__ == "__main__":

	omega = 50
	max_wells = 20
	upper_num , outer_num = 5,3 

	t_start = 10
	t_stop = 200
	t_step = 10

	dts = [t for t in range(t_start,t_stop+1,t_step)]

	data_path = r"C:\Users\liamc\Documents\Caltech\GKP\Data\Analyzed"
	save_path = r"C:\Users\liamc\Documents\Caltech\GKP\Figures\Results\Lattice"

	upper_norms, upper_errs = [] , []
	outer_norms, outer_errs = [] , []
	stab_1s, stab_1_errs = [] , []
	stab_2s, stab_2_errs = [] , []
	for dt1 in dts:
		if dt1 % 100 == 0:
			print("On {} of {}".format(dt1,t_stop))
		
		_upper_norms , _upper_errs = [] , []
		_outer_norms , _outer_errs = [] , []
		_stab_1s , _stab_1_errs = [] , []
		_stab_2s , _stab_2_errs = [] , []	
		for dt2 in dts:
			filename = r"\Analyzed-{}MHz-{}wells-resistor2-dt1-{}-dt2-{}.dat".format(omega,max_wells,dt1,dt2)
			
			with open(data_path+filename,"rb") as f:
				num_samples = unpack("i",f.read(4))[0]

				uppers = unpack("f"*num_samples,f.read(4*num_samples))
				_upper_norms.append(mean(uppers))
				_upper_errs.append(std(uppers)/sqrt(num_samples-1))

				outers = unpack("f"*num_samples,f.read(4*num_samples))
				_outer_norms.append(mean(outers))
				_outer_errs.append(std(outers)/sqrt(num_samples-1))

				stab1s = unpack("f"*num_samples,f.read(4*num_samples))
				_stab_1s.append(mean(stab1s))
				_stab_1_errs.append(std(stab1s)/sqrt(num_samples-1))

				stab2s = unpack("f"*num_samples,f.read(4*num_samples))
				_stab_2s.append(mean(stab2s))
				_stab_2_errs.append(std(stab2s)/sqrt(num_samples-1))

		upper_norms.append(array(_upper_norms))
		upper_errs.append(array(_upper_errs))
		outer_norms.append(array(_outer_norms))
		outer_errs.append(array(_outer_errs))

		stab_1s.append(array(_stab_1s))
		stab_1_errs.append(array(_stab_1_errs))
		stab_2s.append(array(_stab_2s))
		stab_2_errs.append(array(_stab_2_errs))



	upper_norms = array(upper_norms)
	upper_errs = array(upper_errs)
	outer_norms = array(outer_norms)
	outer_errs = array(outer_errs)

	stab_1s = array(stab_1s)
	print(stab_1s.shape)
	stab_1_errs = array(stab_1_errs)
	stab_2s = array(stab_2s)
	stab_2_errs = array(stab_2_errs)



	#Plot the upper norms
	plt.figure()

	plt.pcolormesh(dts,dts,upper_norms.T,norm=colors.LogNorm(vmin=upper_norms.min(),vmax=upper_norms.max()))

	plt.xlabel(r"$\Delta_1$ (ps)",size=15)
	plt.ylabel(r"$\Delta_2$ (ps)",size=25)

	cbar = plt.colorbar()
	cbar.ax.tick_params(labelsize=20)
	cbar.set_label("Norm fraction in upper {} levels".format(upper_num),size=20)

	plt.savefig(save_path+r"\upper-norms-log.png")
	plt.close()

	plt.figure()

	plt.pcolormesh(dts,dts,upper_norms.T)

	plt.xlabel(r"$\Delta_1$ (ps)",size=15)
	plt.ylabel(r"$\Delta_2$ (ps)",size=15)

	cbar = plt.colorbar()
	cbar.ax.tick_params(labelsize=20)
	cbar.set_label("Norm fraction in upper {} levels".format(upper_num),size=20)

	plt.savefig(save_path+r"\upper-norms.png")
	plt.close()


	#Plot the outer norms
	plt.figure()

	plt.pcolormesh(dts,dts,outer_norms.T)

	plt.xlabel(r"$\Delta_1$ (ps)",size=15)
	plt.ylabel(r"$\Delta_2$ (ps)",size=25)

	cbar = plt.colorbar()
	cbar.ax.tick_params(labelsize=20)
	cbar.set_label("Norm fraction in outer {} wells".format(outer_num),size=20)

	plt.savefig(save_path+r"\outer-norms.png")
	plt.close()




	#Plot the first stabilizer
	plt.figure()

	plt.pcolormesh(dts,dts,stab_1s.T)

	plt.xlabel(r"$\Delta_1$ (ps)",size=15)
	plt.ylabel(r"$\Delta_2$ (ps)",size=25)

	cbar = plt.colorbar()
	cbar.ax.tick_params(labelsize=20)
	cbar.set_label(r"$<\exp(i\phi)>$",size=20)

	plt.savefig(save_path+r"\stab-1s.png")
	plt.close()


	#Plot the second stabilizer
	plt.figure()

	plt.pcolormesh(dts,dts,stab_2s.T)

	plt.xlabel(r"$\Delta_1$ (ps)",size=15)
	plt.ylabel(r"$\Delta_2$ (ps)",size=25)

	cbar = plt.colorbar()
	cbar.ax.tick_params(labelsize=20)
	cbar.set_label(r"$<\exp(4\pi iQ/\hbar)>$",size=20)

	plt.savefig(save_path+r"\stab-2s.png")
	plt.close()