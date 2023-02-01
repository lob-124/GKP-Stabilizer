#Add directory with Frederik's code to path
from sys import path
path.insert(0,"../Frederik/")

import jump_static_optimized as js
from params import *

from numpy import linspace,zeros,sqrt,exp,save
import matplotlib.pyplot as plt


from struct import pack

if __name__ == "__main__":
	from sys import argv
	if len(argv) != 4:
		print("Usage: <num> <stride> <outpath>")
		exit(0)

	num = int(argv[1])
	stride = int(argv[2])
	outpath = argv[3]


	#Fourier components for the resistor coupling W(t) (assumed to be Gaussian)
	W_fourier = [sqrt(pi)*tau*exp(-Omega*q*(4j*t_0+q*Omega*tau**2)/4) for q in range(-q_max,q_max+1)]
	frequencies = [Omega*q for q in range(-q_max,q_max+1)]
	W_ft = (W_fourier,frequencies)

	#The parameters we are probing over
	lambda_vals = linspace(0,10,num=250)[1:]*THz
	T = 1e-3
	#temp_vals = linspace(0,10,num=250)[1:]*1e-3


	current_lambdas = lambda_vals[num*stride:(num+1)*stride]

	#The times and energies we maximize over
	t_vals = linspace(-T/2,T/2,num=100)
	E_vals = linspace(-6.0,0.0,num=100)



	#max_f = zeros((len(current_lambdas),len(temp_vals)))
	max_f = zeros((len(current_lambdas),len(E_vals)))


	#for i,_lambda in enumerate(lambda_vals):
	for i,_lambda in enumerate(current_lambdas):
		#for j,_T in enumerate(temp_vals):
		for j, E in enumerate(E_vals):
			max_val = 0.0
			for t in t_vals:
				#for E in E_vals:
				val = js.f(E,t,W_ft,dt_JJ,1.0,T,_lambda,1.0)
				max_val = max(max_val,abs(val))
			
			max_f[i,j] = max_val


	num_l = len(current_lambdas)
	#num_T = len(temp_vals)
	num_E = len(E_vals)
	with open(outpath,"wb") as f:
		f.write(pack('i',num_l))
		#f.write(pack('i',num_T))
		f.write(pack('i',num_E))
		f.write(pack('d'*num_l,*current_lambdas))
		for maxima in max_f:
			#f.write(pack('d'*num_T,*maxima))
			f.write(pack('d'*num_E,*maxima))

	# plt.figure(figsize=(10,10))

	# im = plt.pcolormesh(lambda_vals,temp_vals,max_f.T,cmap='hot')
	# plt.xlabel(r"$\lambda$ (THz)",size=20)
	# plt.ylabel(r"$T$ (K)",size=20)
	# plt.tick_params(which="both",labelsize=15)

	# cbar = plt.colorbar(im)
	# cbar.ax.tick_params(labelsize=15)
	# cbar.set_label(label=r'$\max_{E<0,t \in [-T/2,T/2]}|f(E,t)|$',size=20)

	# plt.savefig("../../Figures/f-heatmap.png")
	# plt.close()