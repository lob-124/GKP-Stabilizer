from numpy import diag,cos,array,complex64,zeros,pi,concatenate,linspace,exp,convolve,load
from numpy.linalg import norm,eigh 
from struct import pack

from multiprocessing import Pool

from analysis import *

from time import perf_counter

if __name__ =="__main__":
	#Add directory with Frederik's code to path
	from sys import path,argv
	path.insert(0,"../Frederik/")


	if len(argv) not in [11,13]:
		print("Usage: omega E_J D max_wells f num_threads num_avg <data_file> <save_file> <overlap_file> [num_level_norms num_well_norms (op)]")
		exit(0)

	omega = float(argv[1])*(1e-3*GHz)
	E_J = float(argv[2])*GHz*hbar
	D = int(argv[3])
	max_wells = int(argv[4])
	f = float(argv[5])
	num_threads = int(argv[6])
	num_avg = int(argv[7])
	data_file = argv[8]
	save_file = argv[9]
	overlap_file = argv[10]

	if len(argv) == 13:
		num_level_norms = int(argv[11])
		num_well_norms = int(argv[12])
	else:
		num_level_norms = 5
		num_well_norms = 3


	data = load(data_file,allow_pickle=True)

	num_samples = len(data[1])
	num_periods = len(data[1][0])


	#wvfn_mids = []
	wvfn_ends = []
	upper_norms = []
	outer_norms = []

	t1 = perf_counter()
	for i in range(num_samples):
		#For each sample, time-average the upper state norms and outer well norms
		#	over the last num_avg periods
		#Also store the wavefunction dicts as arguments to the parallel computation of the 
		#	stabilizers
		#wvfn_mids_this_sample = []
		wvfn_ends_this_sample = []
		_norm_up_mid , _norm_out_mid = 0.0 , 0.0
		_norm_up_end , _norm_out_end = 0.0 , 0.0
		for j in range(-num_avg,0):
			#psi_mid = data[1][i][j] 
			psi_end = data[1][i][j]

			#_norm_up_mid += upper_state_norm(psi_mid,num_level_norms)
			#_norm_out_mid += outer_well_norm(psi_mid,num_well_norms)
			_norm_up_end += upper_state_norm(psi_end,num_level_norms)
			_norm_out_end += outer_well_norm(psi_end,num_well_norms)

			#wvfn_mids.append(psi_mid)
			wvfn_ends.append(psi_end)

		upper_norms.append(array([_norm_up_mid,_norm_up_end])/num_avg)
		outer_norms.append(array([_norm_out_mid,_norm_out_end])/num_avg)

	upper_norms = array(upper_norms)
	outer_norms = array(outer_norms)

	t2 = perf_counter()
	print("Time elapsed: {}".format(t2-t1))


	print("stabilizer time!")

	#Load in the overlap matrices (used for computing Q stabilizer)
	overlap_data = load(overlap_file,allow_pickle=True)
	overlap_matrices = dict(zip(overlap_data[0],overlap_data[1])) 

	#Create the Analysis object (used in call to Pool.map())
	anal_obj = Analysis(omega,E_J,D,overlap_matrices,max_wells,f)

	# stab_1_mids , stab_2_mids = [],[]
	# with Pool(processes=num_threads) as p1:
	# 	count = 0
	# 	s1 , s2 = 0.0 , 0.0 
	# 	for stab_1, stab_2 in p1.map(anal_obj.stabilizers,wvfn_mids,chunksize=int(ceil(num_samples*num_avg/num_threads))):
	# 		s1 += stab_1
	# 		s2 += stab_2

	# 		count += 1
	# 		if count == num_avg:
	# 			stab_1_mids.append(s1/num_avg)
	# 			stab_2_mids.append(s2/num_avg)
	# 			s1 = 0.0
	# 			s2 = 0.0
	# 			count = 0
	print("oh baby")
	stab_1_ends , stab_2_ends = [],[]
	with Pool(processes=num_threads) as p2:
		count = 0
		s1 , s2 = 0.0 , 0.0 
		for stab_1, stab_2 in p2.map(anal_obj.stabilizers,wvfn_ends,chunksize=int(ceil(num_samples*num_avg/num_threads))):
			s1 += stab_1
			s2 += stab_2

			count += 1
			if count == num_avg:
				stab_1_ends.append(s1/num_avg)
				stab_2_ends.append(s2/num_avg)
				s1 = 0.0
				s2 = 0.0
				count = 0

	with open(save_file,"ab") as f:
		f.write(pack("i",num_samples))
		
		#f.write(pack("f"*num_samples,*upper_norms[:,0]))
		f.write(pack("f"*num_samples,*upper_norms[:,1]))
		#f.write(pack("f"*num_samples,*outer_norms[:,0]))
		f.write(pack("f"*num_samples,*outer_norms[:,1]))

		#f.write(pack("f"*num_samples,*stab_1_mids))
		f.write(pack("f"*num_samples,*stab_1_ends))
		#f.write(pack("f"*num_samples,*stab_2_mids))
		f.write(pack("f"*num_samples,*stab_2_ends))