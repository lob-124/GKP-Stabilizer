#Add directory with Frederik's code to path
from sys import path
path.insert(0,"../Frederik/")

from units import *
from _well_eigenstates import Wells

from numpy import array
from multiprocessing import Pool

if __name__ =="__main__":

	from sys import argv

	if len(argv) != 8:
		print("Usage: omega E_J D num_threads well_start well_stop <outfile>")
		print("Units: \n -> omega in MHz\n -> E_J in GHz*hbar ")
		exit(0)

	####
	#### Extract command line args
	####
	omega = float(argv[1])*1e-3*GHz
	E_J = float(argv[2])*GHz*hbar
	D = int(argv[3])
	num_threads = int(argv[4])
	well_start = int(argv[5])
	well_stop = int(argv[6])
	outfile = argv[7]


	####
	#### Set up and diagonalize wells using Well() class
	####
	wells_obj = Wells(omega,E_J,D)

	well_nums = list(range(well_start,well_stop+1))

	_eigvals = []
	mean_squares = []
	X1s , X2s = [] , []
	H_LCs = []
	with Pool(processes=num_threads) as p:
		for eigenvalues , ms, H_LC, X1, X2 in p.map(wells_obj.diagonalize_well,well_nums,chunksize=int(ceil(len(well_nums)/num_threads))):
			_eigvals.append(eigenvalues)
			mean_squares.append(ms)
			H_LCs.append(H_LC)
			X1s.append(X1)
			X2s.append(X2)
	
	quarter_cycle , well_vecs = wells_obj.quarter_cycle_wells(well_stop)
	out_arr = array([well_nums,_eigvals,well_vecs,mean_squares,H_LCs, X1s, X2s,quarter_cycle],dtype=object)
	save(outfile,out_arr)
