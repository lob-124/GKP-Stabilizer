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

	overlaps = []
	with Pool(processes=num_threads) as p:
		for overlap_matrix in p.map(wells_obj.translation_matrix,well_nums,chunksize=int(ceil(len(well_nums)/num_threads))):
			overlaps.append(overlap_matrix)

	out_arr = array([well_nums,overlaps],dtype=object)
	save(outfile,out_arr)
