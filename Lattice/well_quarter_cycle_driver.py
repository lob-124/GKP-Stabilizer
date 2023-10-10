#Add directory with Frederik's code to path
from sys import path
path.insert(0,"../Frederik/")

from units import *
from _well_eigenstates import Wells

from numpy import array

if __name__ =="__main__":

	from sys import argv

	if len(argv) != 8:
		print("Usage: omega E_J D frac max_wells <outfile>")
		print("Units: \n -> omega in MHz\n -> E_J in GHz*hbar ")
		exit(0)

	####
	#### Extract command line args
	####
	omega = float(argv[1])*1e-3*GHz
	E_J = float(argv[2])*GHz*hbar
	D = int(argv[3])
	frac = float(argv[4])
	max_wells = int(argv[5])
	outfile = argv[6]


	####
	#### Set up and diagonalize wells using Well() class
	####
	wells_obj = Wells(omega,E_J,D)

	quarter_cycle = wells_obj.quarter_cycle()

	
	
	out_arr = array([],dtype=object)
	save(outfile,out_arr)
