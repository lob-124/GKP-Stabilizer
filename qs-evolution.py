#Add directory with Frederik's code to path
from sys import path
path.append("./Frederik/")


from units import *
from basic import get_tmat

from numpy import exp,cos,sign,sqrt,pi,load,save,linspace,zeros,eye,complex64,array
from numpy.random import default_rng
from numpy.linalg import norm, eigh, svd
from math import factorial

from time_evo import time_evolve
#from SSE import SSE,SSE2

#from multiprocessing import Pool



def g(omega,Temp,Lambda,omega_0=1):
	"""
	Bath jump correlator g(\omega) = \sqrt{J(\omega)/2pi}
	"""
	if abs(omega) < 1e-14:
		J = Temp/omega_0
	else:
		S_0 = abs(omega)*exp(-(omega**2)/(2*Lambda**2))/omega_0
		BE = (1)/(1-exp(-omega/Temp))*sign(omega)
		J = S_0 * BE

	return sqrt(J/(2*pi))


def bump_function(u,k=2,tol=1e-3):
	"""
	A smooth function which is identically zero for  u <= 0 or u >= 1, and quickly rises to unity
		for u in 0 < u < 1.
	Params:
		u: float
			The argument of the function
		k: int
			Parameter controlling how quickly the function rises to 1 on 0 <= u <= 1. Must be a positive 
				integer. Defaults to 2.

	Returns:
		Value of the bump function at the given u
	"""
	if u <= tol or u >= 1-tol:
		return 0.0
	else:
		x = 2*u - 1
		return 1 - exp(-1/(x**(2*k)-1))/(exp(-1/(x**(2*k)-1)) + exp(-1/(2-x**(2*k))))


		

if __name__ == "__main__":
	from sys import argv
	from time import perf_counter

	if len(argv) != 13:
		print("Usage: omega delta E_J gamma T Lambda num_threads num_samples periods max_wells D <outfile>")
		print("Units: \n -> omega in MHz \n -> delta in picoseconds\n -> E_J in GHz*hbar \n -> gamma in meV\n -> T in Kelvin\n -> Lambda in GHz")
		exit(0)

	#Command line arguments
	omega = float(argv[1])*1e-3*GHz
	delta = float(argv[2])*picosecond
	E_J = float(argv[3])*GHz*hbar
	gamma = float(argv[4])*meV
	Temp = float(argv[5])*Kelvin
	Lambda = float(argv[6])*GHz
	num_threads = int(argv[7])
	num_samples = int(argv[8])
	num_periods = int(argv[9])
	max_wells = int(argv[10])
	D = int(argv[11])
	outfile = argv[12]


	parameters = {"omega": omega, "delta":delta , "E_J":E_J, "gamma":gamma, "Temp":Temp, "Lambda":Lambda, "max_wells": max_wells}

	#Quantization parameter in stabilizer condition fixing \sqrt{L/C}
	quantization_param = 1.0

	#Determine L,C
	impedance = planck_constant/(2*quantization_param*e_charge**2)
	L = impedance/omega
	C = 1/(impedance*omega)

	#The resistance (for the "parameters" dict only, not actually used)
	#NB: we take the normalizing frequency omega_0 = 1 meV here
	omega_0 = 1*meV
	resistance  = hbar*omega_0/(2*pi*e_charge**2*gamma)

	#Record the system parameters
	parameters["L"] = L/Henry
	parameters["C"] = C/Farad
	parameters["R"] = resistance/Ohm


	#### ****                           **** ####
	####      Setting up the simulation **** ####
	#### ****                           **** ####
	phi = linspace(-(2*max_wells+1)*pi,(2*max_wells+1)*pi,num=(2*max_wells+1)*(D-1)+1)
	dphi = phi[1]-phi[0]


	Tmat = get_tmat(len(phi),dtype=complex64)
	Pi = 1.0j*(Tmat-eye(len(phi)))/dphi
	Pi_sq = (Pi @ Pi.conj().T + Pi.conj().T @ Pi)/2

	Q_sq = ((2*e_charge/hbar)**2)*Pi_sq


	H_ind = (hbar/(2*e_charge))**2*diag(phi)**2/(2*L)
	H_cap = Q_sq/(2*C)
	H_JJ = -E_J*diag(cos(phi))

	H = H_ind + H_cap + H_JJ

	#Diagonalize the Hamiltonian
	#t1 = perf_counter()
	eigvals , eigvecs = eigh(H)
	#t2 = perf_counter()
	#print("Time to diagonalize: {}".format(t2-t1))

	#Construct the jump operators
	t1 = perf_counter()
	X1 = eigvecs.conj().T @ diag(exp(1j*phi)) @ eigvecs
	t2 = perf_counter()
	print("Time to construct X1: {}".format(t2-t1))
	X2 = X1.conj().T

	d = len(phi)
	t1 = perf_counter()
	g_mat = zeros((d,d))
	for m in range(d):
		for n in range(d):
			g_mat[m,n] = g(eigvals[n]-eigvals[m],Temp,Lambda)

	L1_en = 2*pi*sqrt(gamma)*g_mat*X1
	L2_en = 2*pi*sqrt(gamma)*g_mat*X2
	t2 = perf_counter()
	print("Time to construct jump ops: {}".format(t2-t1))

	t1 = perf_counter()
	L1 = eigvecs @ L1_en @ eigvecs.conj().T
	L2 = eigvecs @ L2_en @ eigvecs.conj().T
	t2 = perf_counter()
	print("Time to change basis of jump ops: {}".format(t2-t1))


	psi = exp(-phi**2/(4*pi))
	t1 = perf_counter()
	q = L1 @ psi
	t2 = perf_counter()
	print("Time to act on a vector: {}".format(t2-t1))
	

	H_eff_NH = -0.5j*(L1.conj().T @ L1 + L2.conj().T @ L2)

	_,S,_ = svd(H+H_eff_NH)
	dt = hbar/(10*S[0])

	print("Ratio delta/dt: {}".format(delta/dt))


	for i in range(num_periods):
		_t = 0
		t1 = perf_counter()
		while _t + dt < delta:
			H_eff = H + bump_function(_t/delta)**2*H_eff_NH 
			psi = time_evolve(psi,H_eff,dt)
			_t += dt

		if _t < delta:
			H_eff = H + bump_function(_t/delta)**2*H_eff_NH 
			psi = time_evolve(psi,H_eff,delta-_t)

		t2 = perf_counter()
		print("Time for single period: {}".format(t2-t1))




