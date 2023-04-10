from numpy import array,linspace,conj,zeros,exp,pi



def wigner_pure(psi,q_vals,p_vals=None,hbar=1):
	"""
	Compute the Wigner function for the pure state represented by psi.

	Parameters
    ----------
    psi: array
    	The wavefunction for this pure state (assumed to be in the "q" basis). See "q_vals" below 
    		for expected format
    q_vals: array
    	The q values we are sampling. The q values are expected to range between q_min and q_max 
    		(q_min < 0, q_max > 0) in uniform steps of dq with the following format:
    		q_vals[0:q_max/dq]: q = 0, dq, 2dq, ... , q_max
    		q_vals[q_max/dq+1:]: q = q_min, q_min+dq, ... , -dq

    	In other words, q_vals[i] = i*dq, with the formatting above allowing i to be negative
    p_vals: array, optional
    	The conjugate momenta values we are sampling. the expected format is the same as q above.
    	Defaults to none, in which case the same values as q_vals are used
    hbar: float, optional
    	Value of the reduced Planck's constant. Defaults to 1.

    Returns
    -------
    W : ndarray
        The wigner function as a 2d array. First index is conjugate momentum p, second is position q
    q: array
    	Array of q values, with the same format as q_vals above
    p: array
    	Array of p values, with the same format as p_vals above
	"""
	if p_vals is None:
		p_vals = array(q_vals)

	#Find q_min and q_max (see above)
	q_min , q_max = 0.0,0.0 
	for q in q_vals:
		if q < 0.0:
			q_min = q
			break
		else:
			q_max = q

	dq = abs(q_vals[1])	#Spacing dq (abs() accounts for the edge case q_max = 0)
	N_q = len(q_vals)


	W = zeros((len(p_vals),len(q_vals)))
	for i,p in enumerate(p_vals):
		for j,q in enumerate(q_vals):
			_sum = 0.0

			#Adjust the index of q to be negative if q is negative
			if q < 0:
				q_ind = j - N_q
			else:
				q_ind = j

			#Integrate over y
			for k,y in enumerate(q_vals):
				if (q+y) > q_max:
					break
				elif (q-y) < q_min:
					continue
				else:
					#Adjust the index of y to be negative if q is negative
					if y < 0:
						y_ind = k - N_q
					else:
						y_ind = k
					_sum += exp(2j*p*y/hbar)*conj(psi[q_ind+y_ind])*psi[q_ind-y_ind]

			W[i,j] = dq*_sum/(pi*hbar)

	return W, q_vals , p_vals

					
