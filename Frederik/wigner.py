#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:09:03 2022

@author: frederiknathan
Get wigner function from wavefunction. some elements are copied from code I found on stackexchange I think. Don't remember if I use these elements
"""

from numpy import *
from scipy.linalg import * 
from matplotlib.pyplot import *


L = 100

def wigner(psi,x_in,pvec = None ):
    x = x_in
    L = len(x_in)
    dx = x_in[1]-x_in[0]
    x_out = linspace(x[0]-x[-1],x[-1]-x[0],2*L-1)/2
    
    if type(pvec) ==type(None):
        
        prange = x_out
        p_out = prange
    else:
        prange = pvec
        p_out = prange 
        
    W = zeros((len(p_out),2*L-1))
    
    np = 0
    for p in prange:
        v= psi * exp(1j*x_in*p)
        # v2 = psi * exp()
        W[np] = convolve(v,v.conj())*dx/pi
        np+=1 
    
    
    # W = W.T 
    return W,x_out,p_out

def wigner2(rho,x_in):
    x = x_in
    
    L = len(x_in)
    dx = x_in[1]-x_in[0]
    x_out = linspace(x[0]-x[-1],x[-1]-x[0],2*L-1)
    
    prange = x_out
    p_out = prange
    W = zeros((2*L-1,2*L-1))
    
    np = 0
    
    for p in prange:
        v = exp(1j*x_in*p).reshape((L,1))
        mat = v*rho*(v.conj().T)
        
        v = psi * exp(1j*x_in*p)
        W[np] = convolve(v,v.conj())*dx/pi
        np+=1 
    
    
    W = W.T
    return W,x_out,p_out


def wigner_clenshaw(rho, xvec, yvec, g=sqrt(2)):
    """
    Using Clenshaw summation - numerically stable and efficient
    iterative algorithm to evaluate polynomial series.
    
    The Wigner function is calculated as
    :math:`W = e^(-0.5*x^2)/pi * \sum_{L} c_L (2x)^L / sqrt(L!)` where 
    :math:`c_L = \sum_n \\rho_{n,L+n} LL_n^L` where
    :math:`LL_n^L = (-1)^n sqrt(L!n!/(L+n)!) LaguerreL[n,L,x]`
    
    """

    M = np.prod(rho.shape[0])
    X,Y = np.meshgrid(xvec, yvec)
    #A = 0.5 * g * (X + 1.0j * Y)
    A2 = g * (X + 1.0j * Y) #this is A2 = 2*A
    
    B = np.abs(A2)
    B *= B
    w0 = (2*rho.data[0,-1])*np.ones_like(A2)
    L = M-1
    #calculation of \sum_{L} c_L (2x)^L / sqrt(L!)
    #using Horner's method

    rho = rho.full() * (2*np.ones((M,M)) - np.diag(np.ones(M)))
    while L > 0:
        L -= 1
        #here c_L = _wig_laguerre_val(L, B, np.diag(rho, L))
        w0 = _wig_laguerre_val(L, B, np.diag(rho, L)) + w0 * A2 * (L+1)**-0.5
    # else:
        # while L > 0:
        #     L -= 1
        #     diag = _csr_get_diag(rho.data.data,rho.data.indices,
        #                         rho.data.indptr,L)
        #     if L != 0:
        #         diag *= 2
        #     #here c_L = _wig_laguerre_val(L, B, np.diag(rho, L))
        #     w0 = _wig_laguerre_val(L, B, diag) + w0 * A2 * (L+1)**-0.5
        
    return w0.real * np.exp(-B*0.5) * (g*g*0.5 / pi)


def _wig_laguerre_val(L, x, c):
    """
    this is evaluation of polynomial series inspired by hermval from numpy.    
    Returns polynomial series
    \sum_n b_n LL_n^L,
    where
    LL_n^L = (-1)^n sqrt(L!n!/(L+n)!) LaguerreL[n,L,x]    
    The evaluation uses Clenshaw recursion
    """

    if len(c) == 1:
        y0 = c[0]
        y1 = 0
    elif len(c) == 2:
        y0 = c[0]
        y1 = c[1]
    else:
        k = len(c)
        y0 = c[-2]
        y1 = c[-1]
        for i in range(3, len(c) + 1):
            k -= 1
            y0,    y1 = c[-i] - y1 * (float((k - 1)*(L + k - 1))/((L+k)*k))**0.5, \
            y0 - y1 * ((L + 2*k -1) - x) * ((L+k)*k)**-0.5
            
    return y0 - y1 * ((L + 1) - x) * (L + 1)**-0.5

if __name__=="__main__":
    L= 600
    x = linspace(-40,40,L)
    # psi= exp(-x**2/0.1**2)*exp(-1j*0.2*x)
    
    psi = sum(array([exp(-(x-10.2*n)**2/1) for n in range(-10,10)]),axis=0)*exp(-x**2/30**2)
    
    psi = psi/(norm(psi))
    
    W,xg,pg = wigner(psi,x)
    pcolormesh(xg,pg,W)
    plot(xg,pi/10.2+0*xg,'--w')
    ylim(-1,1)
    figure(2)
    plot(psi)
