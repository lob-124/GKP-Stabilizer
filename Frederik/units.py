#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 14:31:50 2020

@author: frederik

"""
from numpy import *

# =============================================================================
# Units
# =============================================================================
meV = 1
eV  = 1000*meV
Å  = 1 
e_charge = 1 
hbar = 1 
planck_constant = 2*pi*hbar 
THz   = 0.6242*meV/hbar #NB! 1 THz in the simulation corresponds to 

flux_quantum = planck_constant/(2*e_charge)
nm     = 10
meter = 1e9*nm
centimeter = 1e-2 *meter
micrometer = 1e-6 *meter
millimeter = 1e-3 *meter
picosecond = 1/THz
second = 1e12*picosecond
GHz = 1e-3*THz 

#h = # 1/0.24180 meV/THz = 

Coulomb = e_charge/1.602e-19
Ampere = Coulomb/second 
Volt  = 1e3 * meV/e_charge
Ohm   = Volt/Ampere
Joule = 1*Volt*Coulomb
# THz_physical = 0.6242*meV/hbar

Henry = Joule/(Ampere**2)
Farad = Coulomb/Volt

# picosecond_physical = 1/THz_physical
# second_physical=10**12*picosecond_physical

Kelvin = 0.0862 *meV


SX = array([[0,1],[1,0]])
SY = array([[0,-1j],[1j,0]])
SZ = array([[1,0],[0,-1]])
I2 = eye(2)