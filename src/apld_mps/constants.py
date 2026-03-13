"""Fundamental physical constants used throughout the APLD-MPS suite."""

import math

# Planck constant [J·s]
HBAR = 1.054571817e-34

# Speed of light in vacuum [m/s]
C_LIGHT = 2.99792458e8

# Boltzmann constant [J/K]
K_BOLTZMANN = 1.380649e-23

# Elementary charge [C]
E_CHARGE = 1.602176634e-19

# Electron mass [kg]
M_ELECTRON = 9.1093837015e-31

# Vacuum permittivity [F/m]
EPSILON_0 = 8.8541878128e-12

# Vacuum permeability [H/m]
MU_0 = 4.0 * math.pi * 1e-7

# Room temperature [K]
T_ROOM = 300.0

# Conversion: eV to Joules
EV_TO_J = E_CHARGE

# Conversion: Joules to eV
J_TO_EV = 1.0 / E_CHARGE

# Conversion: nm to m
NM_TO_M = 1e-9

# Conversion: fs to s
FS_TO_S = 1e-15

# Conversion: ps to s
PS_TO_S = 1e-12
