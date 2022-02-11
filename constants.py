# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-01-31 10:39:57
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-02-11 14:38:16

# Model conversion constants

MA_CM2_TO_UA_CM2 = 1e3
UM2_TO_CM2 = 1e-8
MV_TO_UV = 1e3
UA_TO_NA = 1e3

# Simulations
DT = 0.05  # simulation time step (ms)

# Labels
TIME_S = 'time (s)'
TIME_MS = 'time (ms)'
V_MV = 'Vm (mV)'
PHI_UV = 'phi (uV)'
CURRENT_DENSITY = 'i (uA/cm2)'
I_NA = 'I (nA)'

# Plot parameters
V_LIMS = (-80., 50.)  # mV
STATES_LIMS = (-0.1, 1.1)  # (-)
I_LIMS = (-5., 5.)  # uA/cm2