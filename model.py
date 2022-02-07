# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-01-31 10:35:12
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-02-07 11:32:38

import numpy as np
import re
from scipy.optimize import brentq
import inspect

from constants import MA_CM2_TO_UA_CM2


def vtrap(x, y):
    ''' Generic function used to compute rate constants. '''
    return x / (np.exp(x / y) - 1)
    

class Model:

    dstate_pattern = re.compile('^d([a-z])_dt$')
    current_pattern = re.compile('^i_([A-Z][a-z]*)$')

    def __init__(self, Cm=1.):
        '''
        Model constructor
        
        :param Cm: membrane capactance (uF/cm2)
        '''
        self.Cm = Cm
        self.der_states = {}
        self.eq_states = {}
        self.currents = {}
    
    def copy(self):
        ''' Copy model '''
        other = self.__class__(self.Cm)
        other.currents = self.currents.copy()
        other.der_states = self.der_states.copy()
        other.eq_states = self.eq_states.copy()
        return other
    
    def compute_der_states(self, Vm, x):
        ''' Compute dictionary of states derivatives from membrane potential and states dictionary '''
        return {k: v(Vm, x) for k, v in self.der_states.items()}

    def compute_eq_states(self, Vm):
        ''' Compute dictionary of equilibrium states from membrane potential '''
        return {k: v(Vm) for k, v in self.eq_states.items()}

    def compute_currents(self, Vm, x):
        ''' Compute dictionary of currents from membrane potential and states dictionary '''
        return {k: v(Vm, x) for k, v in self.currents.items()}

    def add_state(self, dfunc, eqfunc):
        ''' Add a new state to model states dictionary '''
        key = self.dstate_pattern.match(dfunc.__name__).group(1)
        params = list(inspect.signature(dfunc).parameters.keys())
        params.remove('Vm')
        self.der_states[key] = lambda Vm, x: dfunc(*[x[k] for k in params], Vm)
        self.eq_states[key] = eqfunc

    def add_current(self, cfunc):
        ''' Add a new current to model currents dictionary '''
        key = self.current_pattern.match(cfunc.__name__).group(1)
        params = list(inspect.signature(cfunc).parameters.keys())
        params.remove('Vm')
        self.currents[f'i_{key}'] = lambda Vm, x: cfunc(*[x[k] for k in params], Vm)

    def update_current(self, cfunc):
        key = self.current_pattern.match(cfunc.__name__).group(1)
        if f'i_{key}' not in self.currents.keys():
            raise ValueError(f'{key} current not found')
        self.add_current(cfunc)
    
    @property
    def states_names(self):
        ''' Names of states '''
        return list(self.eq_states.keys())

    def i_membrane(self, Vm, states):
        ''' net membrane current

            :param Vm: membrane potential (mV)
            :param states: states of ion channels gating and related variables
            :return: current per unit area (uA/cm2)
        '''
        return sum(self.compute_currents(Vm, states).values())

    def derivatives(self, t, y, stim=None):
        ''' Compute system derivatives.

            :param t: specific instant in time (ms)
            :param y: vector of HH system variables at time t
            :param stim (optional): stimulus object
            :return: vector of system derivatives at time t
        '''
        Vm, *states = y
        states_dict = dict(zip(self.states_names, states))
        dstates_dt = self.compute_der_states(Vm, states_dict)
        dQm_dt = -self.i_membrane(Vm, states_dict)  # uA/cm2
        dVm_dt = dQm_dt / self.Cm  # mV/ms
        if stim is not None:
            dVm_dt += stim.compute(t)
        return [dVm_dt, *dstates_dt.values()]

    def equilibrium(self):
        ''' Compute model equilibrium state '''
        if not self.currents:
            Vm0 = 0.
        else:
            Vm0 = brentq(  # steady-state membrane potential (mV)
                lambda v: self.i_membrane(v, self.compute_eq_states(v)), -100., 50.)
        return {'Vm': Vm0, **self.compute_eq_states(Vm0)}


class PyramidalNeuron(Model):

    Cm0 = 1.0        # Membrane capacitance (uF/cm2)
    ELeak = -70.3    # Leakage reversal potential (mV)
    ENa = 50.0       # Sodium reversal potential (mV)
    EK = -90.0       # Potassium reversal potential (mV)
    V_T = -56.2      # Spike threshold adjustment parameter (mV)
    gLeak = 2.05e-5  # Leakage maximal channel conductance (S/cm2)
    gNa_bar = 0.056  # Sodium maximal channel conductance (S/cm2)
    gK_bar = 0.006   # Potassium maximal channel conductance (S/cm2)

    def __init__(self):
        super().__init__(Cm=self.Cm0)
        self.add_current(self.i_Leak)
        self.add_state(self.dm_dt, self.m_inf)
        self.add_state(self.dh_dt, self.h_inf)
        self.add_current(self.i_Na)
        self.add_state(self.dn_dt, self.n_inf)
        self.add_current(self.i_K)

    #--------------------- Leakage current ---------------------

    def i_Leak(self, Vm):
        return self.gLeak * (Vm - self.ELeak) * MA_CM2_TO_UA_CM2  # uA/cm2
    
    #--------------------- Sodium current ---------------------

    def alpha_m(self, Vm):
        return 0.32 * vtrap(13 - (Vm - self.V_T), 4)  # ms-1

    def beta_m(self, Vm):
        return 0.28 * vtrap((Vm - self.V_T) - 40, 5)  # ms-1

    def dm_dt(self, m, Vm):
        return self.alpha_m(Vm) * (1 - m) - self.beta_m(Vm) * m  # ms-1

    def m_inf(self, Vm):
        return self.alpha_m(Vm) / (self.alpha_m(Vm) + self.beta_m(Vm))  # (-)

    def alpha_h(self, Vm):
        return 0.128 * np.exp(-((Vm - self.V_T) - 17) / 18)  # ms-1

    def beta_h(self, Vm):
        return 4 / (1 + np.exp(-((Vm - self.V_T) - 40) / 5))  # ms-1

    def dh_dt(self, h, Vm):
        return self.alpha_h(Vm) * (1 - h) - self.beta_h(Vm) * h

    def h_inf(self, Vm):
        return self.alpha_h(Vm) / (self.alpha_h(Vm) + self.beta_h(Vm))

    def i_Na(self, m, h, Vm):
        return self.gNa_bar * m**3 * h * (Vm - self.ENa) * MA_CM2_TO_UA_CM2  # uA/cm2

    #--------------------- Potassium current ---------------------

    def alpha_n(self, Vm):
        return 0.032 * vtrap(15 - (Vm - self.V_T), 5)  # ms-1

    def beta_n(self, Vm):
        return 0.5 * np.exp(-((Vm - self.V_T) - 10) / 40)  # ms-1

    def dn_dt(self, n, Vm):
        return self.alpha_n(Vm) * (1 - n) - self.beta_n(Vm) * n  # ms-1

    def n_inf(self, Vm):
        return self.alpha_n(Vm) / (self.alpha_n(Vm) + self.beta_n(Vm))  # (-)

    def i_K(self, n, Vm):
        return self.gK_bar * n**4 * (Vm - self.EK)  * MA_CM2_TO_UA_CM2  # uA/cm2


