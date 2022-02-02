# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-01-31 10:35:12
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-02-02 14:27:34

import re
from scipy.optimize import brentq
import inspect

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
