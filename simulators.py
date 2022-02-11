# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-01-31 10:46:44
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-02-11 14:26:44

import numpy as np
from scipy.integrate import odeint, solve_ivp
from tqdm import tqdm

from logger import *
from constants import *
from solution import Solution


class Simulator:
    ''' Generic interface to ODE simulator object. '''

    def __init__(self, ykeys, dfunc, dt=None):
        ''' Initialization.

            :param ykeys: list of differential variables names
            :param dfunc: derivative function
            :param dt: integration time step (ms)
        '''
        self.ykeys = ykeys
        self.dfunc = dfunc
        self.dt = dt

    def get_n_samples(self, t0, tend, dt=None):
        ''' Get the number of samples required to integrate across 2 times with a given time step.

            :param t0: initial time (ms)
            :param tend: final time (ms)
            :param dt: integration time step (ms)
            :return: number of required samples, rounded to nearest integer
        '''
        if dt is None:
            dt = self.dt
        return max(int(np.round((tend - t0) / dt)), 2)

    def get_time_vector(self, t0, tend, **kwargs):
        ''' Get the time vector required to integrate from an initial to a final time with
            a specific time step.

            :param t0: initial time (ms)
            :param tend: final time (ms)
            :return: vector going from current time to target time with appropriate step (ms)
        '''
        return np.linspace(t0, tend, self.get_n_samples(t0, tend, **kwargs))

    def initialize(self, y0, t0=0.):
        ''' Initialize global time vector, state vector and solution array.

            :param y0: dictionary of initial conditions
            :param t0: optional initial time or time vector (ms)
        '''
        keys = list(y0.keys())
        if len(keys) != len(self.ykeys):
            raise ValueError("Initial conditions do not match system's dimensions")
        for k in keys:
            if k not in self.ykeys:
                raise ValueError(f'{k} is not a differential variable')
        y0 = {k: np.array([v]) for k, v in y0.items()}
        ref_size = y0[keys[0]].size
        if not all(v.size == ref_size for v in y0.values()):
            raise ValueError('dimensions of initial conditions are inconsistent')
        self.y = np.array(list(y0.values())).T
        self.t = np.ones(self.y.shape[0]) * t0

    def append(self, t, y):
        ''' Append to global time vector, state vector and solution array.

            :param t: new time vector to append (ms)
            :param y: new solution matrix to append
        '''
        self.t = np.concatenate((self.t, t))
        self.y = np.concatenate((self.y, y), axis=0)

    @staticmethod
    def time_str(t):
        return f'{t:.5f} ms'

    def timed_log(self, s, t=None):
        ''' Add preceding time information to log string. '''
        if t is None:
            t = self.t[-1]
        return f't = {self.time_str(t)}: {s}'

    def integrate_until(self, target_t, remove_first=False):
        ''' Integrate system until a target time and append new arrays to global arrays.

            :param target_t: target time (ms)
            :param remove_first: optional boolean specifying whether to remove the first index
            of the new arrays before appending
        '''
        if target_t < self.t[-1]:
            raise ValueError(f'target time ({target_t} ms) precedes current time {self.t[-1]} ms')
        elif target_t == self.t[-1]:
            t, y = self.t[-1], self.y[-1]
        if self.dt is None:
            sol = solve_ivp(
                self.dfunc, [self.t[-1], target_t], self.y[-1], method='LSODA')
            t, y = sol.t, sol.y.T
        else:
            t = self.get_time_vector(self.t[-1], target_t)
            y = odeint(self.dfunc, self.y[-1], t, tfirst=True)
        if remove_first:
            t, y = t[1:], y[1:]
        self.append(t, y)

    def solve(self, y0, tstop, **kwargs):
        ''' Simulate system for a given time interval for specific initial conditions.

            :param y0: dictionary of initial conditions
            :param tstop: stopping time (ms)
        '''
        # Initialize system
        self.initialize(y0, **kwargs)

        # Integrate until tstop
        self.integrate_until(tstop, remove_first=True)

    @property
    def solution(self):
        ''' Return solution as a pandas dataframe.

            :return: timeseries dataframe with labeled time and variables vectors.
        '''
        sol = Solution({
            TIME_MS: self.t,
            **{k: self.y[:, i] for i, k in enumerate(self.ykeys)}
        })
        sol.rename(columns={'Vm': V_MV}, inplace=True)
        return sol

    def __call__(self, *args, max_nsamples=None, **kwargs):
        ''' Specific call method: solve the system and return solution dataframe. '''
        self.solve(*args, **kwargs)
        return self.solution


class StimulusDrivenSimulator(Simulator):
    ''' Stimulus-driven ODE simulator. '''

    def __init__(self, *args, stim=None, **kwargs):
        ''' Initialization.

            :param stim: stimulus object
        '''
        super().__init__(*args, **kwargs)
        self.stim = stim
        self.stim.update(0.)

    def fire_event(self, xevent):
        ''' Call event function and set new xref value. '''
        if xevent is not None:
            if xevent == 'log':
                self.log_progress()
            else:
                self.stim.update(xevent)

    def init_log(self, logfunc, n):
        ''' Initialize progress logger. '''
        self.logfunc = logfunc
        if self.logfunc is None:
            set_handler(logger, TqdmHandler(my_log_formatter))
            self.pbar = tqdm(total=n)
        else:
            self.np = n
            logger.debug('integrating stimulus')

    def log_progress(self):
        ''' Log simulation progress. '''
        if self.logfunc is None:
            self.pbar.update()
        else:
            logger.debug(self.timed_log(self.logfunc(self.y[-1])))

    def terminate_log(self):
        ''' Terminate progress logger. '''
        if self.logfunc is None:
            self.pbar.close()
        else:
            logger.debug('integration completed')

    def solve(self, y0, tstop, log_period=None, logfunc=None, **kwargs):
        ''' Simulate system for a specific stimulus application pattern.

            :param y0: 1D vector of initial conditions
            :param tstop: stopping time (ms)
        '''
        # Get events
        events = self.stim.stim_events()

        # Make sure all events occur before tstop
        if events[-1][0] > tstop:
            raise ValueError('all events must occur before stopping time')

        if log_period is not None:  # Add log events if any
            tlogs = np.arange(kwargs.get('t0', 0.), tstop, log_period)[1:]
            if tstop not in tlogs:
                tlogs = np.hstack((tlogs, [tstop]))
            events = self.sort_events(events + [(t, 'log') for t in tlogs])
            self.init_log(logfunc, tlogs.size)
        else:  # Otherwise, add None event at tstop
            events.append((tstop, None))

        # Initialize system
        self.initialize(y0, **kwargs)

        # For each upcoming event
        for i, (tevent, xevent) in enumerate(events):
            self.integrate_until(  # integrate until event time
                tevent,
                remove_first=i > 0 and events[i - 1][1] == 'log')
            self.fire_event(xevent)  # fire event

        # Terminate log if any
        if log_period is not None:
            self.terminate_log()
