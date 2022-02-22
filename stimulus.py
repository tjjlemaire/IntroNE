# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-01-31 12:22:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-02-22 11:48:25

import numpy as np
from logger import logger


class PulseTrain:
    ''' Pulse train object '''

    def __init__(self, tpulse=1., npulses=1, PRF=.1, tstart=5.):
        ''' Constructor.

            :param tpulse: pulse duration (ms)
            :param npulses: number of pulses (default = 1)
            :param PRF (default = 0.1 kHz): pulse repetition frequency (kHz)
            :param tstart: pulse start time (defaut = 0 ms)
        '''
        self.tpulse = tpulse
        self.npulses = npulses
        self.PRF = PRF
        self.tstart = tstart

    def inputs(self):
        l = [
            f'tpulse={self.tpulse:.1f}ms',
            f'npulses={self.npulses}',
            f'PRF={self.PRF:.2f}kHz'
        ]
        if self.tstart > 0:
            l.append(f'tstart={self.tstart:.1f}ms')
        return l

    def __repr__(self):
        return f'{self.__class__.__name__}({", ".join(self.inputs())})'

    @property
    def PRF(self):
        return self._PRF

    @PRF.setter
    def PRF(self, value):
        if self.npulses > 1 and value > 1 / self.tpulse:
            raise ValueError('pulse repetition interval must be longer than pulse duration')
        self._PRF = value

    @property
    def T_ON(self):
        ''' Proxy for pulse duration (ms) '''
        return self.tpulse

    @property
    def T_OFF(self):
        ''' OFF interval between pulses (ms) '''
        return 1 / self.PRF - self.tpulse

    def t_OFF_ON(self):
        ''' Compute vector of times of OFF-ON transitions (in ms). '''
        return np.arange(self.npulses) / self.PRF + self.tstart

    def t_ON_OFF(self):
        ''' Compute vector of times of ON-OFF transitions (in ms). '''
        return self.t_OFF_ON() + self.tpulse


class CurrentPulseTrain(PulseTrain):
    ''' Current pulse train object '''

    def __init__(self, I=1., unit='uA/cm2', **kwargs):
        ''' Constructor.

            :param I: current
        '''
        self.I = I
        self.unit = unit
        super().__init__(**kwargs)

    def inputs(self):
        return [f'I={self.I:.2f}{self.unit}'] + super().inputs()   

    def stim_events(self):
        ''' Compute (time, value) pairs for each modulation event '''
        t_off_on, t_on_off = self.t_OFF_ON(), self.t_ON_OFF()
        pairs_on = list(zip(t_off_on, [self.I] * len(t_off_on)))
        pairs_off = list(zip(t_on_off, [0.] * len(t_on_off)))
        return sorted(pairs_on + pairs_off, key=lambda x: x[0])
    
    def compute(self, t):
        ''' Compute the current at time t '''
        return self.I_t

    def update(self, x):
        ''' Update the current modulation factor. '''
        self.I_t = x

    def stim_profile(self, tstop=None):
        ''' Return stimulus profile a list of transitions '''
        events = self.stim_events()
        l = [(0., 0)]
        for e in events:
            l.append((e[0], l[-1][1]))
            l.append(e)
        t, x = zip(*l)
        t, x = np.array(t), np.array(x)
        if tstop is not None:
            t = np.hstack((t, [tstop]))
            x = np.hstack((x, [x[-1]]))
        return t, x


class ExtracellularCurrentPulseTrain(CurrentPulseTrain):
    ''' Extracellular current pulse train object '''

    def __init__(self, pos=(0., 0., 100.), unit='uA', **kwargs):
        ''' Constructor.

            :param pos: electrode (x, y, z) position (um) 
        '''
        self.pos = np.asarray(pos)
        super().__init__(unit=unit, **kwargs)
        logger.info(f'created {self}')


