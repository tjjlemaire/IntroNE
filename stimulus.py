# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-01-31 12:22:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-04-14 10:35:29

import numpy as np
from logger import logger
import matplotlib.pyplot as plt

from constants import *


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
    
    @property
    def tpulse(self):
        return self._tpulse
    
    @tpulse.setter
    def tpulse(self, value):
        if not hasattr(self, '_tpulse'):
            self.ref_tpulse = value
        self._tpulse = value

    @property
    def npulses(self):
        return self._npulses
    
    @npulses.setter
    def npulses(self, value):
        if not hasattr(self, '_npulses'):
            self.ref_npulses = value
        self._npulses = value

    @property
    def PRF(self):
        return self._PRF
    
    @PRF.setter
    def PRF(self, value):
        if self.npulses > 1 and value > 1 / self.tpulse:
            raise ValueError('pulse repetition interval must be longer than pulse duration')
        if not hasattr(self, '_PRF'):
            self.ref_PRF = value
        self._PRF = value

    @property
    def tstart(self):
        return self._tstart
    
    @tstart.setter
    def tstart(self, value):
        if not hasattr(self, '_tstart'):
            self.ref_tstart = value
        self._tstart = value
    
    def reset(self):
        self.tpulse = self.ref_tpulse 
        self.npulses = self.ref_npulses
        self.PRF = self.ref_PRF
        self.tstart = self.ref_tstart

    def copy(self):
        return self.__class__(
            tpulse=self.tpulse, 
            npulses=self.npulses, 
            PRF=self.PRF, 
            tstart=self.tstart
        )

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

    def plot(self, tstop=None):
        ''' Plot stimulus temporal profile '''
        fig, ax = plt.subplots(figsize=(8, 4))
        for sk in ['top', 'right']:
            ax.spines[sk].set_visible(False)
        ax.set_xlabel(TIME_MS)
        ax.set_ylabel(self.unit)
        ax.set_title(self)
        t, I = self.stim_profile()
        if tstop is None:
            tstop = 1.1 * t.max()
        else:
            if tstop < t.max():
                raise ValueError(f'stopping time ({tstop:.1f} ms) precedes last stimulus modulation event ({t.max():.1f} ms)')
        t = np.append(t, [tstop])
        I = np.append(I, [I[-1]])
        ax.plot(t, I)
        return fig


class CurrentPulseTrain(PulseTrain):
    ''' Current pulse train object '''

    def __init__(self, I=1., unit='uA/cm2', **kwargs):
        ''' Constructor.

            :param I: current
        '''
        self.I = I
        self.unit = unit
        super().__init__(**kwargs)

    @property
    def I(self):
        return self._I
    
    @I.setter
    def I(self, value):
        if not hasattr(self, '_I'):
            self.ref_I = value
        self._I = value

    @property
    def unit(self):
        return self._unit
    
    @unit.setter
    def unit(self, value):
        if not hasattr(self, '_unit'):
            self.ref_unit = value
        self._unit = value
    
    def reset(self):
        super().reset()
        self.I = self.ref_I
        self.unit = self.ref_unit
    
    def copy(self):
        return self.__class__(
            I=self.I,
            unit=self.unit,
            tpulse=self.tpulse, 
            npulses=self.npulses, 
            PRF=self.PRF, 
            tstart=self.tstart, 
        )

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


class ExtracellularCurrentPulseTrain(CurrentPulseTrain):
    ''' Extracellular current pulse train object '''

    def __init__(self, pos=(0., 0., 100.), unit='uA', **kwargs):
        ''' Constructor.

            :param pos: electrode (x, y, z) position (um) 
        '''
        self.pos = pos
        super().__init__(unit=unit, **kwargs)
        logger.info(f'created {self}')

    @property
    def pos(self):
        return self._pos
    
    @pos.setter
    def pos(self, value):
        value = np.asarray(value)
        if not hasattr(self, '_pos'):
            self.ref_pos = value
        self._pos = value
    
    def inputs(self):
        s = ', '.join([f'{x:.0f}' for x in self.pos])
        return [f'pos=[{s}]um'] + super().inputs()
    
    def reset(self):
        super().reset()
        self.pos = self.ref_pos

    def copy(self):
        return self.__class__(
            pos=self.pos, 
            I=self.I,
            unit=self.unit,
            tpulse=self.tpulse, 
            npulses=self.npulses, 
            PRF=self.PRF, 
            tstart=self.tstart, 
        )


class LightPulseTrain(PulseTrain):
    ''' Light pulse train object '''

    def __init__(self, λ, I=1., unit='mW/mm2', **kwargs):
        ''' Constructor.

            :param λ: light wavelength (nm)
            :param I: light intensity (mW/mm)
        '''
        self.λ = λ
        self.I = I
        self.I_t = 0.
        self.unit = unit
        super().__init__(**kwargs)

    @property
    def I(self):
        return self._I
    
    @I.setter
    def I(self, value):
        if not hasattr(self, '_λ'):
            self.ref_I = value
        self._I = value

    @property
    def λ(self):
        return self._λ
    
    @λ.setter
    def λ(self, value):
        if not hasattr(self, '_λ'):
            self.ref_λ = value
        self._λ = value
    
    @property
    def λ_t(self):
        return self.λ

    @property
    def unit(self):
        return self._unit
    
    @unit.setter
    def unit(self, value):
        if not hasattr(self, '_unit'):
            self.ref_unit = value
        self._unit = value
    
    def reset(self):
        super().reset()
        self.λ = self.ref_λ
        self.I = self.ref_I
        self.unit = self.ref_unit
    
    def copy(self):
        return self.__class__(
            λ=self.λ,
            I=self.I,
            unit=self.unit,
            tpulse=self.tpulse, 
            npulses=self.npulses, 
            PRF=self.PRF, 
            tstart=self.tstart, 
        )

    def inputs(self):
        return [f'λ={self.λ:.0f}nm', f'I={self.I:.2f}{self.unit}'] + super().inputs()   

    def stim_events(self):
        ''' Compute (time, value) pairs for each modulation event '''
        t_off_on, t_on_off = self.t_OFF_ON(), self.t_ON_OFF()
        pairs_on = list(zip(t_off_on, [self.I] * len(t_off_on)))
        pairs_off = list(zip(t_on_off, [0.] * len(t_on_off)))
        return sorted(pairs_on + pairs_off, key=lambda x: x[0])
    
    def compute(self, t):
        ''' Compute the light wavelength and intensity at time t '''
        return (self.λ_t, self.I_t)

    def update(self, x):
        ''' Update the current modulation factor. '''
        self.I_t = x
