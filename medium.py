# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-02-18 15:25:03
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-03-07 11:13:37

import numpy as np
from logger import logger
from constants import *


class VolumeConductor:
    ''' Interface to volume conductor model to represent extracellular media '''

    def __init__(self, sigma):
        '''
        Initialization

        :param sigma: medium conductivity scalar or tensor (S/m)
        '''
        self.sigma = sigma
        logger.info(f'created {self}')
    
    def copy(self):
        return self.__class__(self.sigma)
    
    def __repr__(self):
        sigma_str = [f'{x:.3f}' for x in self.sigma]
        if self.is_isotropic():
            sigma_str = [sigma_str[0]]
        sigma_str = f'[{", ".join(sigma_str)}]'
        return f'{self.mtype().capitalize()}{self.__class__.__name__}(sigma={sigma_str}S/m)'
    
    def is_isotropic(self):
        ''' Check whether medium is isotropic '''
        return self.sigma[0] == self.sigma[1] == self.sigma[2]
    
    def mtype(self):
        ''' Medium type '''
        s = 'isotropic'
        if not self.is_isotropic():
            s = f'an{s}'
        return s
    
    @property
    def sigma(self):
        return self._sigma
    
    @sigma.setter
    def sigma(self, value):
        if isinstance(value, float):
            value = [value] * 3
        value = np.asarray(value)
        if not hasattr(self, '_sigma'):
            self.ref_sigma = value
        self._sigma = value
    
    def reset(self):
        self.sigma = self.ref_sigma

    def conductance(self, d):
        '''
        Compute the conductance resulting from integrating the medium's conductivity
        along the electrode-target path.
        
        :param d: vectorial x, y, z distance to target point (um)
        :return: integrated conductance (uS)
        '''
        d = np.asarray(d)
        # If d is a 2-dimensional matrix, call function for each row
        if d.ndim > 1:
            return np.array([self.conductance(dd) for dd in d])
        SSx = d[0]**2 * self.sigma[1] * self.sigma[2]
        SSy = d[1]**2 * self.sigma[0] * self.sigma[2]
        SSz = d[2]**2 * self.sigma[0] * self.sigma[1]
        return 4 * np.pi * np.sqrt(SSx + SSy + SSz)  # um S/m = 1e-6 mS/m = 1e-6 S = uS

    def phi(self, I, d):
        '''
        Compute the induced extracellular potential at a specific distance from
        a point-current source in an isotropic volume conductor model.

        :param I: current amplitude (uA)
        :param d: (x, y, z) coordinates (um)
        :return: extracellular potential (mV)
        '''
        return I / self.conductance(d) * V_TO_MV  # mV
