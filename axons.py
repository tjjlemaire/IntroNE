# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-05 14:08:31
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-03-07 10:15:50

from neuron import h
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from constants import *
from logger import logger


def axial_section_area(d_out, d_in=0.):
    ''' Compute the cross-section area of a axial cylinder section expanding between an
        inner diameter (presumably zero) and an outer diameter.
        :param d_out: outer diameter (um)
        :param d_in: inner diameter (um)
        :return: cross-sectional area (um2)
    '''
    return np.pi * ((d_out)**2 - d_in**2) / 4.

def axial_resistance_per_unit_length(rho, *args, **kwargs):
    ''' Compute the axial resistance per unit length of a cylindrical section.
        :param rho: axial resistivity (Ohm.cm)
        :return: resistance per unit length (Ohm/cm)
    '''
    return rho / axial_section_area(*args, **kwargs) / UM2_TO_CM2  # Ohm/cm

def axial_resistance(rho, L, *args, **kwargs):
    ''' Compute the axial resistance of a cylindrical section.
        :param rho: axial resistivity (Ohm.cm)
        :param L: cylinder length (um)
        :return: resistance (Ohm)
    '''
    return axial_resistance_per_unit_length(rho, *args, **kwargs) * L * UM_TO_CM  # Ohm

def periaxonal_resistance_per_unit_length(rho, d, w):
    ''' Compute the periaxonal axial resistance per unit length of a cylindrical section.
        :param rho: periaxonal resistivity (Ohm.cm)
        :param d: section inner diameter (um)
        :param w: periaxonal space width (um)
        :return: resistance per unit length (Ohm/cm)
    '''
    return axial_resistance_per_unit_length(rho, d + 2 * w, d_in=d)


class MyelinatedAxon:
    '''
    Generic double-cable, myelinated axon model based on McIntyre 2002, extended
    to allow use with any axon diameter.

    Reference:
    *McIntyre, C.C., Richardson, A.G., and Grill, W.M. (2002). Modeling the
    excitability of mammalian nerve fibers: influence of afterpotentials on
    the recovery cycle. J. Neurophysiol. 87, 995–1006.*
    '''
    # Constant model parameters
    rhoa = 70.0                    # axoplasm resistivity (Ohm.cm)
    nodeL = 1.                     # node length (um)
    mysaL = 3.                     # MYSA length (um)
    mysa_space = 2e-3              # MYSA periaxonal space width (um)
    flut_space = 4e-3              # FLUT periaxonal space width (um)
    stin_space = 4e-3              # STIN periaxonal space width (um)
    cm = 2.                        # axolamellar membrane capacitance (uF/cm2)
    g_mysa = 0.001                 # MYSA axolammelar conductance (S/cm2)
    g_flut = 0.0001                # FLUT axolammelar conductance (S/cm2)
    g_stin = 0.0001                # STIN axolammelar conductance (S/cm2)
    mycm_per_lamella = 0.1         # myelin capacitance per lamella (uF/cm2)
    mygm_per_lamella = 0.001       # myelin transverse conductance per lamella (S/cm2)
    nstin_per_inter = 6            # number of STIN sections per internode
    vrest = -80.                   # resting membrane potential (mV)

    # Lookup table for diameter-dependent model parameters
    axonD_ref = np.array([5.7, 7.3, 8.7, 10.0, 11.5, 12.8, 14.0, 15.0, 16.0])  # um
    axonD_deps = {
        'nodeD': np.array([1.9, 2.4, 2.8, 3.3, 3.7, 4.2, 4.7, 5.0, 5.5]),  # um
        'interD': np.array([3.4, 4.6, 5.8, 6.9, 8.1, 9.2, 10.4, 11.5, 12.7]),  # um
        'interL': np.array([500., 750., 1000., 1150., 1250., 1350., 1400., 1450., 1500.]),  # um
        'flutL': np.array([35., 38., 40., 46., 50., 54., 56., 58., 60.]),  # um
        'nlayers': np.array([80, 100, 110, 120, 130, 135, 140, 145, 150])  # (-)
    }

    def __init__(self, diameter=5., nnodes=101, pos=(0., 0., 0.)):
        '''
        Model initialization.

        :param diam: axon outer diameter (um)
        :param nnodes: number of nodes (default: 101)
        :param pos: (x, z, z) position of the axon central node (um)
        '''
        self.pos = np.asarray(pos)
        self.diameter = diameter
        self.nnodes = nnodes
        logger.info(f'created {self} model')
    
    def __repr__(self):
        s = ', '.join([f'{x:.0f}' for x in self.pos])    
        return f'{self.__class__.__name__}(pos=[{s}]um, d={self.diameter:.1f}um, {self.nnodes} nodes, L={self.length * 1e-3:.1f}mm)'

    def construct(self):
        ''' Construct model '''
        self.init_parameters()
        self.create_sections()
        self.build_topology()
        self.define_biophysics()
    
    def reinit(self):
        ''' Reinitialize model ''' 
        self.delete_sections()
        self.construct()
        logger.info(f're-initialized {self} model')
    
    def copy(self):
        return self.__class__(
            diameter=self.diameter,
            nnodes=self.nnodes,
            pos=self.pos
        )
    
    @property
    def diameter(self):
        return self._diameter

    @diameter.setter
    def diameter(self, value):
        if not hasattr(self, '_diameter'):
            self.ref_diameter = value
        self._diameter = value
        if hasattr(self, '_nnodes'):
            self.construct()
    
    @property
    def nnodes(self):
        return self._nnodes

    @nnodes.setter
    def nnodes(self, value):
        if not hasattr(self, '_nnodes'):
            self.ref_nnodes = value
        self._nnodes = value
        if hasattr(self, '_diameter'):
            self.construct()
    
    @property
    def pos(self):
        return self._pos
    
    @pos.setter
    def pos(self, value):
        value = np.asarray(value)
        if not hasattr(self, '_pos'):
            self.ref_pos = value
        self._pos = value
    
    def reset(self):
        self.diameter = self.ref_diameter
        self.nnodes = self.ref_nnodes
        self.pos = self.ref_pos
    
    def init_parameters(self):
        ''' Initialize model parameters. '''
        logger.debug('initializting model parameters...')
        # Interpolate diameter-dependent parameters at current axon diameter
        for k, v in self.axonD_deps.items():
            setattr(self, k, interp1d(self.axonD_ref, v, kind='linear', assume_sorted=True, fill_value='extrapolate')(self.diameter))
        
        # Topological & geometrical parameters 
        self.ninters = self.nnodes - 1  # Number of internodes
        self.nMYSA = 2 * self.ninters  # Number of paranodal myelin attachment (MYSA) sections
        self.nFLUT = 2 * self.ninters  # Number of paranodal main (FLUT) sections
        self.nSTIN = self.nstin_per_inter * self.ninters  # Number of internodal (STIN) sections        
        self.mysaD = self.nodeD  # Diameter of paranodal myelin attachment (MYSA) sections (um)
        self.flutD = self.interD  # Diameter of paranodal main (FLUT) sections (um)
        self.stinD = self.interD  # Diameter of internodal (STIN) sections (um)
        self.stinL = (self.interL - (self.nodeL + 2 * (self.mysaL + self.flutL))) / self.nstin_per_inter  # Length of internodal sections (um)
        self.node2node = self.nodeL + 2 * (self.mysaL + self.flutL) + self.nstin_per_inter * self.stinL  # node-to-node distance (um)
        self.length = (self.nnodes - 1) * self.node2node + self.nodeL  # axon length (um)

        # Intracellular resistances 
        self.R_node = axial_resistance(self.rhoa, self.nodeL, self.nodeD)  # Node intracellular axial resistance (Ohm)
        self.R_mysa = axial_resistance(
            self.rhoa, self.mysaL, self.mysaD)  # MYSA intracellular axial resistance (Ohm)
        self.R_flut = axial_resistance(
            self.rhoa, self.flutL, self.flutD)  # FLUT intracellular axial resistance (Ohm)
        self.R_stin = axial_resistance(
            self.rhoa, self.stinL, self.stinD)  # STIN intracellular axial resistance (Ohm)
        self.R_node_to_node = self.R_node + 2 * (self.R_mysa + self.R_flut) + self.nstin_per_inter * self.R_stin  #  Node-to-node intracellular axial resistance (Ohm)

        # Periaxonal resistances
        self.Rp_node = periaxonal_resistance_per_unit_length(
            self.rhoa, self.nodeD, self.mysa_space)  # Node periaxonal axial resistance per unit length (Ohm/cm)
        self.Rp_mysa = periaxonal_resistance_per_unit_length(
            self.rhoa, self.mysaD, self.mysa_space)  # MYSA periaxonal axial resistance per unit length (Ohm/cm)
        self.Rp_flut = periaxonal_resistance_per_unit_length(
            self.rhoa, self.flutD, self.flut_space)  # FLUT periaxonal axial resistance per unit length (Ohm/cm)
        self.Rp_stin = periaxonal_resistance_per_unit_length(
            self.rhoa, self.stinD, self.stin_space)  # STIN periaxonal axial resistance per unit length (Ohm/cm)
        
        # Transverse myelin parameters (computed from their nominal value per lamella membrane).
        # The underlying assumption is that each lamella membrane (2 per myelin layer) is
        # represented by an individual RC circuit with a capacitor and passive resitor, and
        # that these components are connected independently in series to form a global RC circuit.
        self.mycm = self.mycm_per_lamella / (2 * self.nlayers)  # myelin capacitance per unit area (uF/cm2)
        self.mygm = self.mygm_per_lamella / (2 * self.nlayers)  # myelin passive conductance per unit area (S/cm2)

    def create_sections(self):
        ''' Create the sections of the cell. '''
        logger.debug('creating model sections...')
        self.node = [h.Section(name=f'node{x}', cell=self) for x in range(self.nnodes)]
        self.mysa = [h.Section(name=f'mysa{x}', cell=self) for x in range(self.nMYSA)]
        self.flut = [h.Section(name=f'flut{x}', cell=self) for x in range(self.nFLUT)]
        self.stin = [h.Section(name=f'stin{x}', cell=self) for x in range(self.nSTIN)]
        self.sections = self.node + self.mysa + self.flut + self.stin

        # Create vectors of sections axial coordinates centered at zero
        self.xnodes = np.arange(self.nnodes) * self.node2node - self.length / 2  # um
        node2mysa = 0.5 * (self.nodeL + self.mysaL)  # um
        self.xmysa = np.ravel(np.column_stack((self.xnodes[:-1] + node2mysa, self.xnodes[1:] - node2mysa)))  # um
        mysa2flut = 0.5 * (self.mysaL + self.flutL)  # um
        self.xflut = np.ravel(np.column_stack((self.xmysa[::2] + mysa2flut, self.xmysa[1::2] - mysa2flut)))  # um
        xref = self.xflut[::2] + 0.5 * (self.flutL + self.stinL)  # um
        self.xstin = np.ravel([xref + i * self.stinL for i in range(self.nstin_per_inter)], order='F')  # um
        self.xsections = np.hstack((self.xnodes, self.xmysa, self.xflut, self.xstin))

    def build_topology(self):
        ''' connect the sections together '''
        logger.debug('connecting model sections...')
        # PATTERN: childSection.connect(parentSection, [parentX], [childEnd])
        for i in range(self.ninters):
            self.node[i].connect(self.mysa[2 * i], 1, 0)  # node -> MYSA
            self.mysa[2 * i].connect(self.flut[2 * i], 1, 0)  # MYSA -> FLUT
            self.flut[2 * i].connect(self.stin[6 * i], 1, 0)  # FLUT -> STIN
            for j in range(5):  # STIN -> STIN -> STIN -> STIN -> STIN
                self.stin[6 * i + j].connect(self.stin[6 * i + j + 1], 1, 0)
            self.stin[6 * i + 5].connect(self.flut[2 * i + 1], 1, 0)  # STIN -> FLUT
            self.flut[2 * i + 1].connect(self.mysa[2 * i + 1], 1, 0)  # FLUT -> MYSA
            self.mysa[2 * i + 1].connect(self.node[i + 1], 1, 0)  # MYSA -> node

    def define_biophysics(self):
        ''' Assign the membrane properties across the cell. '''
        logger.debug('defining sections biophysics...')
        # Common to all sections
        for sec in self.node + self.mysa + self.flut + self.stin:
            sec.nseg = 1
            sec.Ra = self.rhoa  # Ohm.cm
            sec.cm = self.cm  # uF/cm2

        # Common to all internodal sections
        for sec in self.mysa + self.flut + self.stin:
            sec.insert('pas')
            sec.e_pas = self.vrest  # mV
            sec.insert('extracellular')
            sec.xc[0] = self.mycm  # uF/cm2
            sec.xg[0] = self.mygm  # S/cm2

        # Node-specific
        for sec in self.node:
            sec.diam = self.nodeD  # um
            sec.L = self.nodeL  # um
            sec.insert('MRGnode')
            sec.insert('extracellular')
            sec.xraxial[0] = self.Rp_node * OHM_TO_MOHM  # MOhm/cm
            sec.xc[0] = 0  # uF/cm2
            sec.xg[0] = 1e10  # S/cm2

        # MYSA-specific
        for sec in self.mysa:
            sec.diam = self.mysaD  # um
            sec.L = self.mysaL  # um
            sec.g_pas = self.g_mysa  # S/cm2
            sec.xraxial[0] = self.Rp_mysa * OHM_TO_MOHM  # MOhm/cm

        # FLUT-specific
        for sec in self.flut:
            sec.diam = self.flutD  # um
            sec.L = self.flutL  # um
            sec.g_pas = self.g_flut  # S/cm2
            sec.xraxial[0] = self.Rp_flut * OHM_TO_MOHM  # MOhm/cm

        # STIN-specific
        for sec in self.stin:
            sec.diam = self.stinD  # um
            sec.L = self.stinL  # um
            sec.g_pas = self.g_stin  # S/cm2
            sec.xraxial[0] = self.Rp_stin * OHM_TO_MOHM  # MOhm/cm

    def get_details(self):
        ''' Return a pandas dataframe with the parametric details of each section type '''
        row_labels = ['node', 'MYSA', 'FLUT', 'STIN']
        col_units = {
            'nsec': None,
            'nseg': None,
            'diam': 'um',
            'L': 'um',
            'cm': 'uF/cm2',
            'Ra': 'Ohm.cm', 
            'xr': 'MOhm/cm',
            'xg': 'S/cm2',
            'xc': 'uF/cm2'
        }
        col_labels = [f'{k} ({v})' if v is not None else k for k, v in col_units.items()]
        d = []
        for seclist in [self.node, self.mysa, self.flut, self.stin]:
            sec = seclist[0]
            d.append([len(seclist), sec.nseg, sec.diam, sec.L, sec.cm, sec.Ra,
                      sec.xraxial[0], sec.xg[0], sec.xc[0]])
        df = pd.DataFrame(data=np.array(d), index=row_labels, columns=col_labels)
        df = df.astype({'nsec': 'int', 'nseg': 'int'})
        return df

    def delete_sections(self):
        ''' delete all model sections. '''
        del self.node
        del self.mysa
        del self.flut
        del self.stin

