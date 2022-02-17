# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-05 14:08:31
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-02-17 18:41:06

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from neuron import h

from constants import *
from fibermodel import MyelinatedFiber
from logger import logger


def phi(I, r, sigma=0.5):
    '''
    Compute the induced extracellular potential at a specific distance from
    a point-current source in an isotropic volume conductor model.

    :param I: current amplitude (uA)
    :param r: distance (um)
    :param sigma: medium conductivity (S/m)
    :return: extracellular potential (mV)
    '''
    # uA / (S/m * um) = 1e-6 A / (S/m * 1e-6 m) = 1e0 A / S = 1e0 V = 1e3 mV 
    return I / (4 * np.pi * sigma * r) * V_TO_MV


class Simulation:
    ''' 
    Interface to run NEURON simulations.
    '''

    def __init__(self, fiber, stim, phifunc=phi, tstop=None):
        '''
        Object initialization.
        
        :param fiber: fiber model object
        :param stim: simulus object
        :param phifunc: function converting extracellular current to extracellular potential
        :param tstop (optional): total simulation time
        '''
        # Set global simulation parameters
        h.celsius = 36.  # temperature (Celsius)
        h.dt = 0.01  # time step (ms)

        # Assign input arguments as class attributes
        self.fiber = fiber
        self.stim = stim
        self.phifunc = phifunc
        if tstop is None:
            tstop = 1.5 * self.stim.stim_events()[-1][0]
        self.tstop = tstop  # ms

        # Compute extracellular field per unit extracellular current (if external stimulus) 
        if hasattr(stim, 'pos'):
            self.rel_phis = np.array([self.get_rel_phi(x) for x in self.fiber.xsections])  
  
        # Set internal objects
        self.synapses = []
        self.netstims = []
        self.netcons = []
    
    def get_rel_phi(self, x):
        ''' 
        Compute the extracellular potential at a particular section axial coordinate
        for a unit current amplitude (1 uA)
        '''
        d = self.fiber.pos + np.array((x, 0., 0.)) - self.stim.pos  # (um, um, um)
        r = np.sqrt(np.sum(np.square(d)))  # um
        return self.phifunc(1., r)

    def update_field(self, I):
        ''' 
        Update the extracellular potential value of each fiber section for a specific
        stimulation current. '''
        logger.debug(f't = {h.t:.2f} ms, setting I = {I} {self.stim.unit}')
        for sec, rel_phi in zip(self.fiber.sections, self.rel_phis):
            sec.e_extracellular = I * rel_phi

    def get_update_field_callable(self, value):
        ''' Get callable to update_field with a preset current amplitude '''
        return lambda: self.update_field(value)

    def run(self):
        ''' Run the simulation. '''
        logger.info(f'simulating {self.fiber} stimulation by {self.stim}...')
        tstart = time.perf_counter()
        # Set probes
        tprobe = h.Vector().record(h._ref_t)
        vprobes = [h.Vector().record(self.fiber.node[j](0.5)._ref_v)
                   for j in range(self.fiber.nnodes)]
        # Set field
        h.t = 0.
        self.update_field(0.)
        # Set integration parameters
        self.cvode = h.CVode()
        self.cvode.active(0)
        logger.debug(f'fixed time step integration (dt = {h.dt} ms)')
        # Initialize
        h.finitialize(self.fiber.vrest)
        # Set modulation events
        events = self.stim.stim_events()
        for t, I in events:
            self.cvode.event(t, self.get_update_field_callable(I))
        # Integrate
        while h.t < self.tstop:
            h.fadvance()
        # Extract results
        tvec = np.array(tprobe.to_python())
        vnodes = np.array([v.to_python() for v in vprobes])
        tend = time.perf_counter()
        logger.info(f'simulation completed in {tend - tstart:.2f} s')
        return tvec, vnodes

    def plot_results(self, tvec, vnodes, inodes=None):
        ''' Plot simulation results. '''
        fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)

        # 2D colormap of membrane potential across nodes and time
        ax = axes[0]
        nnodes, nsamples = vnodes.shape
        im = ax.pcolormesh(tvec, self.fiber.xnodes * 1e-3, vnodes)
        fig.subplots_adjust(right=0.9)
        pos = ax.get_position()
        cbax = fig.add_axes([pos.x1 + .02, pos.y0, 0.02, pos.y1 - pos.y0])
        fig.colorbar(im, cax=cbax)
        ax.set_title('membrane potential (mV)')
        ax.set_ylabel('axial position (mm)')
        sns.despine(ax=ax, bottom=True, offset={'left': 10.})
        ax.xaxis.set_ticks_position('none')

        # Membrane potential traces at subset of nodes
        ax = axes[1]
        if inodes is None:
            inodes = [0, nnodes // 2, nnodes - 1]
        for inode in inodes:
            ax.plot(tvec, vnodes[inode, :], label=f'node {inode}')
        ax.legend(loc=9, bbox_to_anchor=(0.95, 0.9))
        sns.despine(ax=ax, bottom=True, offset={'left': 10.})
        ax.set_ylabel(f'membrane potential at \n{inodes} (mV)')
        ax.set_xlim([0, self.tstop])
        ax.xaxis.set_ticks_position('none')

        # Stimulus time profile
        ax = axes[2]
        tstim, Istim = self.stim.stim_profile()
        tstim = np.hstack((tstim, [self.tstop]))
        Istim = np.hstack((Istim, [Istim[-1]]))
        ax.plot(tstim, Istim, color='k')
        ax.set_ylabel(f'Istim ({self.stim.unit})')
        ax.set_xlabel('time (ms)')
        sns.despine(ax=ax, offset={'left': 10., 'bottom': 10.})

        return fig
    
    def get_expsyn(self, sec, tau=0.1, e=50.):
        ''' 
        Create a synapse with discontinuous change in conductance at an event followed by
        an exponential decay with time constant tau.
        
        :param sec: section object
        :param tau: decay time constant (ms)
        :param e: reversal potential (mV) 
        :return: ExpSyn object
        '''
        syn = h.ExpSyn(sec(0.5))
        syn.tau = tau
        syn.e = e
        return syn

    def get_netstim(self, number=1000, freq=10., start=0., noise=0):
        '''
        Create a NetStim object representing a train of presynaptic stimuli. 
        
        :param number: number of spikes (defaults to 1000)
        :param freq: presynaptic spiking frequency (Hz)
        :param start: start time of the first spike (ms)
        :param noise: fractional randomness (0 to 1)
        :return: NetStim object
        '''
        ns = h.NetStim()
        ns.interval = 1e3 / freq  # interval between spikes (ms)
        ns.number = number
        ns.start = start
        ns.noise = noise
        return ns
    
    def connect_netstim_to_synapse(self, ns, syn, weight=1., delay=1.):
        '''
        Connect a NetStim to a synapse
        
        :param ns: NetStim object
        :param syn: synapse object
        :return: NetCon object
        '''
        nc = h.NetCon(ns, syn)
        nc.weight[0] = weight
        nc.delay = delay
        return nc

    def add_presynaptic_input(self, **kwargs):
        '''
        Add a pre-synaptic input on the first fiber node ot induce "physiological" spiking
        at a specific frequency. This is done by:
        - creating a Synapse object on the node of interest
        - creating a NetStim object representing the presynaptic input
        - connecting the two via a NetCon object

        :param sec: section object
        :param kwargs: keyword arguments for NetStim creation
        '''
        self.synapses.append(self.get_expsyn(self.fiber.node[0]))
        self.netstims.append(self.get_netstim(**kwargs))
        self.netcons.append(self.connect_netstim_to_synapse(
            self.netstims[-1], self.synapses[-1]))

