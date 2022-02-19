# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-05 14:08:31
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-02-18 20:26:07

import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from neuron import h
from IPython.display import display
from ipywidgets import interact, FloatSlider, VBox, interactive_output

from constants import *
from logger import logger


class Simulation:
    ''' 
    Interface to run NEURON simulations.
    '''

    def __init__(self, fiber, medium, stim, tstop=None):
        '''
        Object initialization.
        
        :param medium: 
        :param fiber: fiber model object
        :param medium: extracellular medium (volume conductor) object
        :param stim: simulus object
        :param tstop (optional): total simulation time
        '''
        # Set global simulation parameters
        h.celsius = 36.  # temperature (Celsius)
        h.dt = 0.01  # time step (ms)

        # Assign input arguments as class attributes
        self.fiber = fiber
        self.stim = stim
        self.medium = medium
        if tstop is None:
            tstop = 1.5 * self.stim.stim_events()[-1][0]
        self.tstop = tstop  # ms

        # Compute extracellular field per unit extracellular current (if external stimulus) 
        if hasattr(stim, 'pos'):
            self.rel_phis = self.get_phi(self.fiber.xsections)
          
        # Set internal objects
        self.internals = []
    
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
        ns = self.get_netstim(**kwargs)
        syn = self.get_expsyn(self.fiber.node[0])
        nc = self.connect_netstim_to_synapse(ns, syn)
        self.internals.append((ns, syn, nc))

    def get_phi(self, x, I=1.):
        ''' 
        Compute the extracellular potential at a particular section axial coordinate
        for a specific current amplitude

        :param x: axial position of the section on the fiber (um)
        :param I: current amplitude
        :return: extracellular membrane voltage (mV)
        '''
        if isinstance(x, (np.ndarray, list, tuple)):
            x = np.vstack((np.atleast_2d(x), np.zeros((2, x.size)))).T
        else:
            x = np.array((x, 0., 0.))
        d = self.fiber.pos + x - self.stim.pos  # (um, um, um)
        return self.medium.phi(I, d)
    
    def get_activating_function(self, x, phi):
        ''' Compute activating function from a distribution of positions and voltages '''
        d2phidx2 = np.diff(phi, 2) / np.diff(x)[:-1]**2
        return np.hstack(([0.], d2phidx2, [0.]))

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
        logger.debug(f'simulation completed in {tend - tstart:.2f} s')
        return tvec, vnodes
    
    def plot_vprofile(self, ax=None, update=False, redraw=False):
        '''
        Plot the spatial distribution of the extracellular potential along the fiber
        
        :param ax (optional): axis on which to plot
        :return: figure handle
        '''
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.set_xlabel(AX_POS_MM)
            sns.despine(ax=ax)
        else:
            fig = ax.get_figure()
        xnodes = self.fiber.xnodes  # um
        phinodes = self.get_phi(xnodes, I=self.stim.I) # mV
        if update:
            line = ax.get_lines()[0]
            line.set_xdata(xnodes * 1e-3)
            line.set_ydata(phinodes)
            ax.relim()
            ax.autoscale_view()
        else:
            ax.set_title('potential distribution along fiber')
            ax.set_ylabel(PHI_MV)
            ax.plot(xnodes * 1e-3, phinodes)
        if update and redraw:
            fig.canvas.draw()
        return fig
    
    def plot_activating_function(self, ax=None, update=False, redraw=False):
        '''
        Plot the spatial distribution of the activating function along the fiber
        
        :param ax (optional): axis on which to plot
        :return: figure handle
        '''
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.set_xlabel(AX_POS_MM)
            sns.despine(ax=ax)
        else:
            fig = ax.get_figure()
        xnodes = self.fiber.xnodes  # um
        phinodes = self.get_phi(xnodes, I=self.stim.I)  # mV
        d2phidx2 = self.get_activating_function(xnodes * 1e-3, phinodes)  # mV2/mm2
        if update:
            line = ax.get_lines()[0]
            line.set_xdata(xnodes * 1e-3)
            line.set_ydata(d2phidx2)
            ax.relim()
            ax.autoscale_view()
        else:
            ax.set_title('activating function along fiber')
            ax.set_ylabel('d2_phi/dx2 (mV2/mm2)')
            ax.plot(xnodes * 1e-3, d2phidx2)
        if update and redraw:
            fig.canvas.draw()
        return fig
    
    def plot_profiles(self, fig=None):
        '''
        Plot profiles of extracellular potential and activating function along the fiber
        
        :return: figure handle
        '''
        # Get figure
        if fig is None:
            fig, axes = plt.subplots(2, figsize=(6, 4), sharex=True)
            update = False
        else:
            axes = fig.axes
            update = True
        self.plot_vprofile(ax=axes[0], update=update, redraw=False)
        self.plot_activating_function(ax=axes[1], update=update, redraw=False)
        if not update:
            for ax in axes[:-1]:
                sns.despine(ax=ax, bottom=True)
                ax.xaxis.set_ticks_position('none')
            sns.despine(ax=axes[-1])
            axes[-1].set_xlabel(AX_POS_MM)
        else:
            fig.canvas.draw()
        return fig

    def plot_vmap(self, tvec, vnodes, ax=None, update=False, redraw=True):
        '''
        Plot 2D colormap of membrane potential across nodes and time

        :param tvec: time vector (ms)
        :param vnodes: 2D array of membrane voltage of nodes and time
        :param ax (optional): axis on which to plot
        :param update: whether to update an existing figure or not
        :param redraw: whether to redraw figure upon update
        :return: figure handle
        '''
        y = self.fiber.xnodes * 1e-3  # mm
        if ax is None:
            fig, ax = plt.subplots(figsize=(np.ptp(tvec), np.ptp(y) / 50))
            ax.set_xlabel(TIME_MS)
            sns.despine(ax=ax, offset={'left': 10., 'bottom': 10})
        else:
            fig = ax.get_figure()
        # Get normalizer and scalar mapable
        vlims = (min(vnodes.min(), V_LIMS[0]), max(vnodes.max(), V_LIMS[1]))
        norm = plt.Normalize(*vlims)
        sm = cm.ScalarMappable(norm=norm, cmap='viridis')
        if not update:
            # Plot map
            ax.set_ylabel(AX_POS_MM)
            self.pm = ax.pcolormesh(tvec, y, vnodes, norm=norm, cmap='viridis')
            fig.subplots_adjust(right=0.8)
            pos = ax.get_position()
            self.cax = fig.add_axes([pos.x1 + .02, pos.y0, 0.02, pos.y1 - pos.y0])
            self.cbar = fig.colorbar(sm, cax=self.cax)
        else:
            self.pm.set_array(vnodes)
            self.cbar.update_normal(sm)
        # Add colorbar
        self.cbar.set_ticks(vlims)
        if not update:
            self.cbar.set_label(V_MV, labelpad=-15)
        if update and redraw:
            fig.canvas.draw()
        return fig

    def plot_vtraces(self, tvec, vnodes, ax=None, inodes=None, update=False, redraw=True):
        '''
        Plot membrane potential traces at specific nodes

        :param tvec: time vector (ms)
        :param vnodes: 2D array of membrane voltage of nodes and time
        :param ax (optional): axis on which to plot
        :param inodes (optional): specific node indexes
        :param update: whether to update an existing figure or not
        :param redraw: whether to redraw figure upon update
        :return: figure handle
        '''
        if ax is None:
            fig, ax = plt.subplots(figsize=(np.ptp(tvec), 3))
            ax.set_xlabel(TIME_MS)
            sns.despine(ax=ax)
        else:
            fig = ax.get_figure()
        nnodes = vnodes.shape[0]
        if inodes is None:
            inodes = [0, nnodes // 2, nnodes - 1]
        vtraces = {f'node {inode}': vnodes[inode, :] for inode in inodes}
        if update:
            for line, (label, vtrace) in zip(ax.get_lines(), vtraces.items()): 
                line.set_xdata(tvec)
                line.set_ydata(vtrace)
            ax.relim()
            ax.autoscale_view()
        else:
            for label, vtrace in vtraces.items():
                ax.plot(tvec, vtrace, label=label)
            ax.legend(loc=9, bbox_to_anchor=(0.95, 0.9))
            ax.set_ylabel(V_MV)
            ax.set_xlim([tvec[0], tvec[-1]])
        ax.autoscale(True)
        ylims = ax.get_ylim()
        ax.set_ylim(min(ylims[0], V_LIMS[0]), max(ylims[1], V_LIMS[1]))
        if update and redraw:
            fig.canvas.draw()
        return fig
    
    def plot_Itrace(self, ax=None, update=False, redraw=True):
        '''
        Plot stimulus time profile

        :param ax (optional): axis on which to plot
        :param update: whether to update an existing figure or not
        :param redraw: whether to redraw figure upon update
        :return: figure handle
        '''
        if ax is None:
            fig, ax = plt.subplots(figsize=(self.tstop, 3))
            ax.set_xlabel(TIME_MS)
            sns.despine(ax=ax)
        else:
            fig = ax.get_figure()
        tstim, Istim = self.stim.stim_profile()
        tstim = np.hstack((tstim, [self.tstop]))
        Istim = np.hstack((Istim, [Istim[-1]]))
        if update:
            line = ax.get_lines()[0]
            line.set_xdata(tstim)
            line.set_ydata(Istim)
            ax.relim()
            ax.autoscale_view()
        else:
            ax.plot(tstim, Istim, color='k')
            ax.set_ylabel(f'Istim ({self.stim.unit})')
        if update and redraw:
            fig.canvas.draw()
        return fig

    def plot_results(self, tvec, vnodes, inodes=None, fig=None):
        ''' 
        Plot simulation results.

        :param tvec: time vector (ms)
        :param vnodes: 2D array of membrane voltage of nodes and time
        :param ax (optional): axis on which to plot
        :param inodes (optional): specific node indexes
        :param fig (optional): existing figure to use for rendering 
        :param update: whether to update an existing figure or not
        :return: figure handle
        '''
        # Get figure
        if fig is None:
            fig, axes = plt.subplots(3, figsize=(7, 5), sharex=True)
            update = False
        else:
            axes = fig.axes
            update = True
        # Plot results   
        self.plot_vmap(tvec, vnodes, ax=axes[0], update=update, redraw=False)
        self.plot_vtraces(tvec, vnodes, ax=axes[1], inodes=inodes, update=update, redraw=False)
        self.plot_Itrace(ax=axes[2], update=update, redraw=False)        
        # Adjust axes and figure
        if not update:
            for ax in axes[:-1]:
                sns.despine(ax=ax, bottom=True)
                ax.xaxis.set_ticks_position('none')
            sns.despine(ax=axes[-1])
            axes[-1].set_xlabel(TIME_MS)
        else:
            fig.canvas.draw()
        # Return figure
        return fig


def interactive_profiles(sim, Imin=-100., Imax=100., dzmin=10., dzmax=1000.):
    I = FloatSlider(
        description='I (uA)', min=Imin, max=Imax, value=0, step=(Imax - Imin) / 100, continuous_update=False)
    dz = FloatSlider(
        description='Δz (um)', min=dzmin, max=dzmax, value=100., step=(dzmax - dzmin) / 100, continuous_update=False)
    ui = VBox([I, dz])
    fig = sim.plot_profiles()
    def update(I, dz):
        sim.stim.I = I
        sim.stim.pos = (0., 0., dz)
        return sim.plot_profiles(fig=fig)
    out = interactive_output(update, {'I': I, 'dz': dz})
    return display(ui, out)


def interactive_sim(sim, Imin=-100., Imax=500., PWmin=.01, PWmax=1.):
    I = FloatSlider(
        description='I (uA)', min=Imin, max=Imax, value=0, step=(Imax - Imin) / 100, continuous_update=False)
    PW = FloatSlider(
        description='PW (ms)', min=PWmin, max=PWmax, value=0.1, step=(PWmax - PWmin) / 100, continuous_update=False)
    ui = VBox([I, PW])    
    tvec, vnodes = sim.run()
    fig = sim.plot_results(tvec, vnodes)
    def update(I, PW):
        sim.stim.I = I
        sim.stim.tpulse = PW
        tvec, vnodes = sim.run()
        return sim.plot_results(tvec, vnodes, fig=fig)
    out = interactive_output(update, {'I': I, 'PW': PW})
    return display(ui, out)