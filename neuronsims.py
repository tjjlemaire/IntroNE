# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-05 14:08:31
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-03-08 17:08:50

import inspect
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
import seaborn as sns
from neuron import h
from IPython.display import display
from ipywidgets import FloatSlider, FloatLogSlider, VBox, interactive_output
from scipy.signal import find_peaks

from constants import *
from logger import logger


class Simulation:
    ''' 
    Interface to run NEURON simulations.
    '''

    def __init__(self, axon, medium, stim):
        '''
        Object initialization.
        
        :param medium: 
        :param axon: axon model object
        :param medium: extracellular medium (volume conductor) object
        :param stim: simulus object
        '''
        # Set global simulation parameters
        h.celsius = 36.  # temperature (Celsius)
        h.dt = 0.01  # time step (ms)

        # Assign input arguments as class attributes
        self.axon = axon
        self.stim = stim
        self.medium = medium

        # Compute extracellular field per unit extracellular current (if external stimulus) 
        if hasattr(stim, 'pos'):
            self.rel_phis = self.get_phi(self.axon.xsections)
          
        # Set internal objects
        self.internals = []
    
    def copy(self):
        return self.__class__(
            axon=self.axon.copy(),
            medium=self.medium.copy(),
            stim=self.stim.copy(),
            tstop=self.tstop
        )
    
    def reset(self):
        self.axon.reset()
        self.medium.reset()
        self.stim.reset()
    
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
        Add a pre-synaptic input on the first axon node ot induce "physiological" spiking
        at a specific frequency. This is done by:
        - creating a Synapse object on the node of interest
        - creating a NetStim object representing the presynaptic input
        - connecting the two via a NetCon object

        :param sec: section object
        :param kwargs: keyword arguments for NetStim creation
        '''
        ns = self.get_netstim(**kwargs)
        syn = self.get_expsyn(self.axon.node[0])
        nc = self.connect_netstim_to_synapse(ns, syn)
        self.internals.append((ns, syn, nc))

    def get_phi(self, x, I=1., relaxon=True):
        ''' 
        Compute the extracellular potential at a particular section axial coordinate
        for a specific current amplitude

        :param x: axial position of the section on the axon (um)
        :param I: current amplitude
        :return: extracellular membrane voltage (mV)
        '''
        # If 1D array provided, assume vector of axial (x) positions
        if isinstance(x, (list, tuple)) or (isinstance(x, np.ndarray) and x.ndim == 1):
            x = np.vstack((np.atleast_2d(x), np.zeros((2, x.size)))).T
        # Other array cases
        elif isinstance(x, np.ndarray):
            # If 2D array provided, assume vector of x, y, z positions
            if x.ndim == 2:
                dims = x.shape
                if dims[1] != 3:
                    raise ValueError('2D arrays must have the shape (npoints, 3)')
            else:
                raise ValueError('only 1D and 2D coordinate arrays are accepted')
        # If scalar provided, assume it is the axial coordinate and fill the rest with zeros
        else:
            x = np.array((x, 0., 0.))
        d = x - self.stim.pos  # (um, um, um)
        if relaxon:
            d += self.axon.pos
        return self.medium.phi(I, d)
    
    def get_phi_map(self, x, z):
        '''
        Compute extracellular potential map over a grid of x and z coordinates

        :param x: vector of relative x (axial) coordinates w.r.t. the electrode (um)
        :param z: vector of relative z (transverse) coordinates w.r.t. the electrode (um)
        :return: 2D (nx, nz) array of extracellular potential values (mV)  
        '''
        # Construct 3D meshgrid from x and y vectors (adding a zero z vector)
        Y, Z, X = np.meshgrid([0], z, x)  # y, z, x order to get appropriate refinement
        # Flatten meshgrid onto coordinates array
        coords = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
        # Compute extracellular potential value for each coordinate
        phis = self.get_phi(coords, relaxon=False)
        # Reshape result to initial 2D field
        return np.reshape(phis, (x.size, z.size))
    
    def get_activating_function(self, x, phi):
        ''' Compute activating function from a distribution of positions and voltages '''
        d2phidx2 = np.diff(phi, 2) / np.diff(x)[:-1]**2
        return np.hstack(([0.], d2phidx2, [0.]))

    def update_field(self, I):
        ''' 
        Update the extracellular potential value of each axon section for a specific
        stimulation current. '''
        logger.debug(f't = {h.t:.2f} ms, setting I = {I} {self.stim.unit}')
        for sec, rel_phi in zip(self.axon.sections, self.rel_phis):
            sec.e_extracellular = I * rel_phi

    def get_update_field_callable(self, value):
        ''' Get callable to update_field with a preset current amplitude '''
        return lambda: self.update_field(value)

    def run(self, verbose=True):
        '''
        Run the simulation.
        
        :param verbose: whether to print out details of the simulation
        :return: (tvec, vnodes) tuple:
            - tvec is a 1D (nsamples) time vector of the simulation (in ms)
            - vnodes is a 2D (nnodes x nsamples) array of voltage values for each axon node over time (in mV)
        '''
        self.rel_phis = self.get_phi(self.axon.xsections)
        self.tstop = max(10., 1.5 * self.stim.stim_events()[-1][0])
        if verbose:
            logger.info(f'simulating {self.axon} stimulation by {self.stim}...')
        tstart = time.perf_counter()
        # Set probes
        tprobe = h.Vector().record(h._ref_t)
        vprobes = [h.Vector().record(self.axon.node[j](0.5)._ref_v)
                   for j in range(self.axon.nnodes)]
        # Set field
        h.t = 0.
        self.update_field(0.)
        # Set integration parameters
        self.cvode = h.CVode()
        self.cvode.active(0)
        logger.debug(f'fixed time step integration (dt = {h.dt} ms)')
        # Initialize
        h.finitialize(self.axon.vrest)
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

    def plot_phi_map(self, x, z, ax=None, update=False, redraw=True, scale='log', contour=False):
        '''
        Plot 2D colormap of extracellular potential across a 2D space
        
        :param x: vector of absolute x (axial) coordinates (um)
        :param z: vector of absolute z (transverse) coordinates (um)
        :param ax (optional): axis on which to plot
        :param update: whether to update an existing figure or not
        :param redraw: whether to redraw figure upon update
        :return: figure handle
        '''
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_xlabel(TIME_MS)
            sns.despine(ax=ax, offset={'left': 10., 'bottom': 10})
        else:
            fig = ax.get_figure()
        # Compute 2D field of values
        phis = self.get_phi_map(x, z)
        phi_ub = np.percentile(phis, 99)
        # Get normalizer and scalar mapable
        philims = (phis.min(), phi_ub)
        if scale == 'lin':
            norm = plt.Normalize(*philims)
        else:
            norm = LogNorm(*philims)
        sm = cm.ScalarMappable(norm=norm, cmap='viridis')
        if not update:
            # Plot map
            ax.set_xlabel(AX_POS_MM)
            ax.set_ylabel('transverse position (mm)')
            self.pm = ax.pcolormesh(x * 1e-3, z * 1e-3, phis, norm=norm, cmap='viridis')
            ax.set_aspect(1.)
            fig.subplots_adjust(right=0.8)
            pos = ax.get_position()
            self.cax = fig.add_axes([pos.x1 + .02, pos.y0, 0.02, pos.y1 - pos.y0])
            self.cbar = fig.colorbar(sm, cax=self.cax)
            if contour:
                ax.contour(x * 1e-3, z * 1e-3, phis, levels=[phi_ub / 2], colors='w')
        else:
            self.pm.set_array(phis)
            self.cbar.update_normal(sm)
        # Add colorbar
        if scale == 'lin':
            self.cbar.set_ticks(philims)
        if not update:
            self.cbar.set_label(PHI_MV, labelpad=-15)
        if update and redraw:
            fig.canvas.draw()
        return fig
    
    def plot_config(self, nperax=100, zoomout=2., contour=True, **kwargs):
        ''' Plot system configuration in the xz plane '''
        # Get Z-bounds of XZ plane of interest
        zstim = self.stim.pos[-1]  # z-electrode (um)
        zaxon = self.axon.pos[-1]  # z-axon (um)
        zbounds = sorted([zstim, zaxon])  # sorted z coordinates
        zmid = np.mean(zbounds)  # mid-z coordinate
        dz = np.ptp(zbounds)  # z-span between electrode and axon
        zbounds = [zmid - zoomout * dz, zmid + zoomout * dz]  # z-bounds: twice z-span, centered around mid-z 
        # Get X-bounds of XZ plane of interest
        xstim = self.stim.pos[0]  # x-electrode (um)
        xaxon = self.axon.pos[0]  # x-axon (um)
        xmid = (xstim + xaxon) / 2  # mid-x coordinate (um)
        xbounds = np.array(zbounds) - np.mean(zbounds) + xmid  # um
        # Compute and plot phi map over 100-by-100 grid spanning the XZ plane
        zgrid = np.linspace(*zbounds, nperax)
        xgrid = np.linspace(*xbounds, nperax)
        fig = self.plot_phi_map(xgrid, zgrid, contour=contour, **kwargs)
        ax = fig.axes[0]
        # Add marker for stim position
        ax.scatter([xstim * 1e-3], [zstim * 1e-3], label='electrode')
        # Add markers for axon nodes
        xnodes = self.axon.xnodes + self.axon.pos[0] # um
        xnodes = xnodes[np.logical_and(xnodes >= xbounds[0], xnodes <= xbounds[-1])]
        znodes = np.ones(xnodes.size) * zaxon  # um
        ax.axhline(zaxon * 1e-3, c='silver', lw=4, label='axon axis')
        ax.scatter(xnodes * 1e-3, znodes * 1e-3, zorder=80, color='k', label='nodes')
        ax.legend()
        return fig

    def plot_vprofile(self, ax=None, update=False, redraw=False):
        '''
        Plot the spatial distribution of the extracellular potential along the axon
        
        :param ax (optional): axis on which to plot
        :return: figure handle
        '''
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.set_xlabel(AX_POS_MM)
            sns.despine(ax=ax)
        else:
            fig = ax.get_figure()
        xnodes = self.axon.xnodes  # um
        phinodes = self.get_phi(xnodes, I=self.stim.I) # mV
        if update:
            line = ax.get_lines()[0]
            line.set_xdata(xnodes * 1e-3)
            line.set_ydata(phinodes)
            ax.relim()
            ax.autoscale_view()
        else:
            ax.set_title('potential distribution along axon')
            ax.set_ylabel('φ (mV)')
            ax.plot(xnodes * 1e-3, phinodes)
        if update and redraw:
            fig.canvas.draw()
        return fig
    
    def plot_activating_function(self, ax=None, update=False, redraw=False):
        '''
        Plot the spatial distribution of the activating function along the axon
        
        :param ax (optional): axis on which to plot
        :return: figure handle
        '''
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.set_xlabel(AX_POS_MM)
            sns.despine(ax=ax)
        else:
            fig = ax.get_figure()
        xnodes = self.axon.xnodes  # um
        phinodes = self.get_phi(xnodes, I=self.stim.I)  # mV
        d2phidx2 = self.get_activating_function(xnodes * 1e-3, phinodes)  # mV2/mm2
        if update:
            line = ax.get_lines()[0]
            line.set_xdata(xnodes * 1e-3)
            line.set_ydata(d2phidx2)
            ax.relim()
            ax.autoscale_view()
        else:
            ax.set_title('activating function along axon')
            ax.set_ylabel('d2φ/dx2 (mV2/mm2)')
            ax.plot(xnodes * 1e-3, d2phidx2)
        if update and redraw:
            fig.canvas.draw()
        return fig
    
    def plot_profiles(self, fig=None):
        '''
        Plot profiles of extracellular potential and activating function along the axon
        
        :return: figure handle
        '''
        # Get figure
        if fig is None:
            fig, axes = plt.subplots(2, figsize=(8, 4), sharex=True)
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

    def plot_vmap(self, tvec, vnodes, ax=None, update=False, redraw=True, add_rec_locations=False):
        '''
        Plot 2D colormap of membrane potential across nodes and time

        :param tvec: time vector (ms)
        :param vnodes: 2D array of membrane voltage of nodes and time
        :param ax (optional): axis on which to plot
        :param update: whether to update an existing figure or not
        :param redraw: whether to redraw figure upon update
        :param add_rec_locations: whether to add recruitment locations (predicted from
            activatinvg function) on the map
        :return: figure handle
        '''
        y = np.arange(self.axon.nnodes)
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
            ax.set_ylabel('# nodes')
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
        if add_rec_locations:
            # Compute activating function profile
            xnodes = self.axon.xnodes  # um
            phinodes = self.get_phi(xnodes, I=self.stim.I)  # mV
            d2phidx2 = self.get_activating_function(xnodes * 1e-3, phinodes)  # mV2/mm2 
            # Infer recruitment location(s) from maximum point(s) of activating function
            psimax = np.max(d2phidx2)
            if psimax > 0.: 
                irecnodes = np.where(np.isclose(d2phidx2, psimax))
                xrec = xnodes[irecnodes]
            # Remove previous lines
            if update:
                lines = ax.get_lines()
                while lines:
                    l = lines.pop(0)
                    l.remove()
            # Add current lines
            for x in xrec:
                ax.axhline(x * 1e-3, c='r', ls='--')
        if update and redraw:
            fig.canvas.draw()
        return fig

    def plot_vtraces(self, tvec, vnodes, ax=None, inodes=None, update=False, redraw=True, mark_spikes=False):
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
                if mark_spikes:
                    ispikes = self.detect_spikes(tvec, vtrace)
                    if len(ispikes) > 0:
                        ax.scatter(tvec[ispikes], vtrace[ispikes] + 10, marker='v')
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
        if tstim[-1] > self.tstop:
            Istim = Istim[tstim < self.tstop]
            tstim = tstim[tstim < self.tstop]
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

    def plot_results(self, tvec, vnodes, inodes=None, fig=None, mark_spikes=False):
        ''' 
        Plot simulation results.

        :param tvec: time vector (ms)
        :param vnodes: 2D array of membrane voltage of nodes and time
        :param ax (optional): axis on which to plot
        :param inodes (optional): specific node indexes
        :param fig (optional): existing figure to use for rendering 
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
        self.plot_vtraces(tvec, vnodes, ax=axes[1], inodes=inodes, update=update, redraw=False, mark_spikes=mark_spikes)
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

    def detect_spikes(self, t, v):
        '''
        Detect spikes in simulation output data.

        :param t: time vector
        :param v: 1D or 2D membrane potential array
        :return: time indexes of detected spikes:
            - If a 1D voltage array is provided, a single list is returned.
            - If a 2D voltage array is provided, a list of lists is returned (1 list per node)

        Example use:
        ispikes = sim.detect_spikes(tvec, vnodes)
        '''
        if v.ndim > 2: 
            raise ValueError('cannot work with potential arrays of more than 2 dimensions')
        if v.ndim == 2:
            ispikes = [self.detect_spikes(t, vv) for vv in v]
            if all(len(i) == len(ispikes[0]) for i in ispikes):
                ispikes = np.array(ispikes)
            return ispikes
        return find_peaks(v, height=0., prominence=50.)[0]


def copy_slider(slider, **kwargs):
    '''
    Copy an ipywidgets slider object
    
    :param slider: reference slider
    :param kwargs: attributes to be overwritten
    :return: slider copy
    '''
    # Get slider copy
    if isinstance(slider, FloatSlider):
        s = FloatSlider(
            description=slider.description,
            min=slider.min, max=slider.max, value=slider.value, step=slider.step,
            continuous_update=slider.continuous_update, layout=slider.layout)
    elif isinstance(slider, FloatLogSlider):
        s = FloatLogSlider(
            description=slider.description,
            base=slider.base, min=slider.min, max=slider.max, value=slider.value, step=slider.step,
            continuous_update=slider.continuous_update, layout=slider.layout)
    else:
        raise ValueError(f'cannot copy {slider} object')
    # Overwrite specified attributes
    for k, v in kwargs.items():
        setattr(s, k, v)
    return s


def interactive_display(sim, updatefunc, *refsliders):
    '''
    Start an interactive display
    
    :param sim: simulation object
    :param updatefunc: update function that takes the slider values as input and creates/updates a figure
    :param refsliders: list of reference slider objects
    :return: interactive display
    '''
    # Check that number of input sliders corresponds to update function signature
    params = inspect.signature(updatefunc).parameters
    sparams = [k for k, v in params.items() if v.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD][1:]
    assert len(sparams) == len(refsliders), 'number of sliders does not match update signature'

    # Reset simulation object
    sim.reset()

    # Create a copy of reference sliders for this interactive simulation
    sliders = [copy_slider(rs) for rs in refsliders]
    
    # Call update once to generate initial figure
    fig = updatefunc(sim, *[s.value for s in sliders])
    # Define update wrapper for further figure updates
    update = lambda *args, **kwargs: updatefunc(sim, *args, fig=fig, **kwargs)
    # Create UI and render interactive display 
    ui = VBox(sliders)
    out = interactive_output(update, dict(zip(sparams, sliders)))
    return display(ui, out)
