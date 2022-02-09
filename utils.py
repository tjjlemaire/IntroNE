# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-02-02 15:58:12
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-02-09 16:13:18

from email.mime import base
import logging
import numpy as np
from IPython.display import display
from ipywidgets import interact, FloatSlider, VBox, interactive_output
from logger import logger
from constants import DT, MV_TO_UV
from simulators import *
from stimulus import CurrentPulseTrain
from model import Model


def initialize_model(*args, **kwargs):
    ''' Interface function to initialize a model. '''
    return Model(*args, **kwargs)


def simulate(model, tstop, stim=None, dt=DT):
    ''' Interface function to simulate model for a specific time,
        and return output data in a dataframe.

        :param model: model object
        :param tstop: simulation stop time (ms)
        :param stim (optional): stimulus object
        :return: output DataFrame
    '''
    s = f'simulating model'
    if stim is not None:
        s = f'{s} with {stim} stimulus'
    logger.info(f'{s} for {tstop:.1f} ms...')

    # Set initial conditions
    y0 = model.equilibrium()

    # Initialize solver
    varkeys = y0.keys()  # variables
    if stim is None:
        solver = Simulator(
            varkeys,                               # variables
            lambda t, y: model.derivatives(t, y),  # derivatives function
            dt=dt)                                 # time step (ms)
    else:
        solver = StimulusDrivenSimulator(
            varkeys,                                                 # variables
            lambda t, y: model.derivatives(t, y, stim=solver.stim),  # derivatives function
            stim=stim,                                               # stimulus object
            dt=dt)                                                   # time step
    
    # Compute solution
    return solver(y0, tstop)


def sim_and_plot(model, I=0, tpulse=30., tstop=60., tstart=5., hard_ylims=False, **kwargs):
    ''' Run simluation with specific pulse amplitude and plot results '''
    stim = CurrentPulseTrain(I=I, tpulse=tpulse, tstart=tstart)
    solution = simulate(model, tstop, stim=stim)
    solution.hard_ylims = hard_ylims
    return solution.plot_all(model.compute_currents, stim=stim, **kwargs)


def interactive_simulation(*args, Imin=-15., Imax=60., **kwargs):
    ''' Run simulation upon slider move and plot results in interactive figure '''
    logger.setLevel(logging.WARNING)
    fig = sim_and_plot(*args, **kwargs)
    logger.setLevel(logging.INFO)
    def update(I=0.0):
        sim_and_plot(*args, **kwargs, I=I, fig=fig)
    return interact(update, I=FloatSlider(min=Imin, max=Imax, step=(Imax - Imin) / 100, continuous_update=False))


def interactive_extracellular(data, phi_func, rmin=10., rmax=300., **kwargs):
    ''' Plot extracellular potential profile as a function of distance to neuron in interactive figure '''
    logger.setLevel(logging.WARNING)
    data['phi (uV)'] = phi_func(data, rmin)
    fig = data.plot_var('phi (uV)', **kwargs)
    logger.setLevel(logging.INFO)
    def update(r=0.0):
        data['phi (uV)'] = phi_func(data, r)
        data.plot_var('phi (uV)', ax=fig.axes[0], update=True, redraw=False, **kwargs)
    return interact(update, r=FloatSlider(description='r (um)', min=rmin, max=rmax, step=(rmax - rmin) / 100, continuous_update=False))


def shift(xs, n):
    ''' Shift a signal by a specific number of samples, while retaining the same signal size '''
    if n == 0:  # n = 0: no change 
        return xs
    e = np.empty_like(xs)
    if n > 0:  # n > 0: right shift
        e[:n] = xs[0]
        e[n:] = xs[:-n]
    else:
        e[n:] = xs[-1]
        e[:n] = xs[-n:]
    return e
    

def interactive_extracellular_2neurons(data, phi_func, baseline=None, rmin=10., rmax=300., dtmin=-2., dtmax=2., **kwargs):
    '''
    Plot extracellular potential profile as a function of distance
    and relative time shift of 2 neurons in interactive figure
    '''
    r1 = FloatSlider(description='r1 (um)', min=rmin, max=rmax, step=(rmax - rmin) / 100, continuous_update=False)
    r2 = FloatSlider(description='r2 (um)', min=rmin, max=rmax, step=(rmax - rmin) / 100, continuous_update=False)
    dt = FloatSlider(description='Δt (ms)', min=dtmin, max=dtmax, step=(dtmax - dtmin) / 100, continuous_update=False)
    ui = VBox([r1, r2, dt])

    def combined_phi_func(I1, r1, r2, n, baseline=None):
        I2 = shift(I1.values, n)
        phi1 = phi_func(I1, r1)
        phi2 = phi_func(I2, r2)
        phi = phi1 + phi2
        if baseline is not None:
            phi += baseline
        return phi
        
    timestep = np.median(np.diff(data[TIME]))
    data['phi (uV)'] = combined_phi_func(
        data['Im (nA)'], rmin, rmin, 0, baseline=baseline)
    fig = data.plot_var('phi (uV)', **kwargs)
    
    def update(r1, r2, dt):
        n = int(np.round(dt / timestep))
        data['phi (uV)'] = combined_phi_func(
            data['Im (nA)'], r1, r2, n, baseline=baseline)
        data.plot_var('phi (uV)', ax=fig.axes[0], update=True, redraw=False, **kwargs)
    
    out = interactive_output(update, {'r1': r1, 'r2': r2, 'dt': dt})
    return display(ui, out)