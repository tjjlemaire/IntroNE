# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-02-02 15:58:12
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-02-07 17:08:58

import logging
import numpy as np
from ipywidgets import interact, FloatSlider
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


def interactive_extracellular(data, phi_func, rmin=10., rmax=1000., **kwargs):
    ''' Plot extracellular potential profile as a function of distance to neuron in interactive figure '''
    logger.setLevel(logging.WARNING)
    data['phi (uV)'] = phi_func(data, rmin)
    fig = data.plot_var('phi (uV)', **kwargs)
    logger.setLevel(logging.INFO)
    def update(r=0.0):
        data['phi (uV)'] = phi_func(data, r)
        data.plot_var('phi (uV)', ax=fig.axes[0], update=True, redraw=False, **kwargs)
    return interact(update, r=FloatSlider(min=rmin, max=rmax, step=(rmax - rmin) / 100, continuous_update=False))
