# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-02-02 15:58:12
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-04-15 07:31:50

import logging
import numpy as np
from scipy.signal import find_peaks
from IPython.display import display
from ipywidgets import interact, FloatSlider, VBox, interactive_output
from logger import logger
from constants import *
from simulators import *
from stimulus import CurrentPulseTrain
from sigutils import shift


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
    y0 = model.equilibrium(stim)

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


def detect_spikes(data):
    '''
    Detect spikes in simulation output data.

    :param data: simulation results DataFrame
    :return: time indexes of detected spikes

    Example use:
    ispikes = detect_spikes(data)
    '''
    return find_peaks(data[V_MV], height=0., prominence=30.)[0]


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
    data[PHI_UV] = phi_func(data, rmin)
    fig = data.plot_var(PHI_UV, **kwargs)
    logger.setLevel(logging.INFO)
    def update(r=0.0):
        data[PHI_UV] = phi_func(data, r)
        data.plot_var(PHI_UV, ax=fig.axes[0], update=True, redraw=False, **kwargs)
    return interact(update, r=FloatSlider(description='r (um)', min=rmin, max=rmax, step=(rmax - rmin) / 100, continuous_update=False))


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
        
    timestep = np.median(np.diff(data[TIME_MS]))
    data[PHI_UV] = combined_phi_func(
        data[I_NA], rmin, rmin, 0, baseline=baseline)
    fig = data.plot_var(PHI_UV, **kwargs)
    
    def update(r1, r2, dt):
        n = int(np.round(dt / timestep))
        data[PHI_UV] = combined_phi_func(
            data[I_NA], r1, r2, n, baseline=baseline)
        data.plot_var(PHI_UV, ax=fig.axes[0], update=True, redraw=False, **kwargs)
    
    out = interactive_output(update, {'r1': r1, 'r2': r2, 'dt': dt})
    return display(ui, out)


def binary_search(feval, lb, ub, rtol=0.05, atol=1e9, max_nevals=30, ndowns=0, nups=0):
    '''
    Perform a binary search for an evaluatiomn threshold within a defined interval.

    This search is performed as a recursive algorithm, where the search interval is
    progressively refined based on the results of the evaluated function at its mid-point,
    until a convergence criterion is met.
    
    :param feval: evaluation function returning either True or False
    :param lb: lower bound of the search interval
    :param ub: upper bound of the search interval
    :param rtol: relative tolerance w.r.t threshold convergence
    :param atol: relative tolerance w.r.t threshold convergence
    :param max_nevals: maximum number of iterations at which point the recursive algorithm stops
        regardless of convergence status
    :param ndowns: number of times the interval has been refined as lower half interval
    :param nups: number of times the interval has been refined as upper half interval 
    :return: threshold value upon convergence of the algorithm

    Example use:
    thr = binary_search(check_at_current, lower_bound, upper_bound)
    '''
    # Avoid zero values in interval lower bound
    if lb == 0.:
        lb = 1e-6
    # If stop criterion is met
    if (ub - lb) < min(atol, lb * rtol) or (nups + ndowns) > max_nevals:
        # If interval has been continously refined towards an edge, return NaN
        if ndowns == 0 or nups == 0:
            logger.error(f'no "{feval.__name__}" threshold found within specified interval')
            return np.nan
        # Othwerwise, return upper bound
        return ub
    # Determine function evaluation point
    if ub / lb < 2:
        # If interval bounds differ by less than a factor 2, take interval mid-point
        xmid = (lb + ub) / 2
    else:
        # Otherwise, take mid power-of-10 of interval bounds
        xmid = np.power(10., np.log10([lb, ub]).mean())
    if feval(xmid):  # if evaluation is positive, refine to lower half-interval
        bounds = (lb, xmid)
        ndowns += 1
    else:  # otherwise, refine to upper half-interval
        bounds = (xmid, ub)
        nups += 1
    # Recursive call with newly refined interval
    return binary_search(
        feval, *bounds, rtol=rtol, atol=atol, max_nevals=max_nevals, ndowns=ndowns, nups=nups)
