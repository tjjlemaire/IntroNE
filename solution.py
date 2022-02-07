# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-02-01 18:50:11
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-02-07 17:08:22

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import Model

from logger import *
from constants import *


class Solution(pd.DataFrame):
    ''' Wrapper around pandas DataFrame containing a simulation solution '''

    def __init__(self, data, hard_ylims=False):
        super().__init__(data)
        self.hard_ylims = hard_ylims

    @property
    def states(self):
        ''' Automatically extract state names from solution dataset '''
        return list(set(self.columns) - set((TIME, VOLTAGE)))

    def plot(self, *args, **kwargs):
        ''' Wrapper around pandas plot, using time as x variable '''
        return super().plot(x=TIME)

    @staticmethod
    def set_soft_ylims(ax, my_ylims):
        ''' Set a minimum y-axis range, but expand it if data exceeds that range '''
        ax.autoscale(True)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(min(my_ylims[0], ymin), max(my_ylims[1], ymax))

    def set_ylims(self, ax, my_ylims):
        ''' Set y-axis limits '''
        if self.hard_ylims:
            ax.set_ylim(*my_ylims)
        else:
            self.set_soft_ylims(ax, my_ylims)

    @staticmethod
    def update_axis(ax):
        ''' Update axis scaling '''
        ax.relim()
        ax.autoscale_view()

    @staticmethod
    def add_stim_mark(stim, ax):
        ''' Add stimulus marks on plots '''
        t_off_on, t_on_off = stim.t_OFF_ON(), stim.t_ON_OFF()
        for t1, t2 in zip(t_off_on, t_on_off):
            ax.axvspan(t1, t2, color='silver', alpha=0.3)

    def plot_var(self, key, ax=None, stim=None, update=False, redraw=True, ylims=None):
        ''' plot variable time course '''
        if ax is None:
            fig, ax = plt.subplots()
            sns.despine(ax=ax)
            ax.set_xlabel(TIME)
        else:
            fig = ax.get_figure()
        if update:
            line = ax.get_lines()[0]
            line.set_ydata(self[key])
            self.update_axis(ax)
        else:
            ax.set_ylabel(key)
            ax.plot(self[TIME], self[key], c='k')
            if stim is not None:
                self.add_stim_mark(stim, ax)
        if ylims is not None:
            self.set_ylims(ax, ylims)
        if update and redraw:
            fig.canvas.draw()
        return fig

    def plot_voltage(self, *args, **kwargs):
        ''' plot solution voltage time course '''
        return self.plot_var(VOLTAGE, *args, ylims=V_LIMS, **kwargs)

    def plot_states(self, ax=None, stim=None, update=False, redraw=True):
        ''' plot solution states time course '''
        if ax is None:
            fig, ax = plt.subplots()
            sns.despine(ax=ax)
            ax.set_xlabel(TIME)
        else:
            fig = ax.get_figure()
        if update:
            for k, line in zip(self.states, ax.get_lines()):
                line.set_ydata(self[k])
                self.update_axis(ax)
        else:
            ax.set_ylabel('states')
            for k in self.states:
                ax.plot(self[TIME], self[k], label=k)
            ax.legend()
            if stim is not None:
                self.add_stim_mark(stim, ax)
        self.set_ylims(ax, STATES_LIMS)
        if update and redraw:
            fig.canvas.draw()
        return fig    

    def plot_currents(self, cfuncs, stim=None, ax=None, update=False, redraw=True):
        ''' plot solution currents time course '''
        if ax is None:
            fig, ax = plt.subplots()
            sns.despine(ax=ax)
            ax.set_xlabel(TIME)
        else:
            fig = ax.get_figure()
        if isinstance(cfuncs, Model):
            cfuncs = cfuncs.compute_currents
        currents = cfuncs(self[VOLTAGE], self)
        if currents:
            i_cap = -pd.concat(currents.values(), axis=1).sum(axis=1)
        else:
            i_cap = 0
        if stim is not None:
            tstim, Istim = stim.stim_profile(tstop=self[TIME].values[-1])
            Istim_interp = np.interp(self[TIME], tstim, Istim)
            i_cap += Istim_interp
        currents.update({'i_cap': i_cap})
        if update:
            for v, line in zip(currents.values(), ax.get_lines()):
                line.set_ydata(v)
        else:
            ax.set_ylabel(CURRENT)
            colors = plt.get_cmap('Dark2').colors
            for (k, v), c in zip(currents.items(), colors):
                ax.plot(self[TIME], v, label=k, c=c)
        if stim is not None:
            if update:
                ax.get_lines()[-1].set_ydata(Istim)
            else:
                ax.plot(tstim, Istim, label='i_stim', c='k')
                self.add_stim_mark(stim, ax)
        if update:
            self.update_axis(ax)
        else:
            ax.legend()
        self.set_ylims(ax, I_LIMS)
        if update and redraw:
            fig.canvas.draw()
        return fig
    
    def plot_all(self, cfuncs, fig=None, **kwargs):
        ''' plot all time courses (voltage, states & currents) from a solution '''
        naxes = 3 if self.states else 2
        if fig is None:
            fig, axes = plt.subplots(naxes, 1, figsize=(7, 2 * naxes))
            update = False
        else:
            axes = fig.axes
            update = True
        iax = 0
        self.plot_voltage(ax=axes[iax], update=update, redraw=False, **kwargs)
        iax += 1
        if self.states:
            self.plot_states(ax=axes[iax], update=update, redraw=False, **kwargs)
            iax += 1
        self.plot_currents(cfuncs, ax=axes[iax], update=update, redraw=False, **kwargs)
        if not update:
            for ax in axes:
                sns.despine(ax=ax)
            for ax in axes[:-1]:
                ax.set_xticks([])
            axes[-1].set_xlabel(TIME)
        else:
            fig.canvas.draw()
        return fig

