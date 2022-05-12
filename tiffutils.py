# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-05 17:56:34
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-05-12 12:55:55

''' Tiff loading / viewing / saving utilities. '''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy_image_widget as niw
from ipywidgets import IntSlider, FloatSlider, VBox, HTML
from tifffile import imread, imsave

from logger import logger


def loadtif(fpath, verbose=True):
    ''' Load stack/image from .tif file '''
    stack = imread(fpath)
    if stack.ndim > 2:
        func = logger.info if verbose else logger.debug
        func(f'loaded {stack.shape} {stack.dtype} stack from "{fpath}"')
    return stack


def savetif(fpath, stack, overwrite=True):
    ''' Save stack/image as .tif file '''
    if stack.ndim > 2:
        logger.info(f'saving {stack.shape} {stack.dtype} stack as "{fpath}"...')
    imsave(fpath, stack)



class StackViewer:
    '''
    Simple stack viewer, inspired from Robert Haase's stackview package
    (https://github.com/haesleinhuepf/stackview/).
    '''

    npix_label = 10 # number of pixels used for upper-right labeling
    
    def __init__(self, data, title=None, norm=True, fs=None, cmap='viridis', continuous_update=True, 
                 display_width=240, display_height=240, ilabels=None):
        '''
        Initialization.

        :param data: 3D image stack
        :param title: optional title to render above the image(s)
        :param norm (default = True): whether to normalize the stack data to [0-1] range upon rendering
        :param fs (optional): sampling frequency (frames per second) 
        :param cmap (optional): colormap used to display grayscale image. If none, a gray colormap is used by default.
        :param continuous_update: update the image while dragging the mouse (default: True)
        :param display_width: diplay width (in pixels)
        :param display_height: diplay height (in pixels)
        :param ilabels (optional): array of frame indexes to label.
        '''
        self.data = data
        self.fs = fs
        nframes, *frame_shape = self.data.shape

        # Initialize normalizer
        if norm:
            self.normalize_frame = plt.Normalize(vmin=self.data.min(), vmax=self.data.max())
        else:
            self.normalize_frame = None

        # Initialize colormap
        if cmap is None:
            cmap = 'gray'
        self.cmap = plt.get_cmap(cmap)
        
        # Initialize label arrray
        is_labeled = np.zeros(nframes)
        if ilabels is not None:
            is_labeled[ilabels] = 1.
        self.is_labeled = is_labeled.astype(bool)
        
        # Initialize view
        self.view = niw.NumpyImage(self.transform_frame(np.random.rand(*frame_shape)))
        if display_width is not None:
            self.view.width_display = display_width
        if display_height is not None:
            self.view.height_display = display_height
        
        # Initialize slider
        if self.fs is None:
            self.slider = IntSlider(
                value=0,
                min=0,
                max=nframes - 1,
                continuous_update=continuous_update,
                description='Frame'
            )
        else:
            self.slider = FloatSlider(
                value=0,
                min=0,
                max=(nframes - 1) / fs,
                continuous_update=continuous_update,
                description='Time (s)'
            )
        
        # Connect slider and view
        self.slider.observe(self.update_frame)
        self.update_frame(None)

        # Render stack view
        elems = [self.view, self.slider]
        if title is not None:
            elems = [HTML(value=f'<center>{title}</center>')] + elems
        self.gui = VBox(elems)

    def transform_frame(self, arr):
        ''' Transform a grayscale intensity image to a colored image using a specific colormap '''
        return self.cmap(arr)[:, :, :-1]
    
    def label_frame(self, arr):
        ''' Label a frame by setting pixels on the upper-right corner to red. '''
        arr[:self.npix_label, -self.npix_label:, :] = [arr.max(), 0, 0]
        return arr

    def update_frame(self, event):
        '''
        Event handler: update frame view on slider change.
        
        :param event: event object (not used specifically but needed for the callback to work)
        '''
        # Get frame index
        i = self.slider.value
        if self.fs is not None:
            i = int(i * self.fs)
        arr = self.data[i]
        if self.normalize_frame is not None:
            arr = self.normalize_frame(arr)
        arr = self.transform_frame(arr)
        if self.is_labeled[i]:
            arr = self.label_frame(arr)
        self.view.data = arr


def viewstack(stack, *args, **kwargs):
    ''' View a movie stack interactively in a Jupyter notebook '''
    viewer = StackViewer(stack, *args, **kwargs)
    return viewer.gui


def plot_frame(data, cmap='viridis', title=None, um_per_px=None, peaklocs=None):
    '''
    Plot summary images from a TIF stack.
    
    :param stack: TIF stack
    :param cmap (optional): colormap
    :return: figure handle
    '''
    fig, ax = plt.subplots(figsize=(5, 5))
    if title is not None:
        ax.set_title(title)
    x, y = [np.arange(s) for s in data.shape]
    x = x[::-1]
    label = 'pixels'
    if um_per_px is not None:
        x, y = x * um_per_px, y * um_per_px
        label = 'um'
    ax.set_xlabel(label)
    ax.set_ylabel(label)
    ax.pcolormesh(y, x, data, cmap=cmap)
    ax.set_aspect(1.)
    if peaklocs is not None:
        ixpeaks, iypeaks = peaklocs.T
        xpeaks, ypeaks = x[ixpeaks], y[iypeaks]
        ax.scatter(xpeaks, ypeaks, s=100, fc='none', ec='orange')
    return fig


def load_traces(fpath):
    '''
    Load traces file
    
    :param fpath: path to input file
    :return: 3-tuple with:
        - traces dataframe
        - number of trials
        - number ofr frames per trial
    '''
    dff_traces = pd.read_csv(fpath, index_col=['trial', 'frame'])
    # quick fix: only 3 cells
    dff_traces = dff_traces.iloc[:, :3]
    ntrials = len(dff_traces.index.unique(level='trial'))
    npertrial = len(dff_traces.index.unique(level='frame'))
    ncells = len(dff_traces.columns)
    logger.info(f'loaded {ntrials * npertrial} frames fluorescence traces of {ncells} cells')
    return dff_traces, ntrials, npertrial


def plot_traces(traces, delimiters=None, fs=None):
    '''
    Plot ΔF/F0 traces from a dataframe
    
    :param traces: traces dataframe
    :param delimiters (optional): vector of x axis values at which to draw vertical lines 
    :param fs (optional): sampling frequency (Hz)
    :return: figure handle
    '''
    fig, ax = plt.subplots(figsize=(11, 3))
    sns.despine(ax=ax)
    if fs is None:
        xlabel = 'frames'
        fs = 1
    else:
        xlabel = 'time (s)'
    x = np.arange(len(traces)) / fs
    ax.set_xlabel(xlabel)
    ax.set_ylabel('ΔF/F0 (%)')
    for k in traces:
        ax.plot(x, traces[k] * 100, label=k)
    ax.legend(frameon=False)
    if delimiters is not None:
        for x in delimiters:
            ax.axvline(x, ls='--', c='k')
    return fig
