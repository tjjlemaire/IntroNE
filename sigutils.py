# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-02-11 14:35:21
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-02-11 17:25:44

import numpy as np
import random
from scipy.signal import butter, filtfilt, find_peaks
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from constants import *


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


def bandpass_filter(y, fs, bounds, order=2):
    '''
    Apply band-pass filter to signal
    
    :param y: signal array
    :param fs: sampling frequency (Hz)
    :param bounds: bounds of band-pass
    :param order: filter order
    '''
    # Determine Nyquist frequency
    nyq = fs / 2
    # Calculate coefficients
    b, a = butter(order, np.asarray(bounds) / nyq, btype='band')
    # Filter signal forward and backward (to ensure zero-phase) and return
    return filtfilt(b, a, y)


def load_ncs_data(fpath):
    '''
    Extract data from NCS file
    
    :param fpath: path to NCS data file
    :return: pandas dataframe with time (s) and voltage (uV) series
    '''
    # Open file
    with open(fpath, 'rb') as fid:
        # Skip header by shifting position by header size
        header_kb = 16
        fid.seek(header_kb * 1024)
        # Read data according to Neuralynx NCS data format:
        # https://support.neuralynx.com/hc/en-us/articles/360040444811-TechTip-Neuralynx-Data-File-Formats
        ncs_fmt = np.dtype([
            ('TimeStamp', np.uint64),
            ('ChannelNumber', np.uint32),
            ('SampleFreq', np.uint32),
            ('NumValidSamples', np.uint32),
            ('Samples', np.int16, 512)
        ])
        # Extract data from file
        data = np.fromfile(fid, dtype=ncs_fmt)
    
    # Extract sampling frequency and voltage signal
    fs = data['SampleFreq'][0]  # Hz
    voltage = data['Samples'].ravel()  # uV

    # Return as timeseries dataframe
    return pd.DataFrame({
        TIME_S: np.arange(voltage.size) / fs,  # s
        'Vraw': voltage  # uV
    })


def plot_signals(data, xlabel=TIME_S, ylabel=PHI_UV, title='signals', keys=None):
    '''
    Plot time-varying signals from a pandas dataframe
    
    :param data: pandas dataframe containing the signals
    :param xlabel: label of the x-axis (time unit)
    :param ylabel: label of the y-axis (signal unit)
    :param keys: keys of the columns containing the signals of interest
    :return: figure handle
    '''
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    sns.despine(ax=ax)
    if keys is None:
        keys = list(data.columns)
        keys.remove(TIME_S)
    for k in keys:
        ax.plot(data[TIME_S], data[k], label=k)
    ax.legend()
    return fig


def get_spikes(y, wlen=80, thr_factor=5, offset=10, ybounds=(-150., 300.)):
    '''
    Extract spikes from a voltage signal
    
    :param y: signal array
    :param wlen: number of signal samples to extract for each spike
    :param thr_factor: threshold factor for spike detection (thr = mean(signal) * tf)
    :param offset: an offset expressed in number of samples which shifts the maximum peak from the center
    :param ybounds: lower and upper signal bounds: spikes with data points outside this range are discarded.
    :return: ???
    '''
    # Cast window length as integer
    if isinstance(wlen, float):
        wlen = int(np.round(wlen))
    # Compute relative window bounds
    rel_bounds = np.array([-wlen // 2, wlen // 2 - 1]) + offset

    # Calculate absolute threshold based on signal mean
    thr = np.mean(np.abs(y)) * thr_factor

    # Identify spikes as peaks in signal
    ispikes, _ = find_peaks(
        y,  # signal
        height=thr,  # minimal peak height
        distance=wlen  # minimal horizontal distance (in samples) between neighbouring peaks
    )

    # Gather spikes around each peak
    spikes = np.array([y[slice(*(rel_bounds + i))] for i in ispikes])

    # Filter out spikes with outlier data points
    if ybounds is not None:
        spikes = np.array(list(filter(
            lambda x: x.min() > ybounds[0] and x.max() < ybounds[1], spikes)))

    # Return spikes 
    return ispikes, spikes


def plot_spikes(data, fs, n=None):
    '''
    Plot spike traces
    
    :param data: array of spikes
    :param fs: sampling frequency
    :param n (optional): number of (randomly selected) spikes to plot.
        If none, all spikes are plotted.
    :return: figure handle
    '''
    nspikes, nperspike = data.shape
    tspike = np.arange(nperspike) / fs * 1e3  # ms
    ispikes = range(nspikes)
    if n is not None:
        ispikes = random.sample(ispikes, n)
    fig, ax = plt.subplots(figsize=(7, 3))
    sns.despine(ax=ax)
    ax.set_title(f'spikes (n = {len(ispikes)})')
    ax.set_xlabel(TIME_MS)
    ax.set_ylabel(PHI_UV)
    if len(ispikes) > 100:
        ispikes = tqdm(ispikes)
    for i in ispikes:
        ax.plot(tspike, data[i])
    return fig
