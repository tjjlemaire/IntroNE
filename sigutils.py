# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-02-11 14:35:21
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-02-16 09:49:08

import numpy as np
import random
from scipy.signal import welch, butter, filtfilt, find_peaks
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from logger import logger
from constants import *


def time_str(seconds, precision=2):
    ''' Compute hour:min:seconds formatted string represeting time interval '''
    sec_fmt = f'.{precision}f'
    # Separate minutes and seconds
    minutes, seconds = seconds // 60, seconds % 60
    # Separate hours and minutes
    hours, minutes = minutes // 60, minutes % 60
    # Return formatted time
    return f'{int(hours):02d}:{int(minutes):02d}:{seconds:{sec_fmt}}'


def load_ncs_data(fpath):
    '''
    Extract data from NCS file
    
    :param fpath: path to NCS data file
    :return: pandas dataframe with time (s) and voltage (uV) series
    '''
    logger.info(f'loading data from "{fpath}"...')
    # Open file
    with open(fpath, 'rb') as fid:
        # Skip header by shifting position by header size
        header_kb = 16
        fid.seek(header_kb * 1024)
        # Read data according to Neuralynx NCS data format:
        # https://neuralynx.com/software/nlx-file-formats
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
    logger.info(f'sampling rate = {fs * 1e-3:.2f} kHz')

    # Compute dataset size and total time
    nsamples = voltage.size
    ttot = (nsamples - 1) / fs
    logger.info(f'dataset is {time_str(ttot)} s ({nsamples} samples) long')

    # Construct and return timeseries dataframe
    data = pd.DataFrame({
        TIME_S: np.arange(voltage.size) / fs,  # s
        'Vraw': -voltage  # correct voltage inversion
    })
    return data


def get_sampling_rate(data):
    ''' Extract the sampling rate from a timeseries dataframe '''
    return 1 / (data.loc[1, TIME_S] - data.loc[0, TIME_S])


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


def filter_signal(y, fs, fc, order=2):
    '''
    Apply filter to signal
    
    :param y: signal array
    :param fs: sampling frequency (Hz)
    :param fc: tuple of cutoff frequencies (Hz)
    :param order: filter order
    '''
    fc = np.asarray(fc)
    # Determine Butterworth type and cutoff
    btype = 'band'
    if fc[0] == 0.:
        btype = 'low'
        fc = fc[1]
    elif fc[1] == np.inf:
        btype = 'high'    
        fc = fc[0]
    logger.info(f'{btype}-pass filtering signal (cutoff = {fc} Hz)...')
    # Determine Nyquist frequency
    nyq = fs / 2
    # Calculate Butterworth filter coefficients
    b, a = butter(order, fc / nyq, btype=btype)
    # Filter signal forward and backward (to ensure zero-phase) and return
    return filtfilt(b, a, y)


def get_spikes(y, thr, wlen=80, offset=10):
    '''
    Extract spikes from a voltage signal
    
    :param y: signal array
    :param wlen: number of signal samples to extract for each spike
    :param thr: absolute threshold for spike detection
    :param offset: an offset expressed in number of samples which shifts the maximum peak from the center
    :return: tuple with spikes indexes list and spikes waveforms array
    '''
    # Identify spikes as peaks in signal
    logger.info(f'detecting peaks in signal beyond {-thr:.2f} uV threshold...')
    ispikes, _ = find_peaks(
        -y,  # reversed signal for positive peak detection
        height=np.abs(thr),  # minimal peak height (positive)
        distance=wlen  # minimal horizontal distance (in samples) between neighbouring peaks
    )
    # Cast window length as integer
    if isinstance(wlen, float):
        wlen = int(np.round(wlen))
    # Compute relative window bounds
    rel_bounds = np.array([-wlen // 2, wlen // 2 - 1]) + offset
    # Gather spikes around each peak
    logger.info(f'extracting spikes from {rel_bounds} samples window around each peak...')
    spikes = np.array([y[slice(*(rel_bounds + i))] for i in ispikes])
    # Return spikes 
    logger.info(f'{len(ispikes)} spikes detected')
    return ispikes, spikes


def filter_spikes(ispikes, spikes, Vbounds):
    '''
    Filter out spikes with outlier data points.

    :param ispikes: list of spikes indexes
    :param spikes: spikes waveform array
    :param Vbounds: voltage interval (spikes extending beyond are discarded)
    :return: tuple with filtered spikes indexes list and spikes waveforms array
    ''' 
    nin = len(ispikes)
    logger.info('discarding spikes outside of {Vbounds} interval...')
    out = list(filter(
        lambda x: x[1].min() > Vbounds[0] and x[1].max() < Vbounds[1],
        zip(ispikes, spikes)))
    ispikes, spikes = list(zip(*out))
    nout = len(ispikes)
    logger.info(f'filtered out {nin - nout} spikes')
    return np.array(ispikes), np.array(spikes)


def pca(data, n_components=10):
    '''
    Normalize dataset and perform PCA on it
    
    :param data: (n_observations, n_samples_per_observartion) data array
    :param n_components: number of components to extract
    :return: (n_observations, n_components) projected data array
    '''
    logger.info(f'applying PCA on {data.shape} dataset...')
    # Apply min-max scaling
    data_scaled = MinMaxScaler().fit_transform(data)
    # Apply PCA and return its output
    return PCA(n_components=n_components).fit_transform(data_scaled)


def kmeans(data, n_clusters=5, **kwargs):
    '''
    Apply k-means clustering
    
    :param data: (n_observations, n_components) data array
    :return: array of cluster index for each observation 
    '''
    logger.info(f'applying k-means clustering with {n_clusters} clusters...')
    out = KMeans(n_clusters=n_clusters, random_state=0, **kwargs).fit(data)
    return out.labels_


def plot_signals(data, tbounds=None, xlabel=TIME_S, ylabel=PHI_UV, title='signals', keys=None,
                 events=None, thr=None):
    '''
    Plot time-varying signals from a pandas dataframe
    
    :param data: pandas dataframe containing the signals
    :param tbounds (optional): time limits
    :param xlabel: (optional) label of the x-axis (time unit)
    :param ylabel (optional): label of the y-axis (signal unit)
    :param keys (optional): keys of the columns containing the signals of interest
    :param events (optional): dictionary of categorized events
    :param thr (optional): absolute spike detection threshold
    :return: figure handle
    '''
    logger.info('plotting signals...')
    # Cast events as dictionray (if provided)
    if events is not None and isinstance(events, np.ndarray):
        events = {'events': events}
    # Restrict data (and events) to specific time interval (if specified)
    if tbounds is not None:
        data = data[(data[TIME_S] >= tbounds[0]) & (data[TIME_S] <= tbounds[1])] 
        if events is not None:
            ibounds = (np.asarray(tbounds) * get_sampling_rate(data)).astype(int)
            events = {k: v[(v >= ibounds[0]) & (v <= ibounds[1])] for k, v in events.items()}
    # Get list of keys to plot (if not provided)
    if keys is None:
        keys = list(data.columns)
        keys.remove(TIME_S)
    elif isinstance(keys, str):
        keys = [keys]
    # Create figure
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    sns.despine(ax=ax)
    # Plot all specified signals
    for k in keys:
        ax.plot(data[TIME_S], data[k], label=k)
    # Plot events per category (if specified)
    if events is not None:
        if len(keys) > 1:
            raise ValueError('cannot plot events on more than 1 signal')
        dy = 0.1 * np.ptp(data[k])
        for event_key, event_indexes in events.items():
            ax.plot(data[TIME_S].values[event_indexes], data[k].values[event_indexes] - dy,
                    '*', label=event_key)
    # Add detection threshold line (if specified)
    if thr is not None:
        ax.axhline(-thr, ls='--', c='k')
    ax.legend()
    # Return figure handle
    return fig


def plot_frequency_spectrum(data, keys=None, xscale='log', yscale='log', band=None):
    ''' Plot frequency spectrum of a signal

        :param data: pandas dataframe containing the signals
        :param keys (optional): keys of the columns containing the signals of interest
        :param xscale (optional): x-axis scale ('lin' or 'log')
        :param yscale (optional): y-axis scale ('lin' or 'log')
        :param band (optional): frequency band to highlight
        :return: figure handle
    '''
    # Get list of keys to plot (if not provided)
    if keys is None:
        keys = list(data.columns)
        keys.remove(TIME_S)
    elif isinstance(keys, str):
        keys = [keys]
    # Create figure
    fig, ax = plt.subplots(figsize=(11, 4))
    sns.despine(ax=ax)
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('PSD (uV2)')
    ax.set_title('signal frequency spectrum')
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    # Compute signals sampling frequency
    fs = get_sampling_rate(data)
    # For each signal
    for k in keys:
        # Compute frequency power spectrum using Welch's method
        logger.info(f'extracting and plotting {k} frequency sectrum...')
        f, p = welch(data[k].values, fs=fs, nperseg=10000, scaling='spectrum')
        # Plot spectrum
        ax.plot(f, p, label=k)
    ax.legend()
    # Plot frequency band of interest (if provided)
    if band is not None:
        ax.axvspan(*band, ec='none', fc='g', alpha=0.2)
    # Return figure handle
    return fig


def plot_spikes(data, fs, thr=None, clusters=None, n=None, ax=None, ybounds=(-400, 300), labels=None):
    '''
    Plot spike waveforms.
    
    :param data: array of spike waveforms
    :param fs: sampling frequency
    :param n (optional): number of (randomly selected) spikes to plot.
        If none, all spikes are plotted.
    :return: figure handle
    '''
    nspikes, nperspike = data.shape
    tspike = np.arange(nperspike) / fs * 1e3  # ms
    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()
    sns.despine(ax=ax)
    ax.set_xlabel(TIME_MS)
    ax.set_ylabel(PHI_UV)
    # If no cluster info is provided -> plot individual spike waveforms
    if clusters is None:
        # Get spike indexes
        ispikes = range(nspikes)
        nspikes = len(ispikes)
        # Restrict spikes to random subset if n provided  
        if n is not None:
            ispikes = random.sample(ispikes, n)
            nspikes = f'{n} randomly selected'
        # Plot spikes
        logger.info(f'plotting {nspikes} spike traces...')
        ax.set_title(f'spikes (n = {nspikes})')
        if len(ispikes) > 100:
            ispikes = tqdm(ispikes)
        for i in ispikes:
            ax.plot(tspike, data[i])
    # If cluster info provided -> plot mean +/- std waveform for each cluster
    else:
        # Get group values
        groups = np.unique(clusters)
        logger.info(f'plotting average spike traces for {len(groups)} clusters...')
        ax.set_title(f'spikes per cluster (n = {len(groups)})')
        # For each group
        for i in groups:
            # Restrict dataset to subgroup, and compute mean and std traces
            subdata = data[clusters==i, :]
            cluster_mean = subdata.mean(axis=0)
            cluster_std = subdata.std(axis=0)
            if labels is not None:
                label = labels[i]
            else:
                label = f'cluster {i}'
            # Plot mean trace with +/- std shaded contour
            ax.plot(tspike, cluster_mean, label=f'{label} (n = {subdata.shape[0]})')
            ax.fill_between(
                tspike, cluster_mean - cluster_std, cluster_mean + cluster_std, alpha=0.15) 
        ax.legend()
    # Add detection threshold line (if specified)
    if thr is not None:
        ax.axhline(-thr, ls='--', c='k')
    # Restrict y range if too large
    ylim = ax.get_ylim()
    ax.set_ylim(max(ylim[0], ybounds[0]), min(ylim[1], ybounds[1]))
    # Return figure handle
    return fig


def plot_principal_components(data, ax=None, clusters=None, labels=None):
    '''
    Plot the PC1 against PC2 and use either the PC3 or label for color
    
    :param data: (n_observations, n_components) data array
    :param clusters (optional): (n_observations) cluster index array 
    :return: figure handle
    '''
    logger.info('plotting distribution of first principal components...')
    # Create figure
    if clusters is None:
        size = (6, 5)
        nPC_plt = 3
    else:
        size = (5, 5)
        nPC_plt = 2
    if ax is None:
        fig, ax = plt.subplots(figsize=size)
    else:
        fig = ax.get_figure()
    ax.set_title(f'first {nPC_plt} PCs')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    # If not cluster info provided -> use PC3 as color code and add colorbar
    if clusters is None:
        out = ax.scatter(data[:, 0], data[:, 1], c=data[:, 2])
        cbar_ax = fig.colorbar(out)
        cbar_ax.set_label('PC3')
    # If culster info provided -> use cluster index as categorical color and add legend 
    else:
        inds = np.unique(clusters)
        for i in inds:
            subdata = data[clusters == i, :]
            if labels is not None:
                label = labels[i]
            else:
                label = f'cluster {i}'
            ax.scatter(subdata[:, 0], subdata[:, 1], label=f'{label} (n = {subdata.shape[0]})')
        ax.legend()
    # Return figure handle
    return fig


def plot_PCs_and_spikes_per_cluster(pca_data, spikes_data, clusters, fs, labels=None, Vthr=None):
    ''' Plot principal componentns and average spike traces per cluster '''
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_principal_components(pca_data, ax=axes[0], clusters=clusters, labels=labels)
    plot_spikes(spikes_data, fs, clusters=clusters, ax=axes[1], thr=Vthr, labels=labels)
    return fig
