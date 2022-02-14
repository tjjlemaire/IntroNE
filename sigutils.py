# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-02-11 14:35:21
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-02-14 17:57:24

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
    logger.info(f'sampling rate = {fs * 1e-3:.2f} kHz')

    # Compute dataset size and total time
    nsamples = voltage.size
    ttot = (nsamples - 1) / fs
    logger.info(f'dataset is {time_str(ttot)} s ({nsamples} samples) long')

    # Return as timeseries dataframe
    return pd.DataFrame({
        TIME_S: np.arange(voltage.size) / fs,  # s
        'Vraw': voltage  # uV
    })


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
    btype = 'band'  # by default: band-pass
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


def plot_signals(data, tbounds=None, xlabel=TIME_S, ylabel=PHI_UV, title='signals', keys=None):
    '''
    Plot time-varying signals from a pandas dataframe
    
    :param data: pandas dataframe containing the signals
    :param tbounds (optional): time limits
    :param xlabel: label of the x-axis (time unit)
    :param ylabel: label of the y-axis (signal unit)
    :param keys: keys of the columns containing the signals of interest
    :return: figure handle
    '''
    logger.info('plotting signals...')
    if tbounds is not None:
        data = data[(data[TIME_S] >= tbounds[0]) & (data[TIME_S] <= tbounds[1])] 
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


def plot_frequency_spectrum(data, fs, keys=None, xscale='log', yscale='log', band=None):
    ''' Plot frequency spectrum of a signal

        :param data: pandas dataframe containing the signals
        :return: figure handle
    '''
    if keys is None:
        keys = list(data.columns)
        keys.remove(TIME_S) 
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.despine(ax=ax)
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('PSD (uV2)')
    ax.set_title('signal frequency spectrum')
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    for k in keys:        
        logger.info(f'extracting and plotting {k} frequency sectrum...')
        f, p = welch(data[k].values, fs=fs, nperseg=10000, scaling='spectrum')
        ax.plot(f, p, label=k)
    ax.legend()
    if band is not None:
        ax.axvspan(*band, ec='none', fc='g', alpha=0.2)
    return fig


def get_spikes(y, thr, wlen=80, offset=10, ybounds=None):
    '''
    Extract spikes from a voltage signal
    
    :param y: signal array
    :param wlen: number of signal samples to extract for each spike
    :param thr: absolute threshold for spike detection
    :param offset: an offset expressed in number of samples which shifts the maximum peak from the center
    :param ybounds: lower and upper signal bounds: spikes with data points outside this range are discarded.
    :return: tuple with spikes indexes list and spikes waveforms array
    '''
    # Identify spikes as peaks in signal
    logger.info(f'detecting peaks in signal above {thr:.2f} uV threshold...')
    ispikes, _ = find_peaks(
        y,  # signal
        height=thr,  # minimal peak height
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

    # Filter out spikes with outlier data points
    if ybounds is not None:
        logger.info('discarding spikes outside of {ybounds} interval...')
        spikes = np.array(list(filter(
            lambda x: x.min() > ybounds[0] and x.max() < ybounds[1], spikes)))

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
    spikes = np.array(spikes)
    nout = len(ispikes)
    logger.info(f'filtered out {nin - nout} spikes')
    return ispikes, spikes


def plot_spikes(data, fs, thr=None, clusters=None, n=None, ax=None, ybounds=(-300, 400)):
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
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()
    sns.despine(ax=ax)
    ax.set_xlabel(TIME_MS)
    ax.set_ylabel(PHI_UV)
    if clusters is None:
        ispikes = range(nspikes)
        nspikes = len(ispikes)
        if n is not None:
            ispikes = random.sample(ispikes, n)
            nspikes = f'{n} randomly selected'
        logger.info(f'plotting {nspikes} spike traces...')
        ax.set_title(f'spikes (n = {nspikes})')
        if len(ispikes) > 100:
            ispikes = tqdm(ispikes)
        for i in ispikes:
            ax.plot(tspike, data[i])
    else:
        groups = np.unique(clusters)
        logger.info(f'plotting average spike traces for {len(groups)} clusters...')
        ax.set_title(f'spikes per cluster (n = {len(groups)})')
        for i in groups:
            subdata = data[clusters==i, :]
            cluster_mean = subdata.mean(axis=0)
            cluster_std = subdata.std(axis=0)
            ax.plot(tspike, cluster_mean, label=f'cluster {i}')
            ax.fill_between(
                tspike, cluster_mean - cluster_std, cluster_mean + cluster_std, alpha=0.15) 
        ax.legend()
    if thr is not None:
        ax.axhline(thr, ls='--', c='k')
    ylim = ax.get_ylim()
    ax.set_ylim(max(ylim[0], ybounds[0]), min(ylim[1], ybounds[1]))
    return fig


def apply_pca(data, n_components=10):
    '''
    Normalize dataset and perform PCA on it
    
    :param data: (n_observations, n_samples_per_observartion) data array
    :param n_components: number of components to extract
    :return: (n_observations, n_components) projected data array
    '''
    logger.info(f'applying PCA on {data.shape} dataset...')
    # Apply min-max scaling
    data_scaled = MinMaxScaler().fit_transform(data)
    # Do PCA
    return PCA(n_components=n_components).fit_transform(data_scaled)


def plot_principal_components(data, ax=None, clusters=None):
    '''
    Plot the PC1 against PC2 and use either the PC3 or label for color
    
    :param data: (n_observations, n_components) data array
    :param clusters (optional): (n_observations) cluster index array 
    :return: figure handle
    '''
    logger.info('plotting distribution of first principal components...')
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
    if clusters is None:
        out = ax.scatter(data[:, 0], data[:, 1], c=data[:, 2])
        cbar_ax = fig.colorbar(out)
        cbar_ax.set_label('PC3')
    else:
        for i in np.unique(clusters):
            subdata = data[clusters == i, :]
            ax.scatter(subdata[:, 0], subdata[:, 1], label=f'cluster {i}')
        ax.legend()
    return fig


def apply_kmeans(data, n_clusters=5, **kwargs):
    '''
    Apply k-means clustering
    
    :param data: (n_observations, n_components) data array
    :return: array of cluster index for each observation 
    '''
    logger.info(f'applying k-means clustering with {n_clusters} clusters...')
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, **kwargs).fit(data)
    return kmeans.labels_


def plot_PCs_and_spikes_per_cluster(pca_data, spikes_data, clusters, fs):
    ''' Plot principal componentns and average spike traces per cluster '''
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_principal_components(pca_data, ax=axes[0], clusters=clusters)
    plot_spikes(spikes_data, fs, clusters=clusters, ax=axes[1])
    return fig