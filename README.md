# IntroNE

Code material for the interactive recitations and homeworks of the NYU course ***Introduction to Neural Engineering***.

These recitations are based on Jupyter notebooks that can be either run locally on your laptop, or executed in an online [Binder](https://mybinder.org) environment.

## Local use

### Installation

- Download and install a Python distribution from https://www.anaconda.com/download/ using the Anaconda installer
- Open the Anaconda prompt
- Clone this repository: `git clone https://github.com/tjjlemaire/IntroNE.git`. To do this you will need to have [Git](https://git-scm.com/downloads) installed on your computer. Alternatively, you can download the repo archive and unzip it (although the Git way is highly advised).
- Move to the repo folder: `cd IntroNE`
- Create a new anaconda environment: `conda create -n introne python=3.8`
- Activate the anaconda environment `conda activate introne`
- Install the code dependencies: `pip install -r requirements.txt`
- You're all set!

### Notebook execution

- Open an anaconda prompt
- Activate the anaconda environment: `conda activate introne`
- Move to this repository
- Download the latest updates from the online repository: `git pull`
- Install the code dependencies: `pip install -r requirements.txt`
- Start a jupyter lab session: `jupyter lab`
- Open the notebook of interest by double-clicking on it
- In the upper right corner, make sure that jupyter is running with the right kernel (`Python 3.8.12 64-bit ('introne': conda)`). If not, click and select the appropriate kernel from the drop-down list.
- You're all set!

## Online use

You can use *Binder* to access and run the notebooks online without a local installation:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tjjlemaire/IntroNE.git/HEAD)

## Recitation 1 (2022.02.03) - the action potential dynamics and the Hodgkin-Huxley model.

Notebook name: `tuto_HH.ipynb`
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tjjlemaire/IntroNE/1edb281a439b44561ac31d38ddec9c5ae5996e2c?urlpath=lab%2Ftree%2Ftuto_HH.ipynb)

## Recitation 2 (2022.02.10) - extracellular action potentials and recordings.

Notebook name: `tuto_extracellular_recordings.ipynb`
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tjjlemaire/IntroNE/686ad727554944af541099cf054bb1b8bc0d6fa7?urlpath=lab%2Ftree%2Ftuto_extracellular_recording.ipynb)

## Recitation 3 (2022.02.17) - tutorial on spike detection and classification.

Notebook name: `tuto_spike_sorting.ipynb`
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tjjlemaire/IntroNE/67473903c5b828e05e09f08683c38ca6a80a1ff8?urlpath=lab%2Ftree%2Ftuto_spike_sorting.ipynb)

**Note: This tutorial uses a significant amount of RAM, hence execution in the Binder environment will be ridiculously slow. It is therefore highly advised to execute the notebook locally.**

## Recitation 4 (2022.02.24) - tutorial on extracellular electrical stimulation.

Notebook name: `tuto_extracellular_stim.ipynb`
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tjjlemaire/IntroNE/5002b0c8e27b8d4f5e673d4f73bceca946455d37?urlpath=lab%2Ftree%2Ftuto_extracellular_stim.ipynb)

***Note***: for Windows users, you will to download and install a [NEURON](https://www.neuron.yale.edu/neuron/download) distribution in order to run this notebook on your machine.

## Homework 1 (2022.03.07)

Notebook name: `homework1.ipynb`
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tjjlemaire/IntroNE/38702e0294a07db8ace0b77094e5cea1feee7fdf?urlpath=lab%2Ftree%2Fhomework1.ipynb)

***Note***: for Windows users, you will to download and install a [NEURON](https://www.neuron.yale.edu/neuron/download) distribution in order to run this notebook on your machine.

## Homework 2 (2022.04.15)

Notebook name: `hw2.ipynb`
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tjjlemaire/IntroNE/92a7eef4d2888c425831b6949342c273b5a88807?urlpath=lab%2Ftree%2Fhw2.ipynb)

## Homework 3 (2022.05.12)

Notebook name: `hw3.ipynb`
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tjjlemaire/IntroNE/9a591d79189f80e412e5cbe1203fbbadf0e7746f?urlpath=lab%2Ftree%2Fhw3.ipynb)

**This notebook requires some input data to run.**

## Questions

For any questions, you can contact me by email: *theo.lemaire@nyulangone.org*