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

## Recitations schedule

- **2025.01.30**: the action potential dynamics and the Hodgkin-Huxley model (`tuto_HH.ipynb`)
- **2025.02.06**: extracellular action potentials and recordings (`tuto_extracellular_recordings.ipynb`)
- **2025.02.13**: spike detection and classification (`tuto_spike_sorting.ipynb`) *This tutorial uses a significant amount of RAM, hence execution in the Binder environment will be ridiculously slow. It is therefore highly advised to execute the notebook locally.*
- **2025.02.20**: extracellular electrical stimulation (`tuto_extracellular_stim.ipynb`) *For Windows users, you will to download and install a [NEURON](https://www.neuron.yale.edu/neuron/download) distribution in order to run this notebook on your machine.*

## Homeworks schedule: T.B.D.

## Questions

For any questions, you can contact me by email: *theo.lemaire@nyulangone.org*