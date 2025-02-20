# IntroNE

Code material for the interactive recitations and homeworks of the NYU course ***Introduction to Neural Engineering***.

These recitations are based on Jupyter notebooks that can be executed either locally on your laptop (this will require some installations, see instructions below) or online in the following *Binder* environment (does not required any installation):

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tjjlemaire/IntroNE.git/HEAD)

## Local use instructions

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

## Recitations schedule

- **2025.01.30**: the basics of action potential dynamics (`recitation1_AP_basics.ipynb`)
- **2025.02.06**: extracellular action potentials and recordings (`recitation2_ext_recording.ipynb`)
- **2025.02.13**: spike detection and classification (`recitation3_spikesorting.ipynb`) *This tutorial uses a significant amount of RAM, hence execution in the Binder environment will be very slow. It is therefore highly advised to execute the notebook locally.*
- **2025.02.20**: extracellular electrical stimulation (`recitation4_ext_stim.ipynb`) *For Windows users, you will to download and install a [NEURON](https://www.neuron.yale.edu/neuron/download) distribution if you want to run this notebook on your machine.*

## Homeworks schedule: T.B.D.

## Questions

For any questions, you can contact me by email: *theo.lemaire@nyulangone.org*