# IntroNE

Code material for the interactive "recitations" of the NYU course "Introduction to Neural Engineering"

## Local use

### Installation

- Download and install a Python distribution from https://www.anaconda.com/download/ using the Anaconda installer
- Open the Anaconda prompt
- Clone this repository: `git clone https://github.com/tjjlemaire/IntroNE.git`. To do this you will need to have Git installed on your computer (you can download it from https://git-scm.com/downloads). Alternatively, you can download the code archive and unzip it.
- Move to the code folder: `cd IntroNE`
- Create a new anaconda environment: `conda create -n introne python=3.8`
- Activate the anaconda environment `conda activate introne`
- Install the code dependencies: `pip install -r requirements.txt`
- You're all set!

### Running a notebook

To execute a notebook locally:
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

You can you Binder to access and run the notebooks online without a local installation.

Here's the link to the repo: https://mybinder.org/v2/gh/tjjlemaire/IntroNE.git/HEAD

## Session 1 (2022.02.03) - tutorial on the action potential dynamics and Hodgkin-Huxley model.

- Notebook name: `tuto_HH.ipynb`
- Binder link: https://mybinder.org/v2/gh/tjjlemaire/IntroNE/1edb281a439b44561ac31d38ddec9c5ae5996e2c?urlpath=lab%2Ftree%2Ftuto_HH.ipynb

## Session 2 (2022.02.10) - tutorial on extracellular action potentials and recordings.

- Notebook name: `tuto_extracellular_recordings.ipynb`
- Binder link: https://mybinder.org/v2/gh/tjjlemaire/IntroNE/686ad727554944af541099cf054bb1b8bc0d6fa7?urlpath=lab%2Ftree%2Ftuto_extracellular_recording.ipynb

## Session 3 (2022.02.17) - tutorial on spike detection and classification.

- Notebook name: `tuto_spike_sorting.ipynb`
- Binder link: https://mybinder.org/v2/gh/tjjlemaire/IntroNE/54782e8179eb3927d80360c6bf8a53aa30f92358?urlpath=lab%2Ftree%2Ftuto_spike_sorting.ipynb

## Questions

For any questions, you can contact me by email: theo.lemaire@nyulangone.org