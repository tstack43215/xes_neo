# XES Neo
#### Versions: 0.0.1
#### Last update: Aug 22, 2023

This program utilizes Genetic algorithm in fitting of x-ray emission spectroscopy.

## Pre-requisites
Usage of this software is highly recommend to use `anaconda` or `pip` package managers.

  - Python: 3.7>
  - Numpy: 1.17.2>
  - Matplotlib: 3.1.2>

It is highly recommend to create a new environment in `anaconda` to prevent packages conflicts.

        conda create --name xes_neo python=3.7 numpy matplotlib pyqt psutil
        conda activate xes_neo


## Installations
To install xes_neo, simply clone the repo:

        git clone https://github.com/tstack43215/xes_neo.git
        cd xes_neo
        pip install .


## Usage
To run a sample test, make sure the environment is set correctly, and select a input file:

        xes_neo -i test/test.ini

## Update
XES Neo is under active development, to update the code to the latest version:

        git pull
        pip install .

## GUI

We also have provided a GUI for use in additions to our program, with additional helper script to facilitate post-analysis. To use the GUI:

        cd gui
        python xes_neo_gui.py

The GUI allows for custom parameters for different indentator during the post-analysis process.

## Potential Errors
If you get an error message involving `psutl`, make sure you are in the right conda environment and install psutil

        conda activate xes_neo
        conda install psutil

## Citation:

TBA
