## Hidden Markov Models - An exploration into the effectiveness of HMMs for location detection
[![Preview of Location Estimation](https://i.gyazo.com/ce90b5119f19df33d90261bf0af36b74.gif)](https://gyazo.com/ce90b5119f19df33d90261bf0af36b74)
*Figure 1: Location of the particle (right) and a heatmap of the predicted location of the particle (left)*

### Model Specification

A particle performing a random walk on a maze. At every timestep it moves into a uniformly selected adjacent empty tile. It also has a small probability to jump to a random tile in the maze. After each step, recievers at fixed locations recieve independent poisson counts with rates dependent on their distance from the particle. We observe these counts, and want to predict the location of the particle.

### Findings

Details of my findings can be found in [HiddenMarkovModels.ipynb](https://github.com/DMarke99/Machine-Learning-Gym/blob/master/Hidden%20Markov%20Model/HiddenMarkovModels.ipynb), with the implementation of the model found in [hmm.py](https://github.com/DMarke99/Machine-Learning-Gym/blob/master/Hidden%20Markov%20Model/hmm.py). The notebook can be rendered at the [following link](https://nbviewer.jupyter.org/github/DMarke99/Machine-Learning-Gym/blob/master/Hidden%20Markov%20Model/HiddenMarkovModels.ipynb).

### Implementation

Code is implemented in Python 3, optimised using Numba and Numpy.
