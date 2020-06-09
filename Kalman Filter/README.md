## Gaussian State Space Model - An exploration into the effectiveness of Kalman Filtration for location detection
[![Image from Gyazo](https://i.gyazo.com/9ab2a6974daa767b98a2bd4825d2afa4.gif)](https://gyazo.com/9ab2a6974daa767b98a2bd4825d2afa4)

### Model Specification

A particle traverses a 2D map according to Newton's Laws of Motion, with random acceleration. We periodically observe noisy measurements of its location, and want to estimate the true location of the particle.

### Findings

Details of my findings can be found in [KalmanFilter.ipynb](https://github.com/DMarke99/Machine-Learning-Gym/blob/master/Kalman%20Filter/KalmanFilter.ipynb). The notebook can be rendered at the [following link](https://nbviewer.jupyter.org/github/DMarke99/Machine-Learning-Gym/blob/master/Kalman%20Filter/KalmanFilter.ipynb).

### Implementation

Code is implemented in Python 3, optimised using Numba and Numpy.
