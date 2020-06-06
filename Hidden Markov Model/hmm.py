import numpy as np
from scipy import special as sp
import random
from numba import njit

# uses maze generation code from @fcogama on GitHub
def maze(width=81, height=51, complexity=1, density =1):
    
    # Only odd shapes
    shape = ((height//2)*2+1, (width//2)*2+1)
    
    # Adjust complexity and density relative to maze size
    complexity = int(complexity*(5*(shape[0]+shape[1])))
    density    = int(density*(shape[0]//2*shape[1]//2))
    
    # Build actual maze
    Z = np.zeros(shape)
    
    # Fill borders
    Z[0,:] = Z[-1,:] = 1
    Z[:,0] = Z[:,-1] = 1
    
    # Make isles
    for i in range(density):
        x, y = np.random.randint(0,shape[1]//2 + 1)*2, np.random.randint(0,shape[0]//2 + 1)*2
        Z[y,x] = 1
        
        for j in range(complexity):
            neighbours = []
            if x > 1:           neighbours.append((y,x-2))
            if x < shape[1]-2:  neighbours.append((y,x+2))
            if y > 1:           neighbours.append((y-2,x))
            if y < shape[0]-2:  neighbours.append((y+2,x))
                
            if len(neighbours):
                y_,x_ = neighbours[np.random.randint(0,len(neighbours))]
                if Z[y_,x_] == 0:
                    Z[y_,x_] = 1
                    Z[y_+(y-y_)//2, x_+(x-x_)//2] = 1
                    x, y = x_, y_
    return Z

transitions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
transitions = [np.array(t) for t in transitions]

# transition matrix for particle
def transition_matrix(maze, jump_prob):
    row, col = maze.shape
    P = np.zeros((row, col, row, col))
    jump_map = 1 - maze
    jump_map = jump_map / jump_map.sum()
    
    for i in range(row):
        for j in range(col):
            if maze[i,j] != 0:
                P[i,j,:,:] = 0
                P[i,j,i,j] = 1
            else:
                pos = (i, j)
                places = [tuple(pos + t) for t in transitions if maze[tuple(pos + t)] != 1]
                for i_ in range(row):
                    for j_ in range(col):
                        P[i,j,i_,j_] = 1.0/len(places) if ((i_, j_) in places) else 0.0
                        
                P[i,j,:,:] = jump_prob * jump_map + (1 - jump_prob) * P[i,j,:,:]
                        
    return P

# emission probability for particle
@njit
def P_xy(pos, rate, recievers, y):
    res = 1.0
    n = recievers.shape[0]
    res = 0
    
    # only needs to be specified up to proportionality
    # computed log likelihood to avoid intermediate overflow errors
    for i in range(n):
        dist = (((recievers[i,:] - pos) ** 2).sum() + 1)
        r = rate/dist
        res = res - r + y[i] * np.log(r)
            
    return np.exp(res)

# calculates P(x_t|y_1:t)
@njit
def alpha_recursion(y, mu, P_xx, rate, recievers):
    T = len(y)
    row, col = mu.shape
    alpha = np.zeros((T, row, col))
    
    for i in range(row):
        for j in range(col):
            alpha[0, i, j] = P_xy(np.array((i, j)), rate, recievers, y[0,:]) * np.sum(P_xx[:,:,i, j] * mu)
        
    alpha[0, :, :] = alpha[0, :, :] / np.sum(alpha[0, :, :])
        
    for t in range(1, T):
        for i in range(row):
            for j in range(col):
                alpha[t, i, j] = P_xy(np.array((i, j)), rate, recievers, y[t,:]) * np.sum(P_xx[:,:,i,j] * alpha[t-1,:])
            
        alpha[t,:,:] = alpha[t,:,:] / np.sum(alpha[t,:,:])
        
    return alpha

# prediction of particle position based on emissions of particle
class random_walk:
    
    def __init__(self, row, col, rate=1000, jump_prob = 0.04):
        
        self.row = 2 * row + 1
        self.col = 2 * col + 1
        self.maze = maze(self.row, self.col, complexity=1, density=1)
        self.P_xx = transition_matrix(self.maze, jump_prob)
        self.rate = rate
        self.jump_prob = jump_prob
        
        # initialises stored information
        self.emissions = []
        self.states = []
        
        # initialises particle and recievers
        self.pos = 2 * np.array((np.random.randint(row//2)+row//4, np.random.randint(col//2)+col//4)) + 1
        self.recievers = np.array([[0, 0], [0, 2*col], [2*row, 0], [2*row, 2*col], [0, col], [row, 2*col], [2*row, col], [row, 0]])
        
        self.maze[tuple(self.pos)] = 3
        
    def generate_maze(self):
        self.maze = maze(self.row, self.col)
        self.P_xx = transition_matrix(self.maze, jump_prob)
        
    def step(self):
        
        # moves into random empty adjacent space
        self.maze[tuple(self.pos)] = 0
        
        if np.random.uniform() < self.jump_prob:
            places = [(i, j) for i in range(self.row) for j in range(self.col) if self.maze[i, j] == 0]
        else:
            places = [tuple(self.pos + t) for t in transitions if self.maze[tuple(self.pos + t)] == 0]
            
        self.pos = random.choice(places)
        
        # particle sends poisson counters to recievers
        dists = self.recievers - self.pos
        dist = ((dists ** 2).sum(axis=1) + 1)
        emissions = []
        
        for d in dist:
            emissions.append(np.random.poisson(self.rate/d))
            
        self.maze[self.pos] = 3
        
        # stores generated data
        self.emissions.append(emissions)
        self.states.append(self.maze.copy())
        
    # predicts position of the particle based on emissions
    def predict(self):
        mu = np.abs(self.maze - 1.0)
        mu = mu/np.sum(mu)
        
        # returns filtered mean positions
        return alpha_recursion(np.array(self.emissions), mu, self.P_xx, self.rate, self.recievers)
    
    # returns all board states
    def get_states(self):
        return np.array(self.states)
    
    # returns argmax (x_t) p(y_t|x_t)
    def mle(self):
        res = []
        emissions = np.array(self.emissions)
        for t in range(len(self.emissions)):
            likelihood = {}
            for i in range(self.row):
                for j in range(self.col):
                    if self.maze[i,j] != 1:
                        likelihood[i,j] = P_xy(np.array([i,j]), self.rate, self.recievers, emissions[t])

            res.append(max(likelihood, key=likelihood.get))
            
        return res
    
    #returns argmax (x_t) p(x_t|y_1:t)
    def predict_loc(self, l, dist):
        res = (0,0)
        for i in range(row):
            for j in range(col):
                if dist[i, j] > dist[res]:
                    res = (i, j)

        return res
    
def predict_loc(walk, dist=None):
    if dist is None:
        dist = walk.predict()
    
    res = []
    for t in range(len(walk.emissions)):
        likelihood = {}
        for i in range(walk.row):
            for j in range(walk.col):
                if walk.maze[i,j] != 1:
                    likelihood[i,j] = dist[t, i, j]

        res.append(max(likelihood, key=likelihood.get))
        
    return res

def accuracy(walk, states):
    return [walk.states[t][states[t]] == 3 for t in range(len(states))]