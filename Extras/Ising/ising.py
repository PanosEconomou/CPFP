############################################
#     ISING MODEL FOR PHASE TRANSITIONS    #
#                                          #
# This is a sketch to simulate phase       #
# transitions uding the ising model and to #
# plot everything an a nice way.           #
#                                          #
# Created by Panos Oikonomou (po524)       #
############################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import scipy.constants as c
from tqdm import tqdm
import matplotlib.animation as animation

def sign(x):
    if x>0: return 1
    if x<0: return -1
    else: return 0

#Class representing a discretised field.
class field:

    def linear(x): return x
    def quadratic(x): return x**2
    def order3(x): return x**3
    def order4(x): return x**4
    def equal(x):
        out = []
        for i in x:
            if not i == 0: out.append(1.)
            else: out.append(0.)
        return np.array(out)
    def root(x): return (abs(x))**0.5
    def fractional(x): return (1/(1-0.1*x)-1)*0.25
    def partitioned(x):
        out = []
        for i in x:
            if i > 0:
                out.append(1.)
            elif i < 0:
                out.append(0.5)
            else:
                out.append(0)
        return np.array(out)

    distfuncs = {"linear": linear,
                "quadratic": quadratic, 
                "cubic": order3, 
                "order4": order4,
                "equal": equal,
                "root":root,
                "fractional":fractional,
                "partitioned":partitioned}

    # Constructor
    def __init__(self,x:float = 10, y:float = 10, dx:float = 0.5, seed:int = 1249834,
                    distString='linear',distfunc=linear):

        self.x    = x       # Length in x-coordinate
        self.y    = y       # Height in y-coordinate
        self.dx   = dx      # Resolution of grid
        self.seed = seed    # Seed for random generation
        self.prob = field.createDistribution(distString,distfunc)
        self.dimx = int(x/dx)
        self.dimy = int(y/dx)


        #Create the field grid with random spins
        np.random.seed(self.seed)
        self.grid = np.random.rand(self.dimy, self.dimx)
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                if self.grid[i][j] < 0.5: self.grid[i][j] = -1
                else: self.grid[i][j] = 1

    
    # This will create a dictionary with the probability distribution
    @staticmethod
    def createDistribution(distString:str = 'linear',f = linear):
        p = np.linspace(-8,8,17)

        #Get the proper names
        if not type(distString) == None:
            try:
                f = field.distfuncs[distString]
            except NameError:
                print("No function exists with identifier:",distString,"\nUsing linear:")
                f = field.linear
        
        prob = f(p)
        prob /= max(prob)
        p = np.linspace(-8, 8, 17)
        dict = {}
        for i in range(len(p)):
            dict.update({p[i]:abs(prob[i])})
        
        return dict

    def step(self):
        # First create a new matrix that has the sum of all of the neighbors at each cell
        # To do that add the relevant submatrices together
        zeroesX = np.array([[0]*len(self.grid[0])])
        zeroesY = np.array([[0]*len(self.grid   )]).T

        # we need to add 8 cells
        l  = np.append(zeroesY,self.grid[:, :-1],axis = 1)
        r  = np.append(self.grid[:, 1:],zeroesY, axis = 1)
        t  = np.append(self.grid[1:, :],zeroesX, axis = 0)
        b  = np.append(zeroesX,self.grid[:-1, :],axis = 0)
        tl = np.append(zeroesY, np.append([zeroesX[0][:-1]], self.grid[:-1, :-1], axis=0), axis=1)
        tr = np.append(np.append([zeroesX[0][:-1]], self.grid[:-1, 1:], axis=0), zeroesY, axis=1)
        bl = np.append(zeroesY, np.append(self.grid[1:, :-1], [zeroesX[0][:-1]], axis=0), axis=1)
        br = np.append(np.append(self.grid[1:, 1:], [zeroesX[0][:-1]], axis=0), zeroesY, axis=1)

        #And then we add the matrices to the grid
        score = l+r+t+b+tl+tr+bl+br

        #Pass the normalisation function:
        for i in range(len(score)):
            for j in range(len(score[i])):
                p = self.prob[score[i][j]]
                self.grid[i][j] = np.random.choice([self.grid[i][j],sign(score[i][j])],p=[1-p,p])
    
    def printDist(self):
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        ax.plot(self.prob.keys(),self.prob.values(),c='k')
        ax.set_xlabel("Sum")
        ax.set_ylabel("Probability")
        plt.show()
        

sim = field(distString='partitioned',seed=945435,dx=0.2)

# Here are commands for pltting the animation.
fig = plt.figure(figsize=(10, 10), dpi=70)
ax = fig.add_subplot(111)
ax.axis('off')
fig.suptitle("Timestamp: 0")
plot = ax.matshow(sim.grid, cmap=plt.get_cmap('winter'))

def animInit():
    plot.set_array(sim.grid)
    plt.draw()
    return plot,

def update(i):
    fig.suptitle("Iteration: "+str(i))
    # print(i)
    # print(sim.grid)
    sim.step()
    plot.set_array(sim.grid)
    # ax.draw()

    return plot,


# Set up formatting for the movie files
anim = animation.FuncAnimation(fig,update,frames = 1000,interval = 10,init_func=animInit,repeat=False,blit=True)

# Uncomment to save animation as .gif
# anim.save("out.gif", fps=30, writer='imagemagick')
plt.show()
print("Done")



