import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

dt=0.1
dx=0.1

L=50                        # length of the plate
B=50                        # width of the plate


#heating device shaped like X
Gr=np.eye(10)*2000
for iGr in range(10):
    Gr[iGr,-iGr-1]=2000

# Function to set M values corresponding to non-zero Gr values
def assert_heaters(M, Gr):
    M[20:30,10:20] = np.where(Gr > 0, Gr, M[20:30,10:20])
    M[20:30,30:40] = np.where(Gr > 0, Gr, M[20:30,30:40])


M=np.zeros([L,B])           # matrix
assert_heaters(M, Gr)

# Build MM, a list of matrices, each element corresponding to M at a given step
T = np.arange(0,10,dt)
MM = []
for i in range(len(T)):
    for j in range(1,L-1):
        for i in range(1,B-1):
            k=0.5  # default k
            if 24<j<28:
                # holes for liquid
                if 29<i<32 or 23<i<20: k=0

            #dm = k * ((dt)/dx**2) * (M[i,j+1] + M[i,j-1] - 2*M[i,j]) + \
            #     k * ((dt)/dx**2) * (M[i+1,j] + M[i-1,j] - 2*M[i,j])
            #M[i,j] += dm
            M[i,j] = (M[i-1,j] + M[i+1,j] + M[i,j-1] + M[i,j+1])/4

    # Re-assert heaters
    assert_heaters(M, Gr)

    MM.append(M.copy())



fig = plt.figure()
pcm = plt.pcolormesh(MM[0])
plt.colorbar()

# Function called to update the graphic
def step(i):
    if i >= len(MM): return
    pcm.set_array(MM[i].ravel())
    plt.draw()

anim = FuncAnimation(fig, step, interval=50)
plt.show()