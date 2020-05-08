#Solve a 2D transient Wavefunction
import numpy as np 
import scipy.constants as c
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import axes3d
from LA import *

#Creates dim axes, and also a meshgrid if helpful.
def getGrid(dx: float, L: float):
    N = int(L/dx)
    axes = [np.linspace(0,L,N)]*dim
    grid = np.meshgrid(*axes)

    return axes,grid

# Define a gaussian wavepacket centeres at x0 with a particular wave attached ot it.
def getPsi0(axes,k = None,C:float = 1, x0=None, s0 = 0.5e-2**0.5):
    N = int(len(axes[0]))
    if k == None:
        k = np.zeros(dim)
        k[1] = 40

    if x0 == None:
        x0 = np.zeros(dim)
        x0[0] = 0.5
        x0[1] = 0.2

    psi0 = np.zeros((N,)*dim)*1j

    for i in tqdm(range(len(axes[0]))): #xaxis
        for j in range(len(axes[1])): #yaxis
            #set the wavefunction
            psi0[i][j] = C*np.exp(-((axes[0][i]-x0[0])**2 + (axes[1][j]-x0[1])**2)/(s0**2))*np.exp((k[0]*axes[0][i]+k[1]*axes[1][j])*1j)

    return psi0

# Creates a potential with a really big thing in its center
def getV(axes,r = 0.05, x0 = [0.5,0.5]):
    N = int(len(axes[0]))
    V = np.zeros((N,)*dim)

    for i in range(len(axes[0])):
        for j in range(len(axes[1])):
            x = axes[0][i]
            y = axes[0][j]

            if ((x-x0[0])**2+(y-x0[1])**2)**0.5 <= r:
                V[i][j] = 0*1e4

            # x1 = 0.8
            # x2 = 0.5
            # b = 1-(x2)/(x1-x2)
            # m = 1/(x1-x2)
            # if y>=m*x+b:
            #     V[i][j] = -1e4
    print(V)

    return V

#########################################################
#########################################################
# Simulation functions

# Perform a step of the imaginary wavefuntion
def stepImag(R,I,V,dx,dt,axes):
    Inew = np.zeros(I.shape)
    #Do everything but the boundary
    for i in range(1,len(axes[0])-1):
        for j in range(1,len(axes[1])-1):
            S = R[i+1][j]-2*R[i][j]+R[i-1][j]+\
                R[i][j+1]-2*R[i][j]+R[i][j-1]

            Inew[i][j] = I[i][j] + h*dt/(2*m*dx**2)*S - (1/h)*V[i][j]*dt*R[i][j]

    #Do the boundary
    for i in [0,-1]:
        for j in [0,-1]:
            Inew[i][j] = Inew[3*i+1][3*i+1]
    
    return Inew


# Perform a step of the real wavefuntion
def stepReal(R, I, V, dx, dt, axes):
    Rnew = np.zeros(R.shape)
    #Do everything but the boundary
    for i in range(1, len(axes[0])-1):
        for j in range(1, len(axes[1])-1):
            S = I[i+1][j]-2*I[i][j]+I[i-1][j]+\
                I[i][j+1]-2*I[i][j]+I[i][j-1]

            Rnew[i][j] = R[i][j] - h*dt/(2*m*dx**2)*S + (1/h)*V[i][j]*dt*I[i][j]

    #Do the boundary
    for i in [0, -1]:
        for j in [0, -1]:
            Rnew[i][j] = Rnew[3*i+1][3*i+1]
    
    return Rnew

def step(R,I,V,dx,dt,axes):
    Inew = stepImag(R, I, V, dx, dt, axes)
    Rnew = stepReal(R, Inew, V, dx, dt, axes)

    prob = Rnew**2 + Inew*I

    return Rnew,Inew,prob



#########################################################
#########################################################
# To solve the wavefuncion

# Simulation parameters
dx = 2e-2
dt = 5e-5

dim = int(2)
h = 1
m = 1
L = 1
time = 10

axes, grid = getGrid(dx,L)

# Initial conditions
V = getV(axes)

psi = getPsi0(axes)
R = psi.real
Iprev = psi.imag

# Do half a time step to calculate the leapfrog imaginary. FTCS
I = stepImag(R,Iprev,V,dx,dt/2,axes)
prob = R**2 + Iprev*I

#########################################################
#########################################################
# Plotting and stuffs

#Create figure
fig = plt.figure(figsize=(15,15),dpi = 40)
ax = fig.add_subplot(111,projection='3d')
plotEvery = 1

# Axis specific
ax.set_title("Wavefunction over time\nTimestep: 0")
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")

ax.set_xlim(0,L)
ax.set_ylim(0,L)
ax.set_zlim(0,1)

# wave = plt.pcolormesh(grid[0], grid[1], prob, cmap='gray_r', shading='gouraud')
wireframe = None

def animInit():
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)

    wave.set_array(prob.ravel())

    return wave,

def update(i):
    # ax.set_title("Wavefunction over time\nTimestep: %d"%i)
    for j in range(plotEvery): 
        global R,I,prob
        R,I,prob = step(R,I,V,dx,dt,axes)
    # wave.set_array(prob.ravel())
    # print(i)
    
    global wireframe
    if wireframe:ax.collections.remove(wireframe)
    wireframe = ax.plot_wireframe(grid[0], grid[1], prob, rstride=1, cstride=1, color='k', linewidth=0.5)

    # return wave,


# anim = animation.FuncAnimation(fig, update, init_func=animInit, frames=int(time/dt), interval=1, blit=False)
anim = animation.FuncAnimation(fig, update, frames=500, interval=20, blit=False)
# anim.save("example1.gif", fps=30, writer='imagemagick')

plt.show()
