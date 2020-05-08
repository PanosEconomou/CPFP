#Solve a 3D transient Wavefunction
import numpy as np
import scipy.constants as c
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import axes3d
from LA import *

def mag(x):
    return x.dot(x)**0.5

#Creates dim axes, and also a meshgrid if helpful.
def getGrid(dx: float, L: float=1, Lx=None,Ly=None,Lz=None):
    if type(Lx) == None or type(Ly) == None or type(Lz) == None:
        N = int(L/dx)
        axes = [np.linspace(0, L, N)]*dim
    else:
        L = [Lx,Ly,Lz]
        N = [int(l/dx) for l in L]
        axes = [np.linspace(0,L[i],N[i]) for i in range(dim)]
    
    grid = np.meshgrid(*axes)
    print(axes[0])
    print(axes[1])
    print(axes[2])
    return axes, grid

# Define a gaussian wavepacket centeres at x0 with a particular wave attached ot it
def getPsi0(axes, K=None, C: float = 1, x0=None, s0=0.5e-2**0.5):
    Nx = int(len(axes[0]))
    Ny = int(len(axes[1]))
    Nz = int(len(axes[2]))

    if K == None:
        K = np.zeros(dim)
        K[1] = 40

    if x0 == None:
        x0 = np.zeros(dim)
        x0[0] = 0.2
        x0[1] = 1
        x0[2] = 1

    psi0 = np.zeros((Nx,Ny,Nz))*1j

    for i in tqdm(range(len(axes[0]))):  # xaxis
        for j in range(len(axes[1])):  # yaxis
            for k in range(len(axes[2])): # z axis
                #set the wavefunction
                sum =   (axes[0][i]-x0[0])**2 +\
                        (axes[1][j]-x0[1])**2 +\
                        (axes[2][k]-x0[2])**2
                
                kf  =   K[0]*axes[0][i]+\
                        K[1]*axes[1][j]+\
                        K[2]*axes[2][k]

                psi0[i][j][k] = C*np.exp(-(sum/(s0**2))*np.exp(kf*1j))

    return psi0

# Creates a potential with a really big thing in its center


def getV(axes, r=0.1, x0=np.array([0.5, 0.5,0.5])):
    Nx=int(len(axes[0]))
    Ny=int(len(axes[1]))
    Nz=int(len(axes[2]))

    V = np.zeros((Nx,Ny,Nz))

    for i in tqdm(range(len(axes[0]))):
        for j in range(len(axes[1])):
            for k in range(len(axes[2])):
                x = axes[0][i]
                y = axes[1][j]
                z = axes[2][k]
                
                R = np.array([x,y,z])
                
                if (mag(R-x0))**0.5 <= r:
                    V[i][j][k] = 0#1e4

                # x1 = 0.8
                # x2 = 0.5
                # b = 1-(x2)/(x1-x2)
                # m = 1/(x1-x2)
                # if y >= m*x+b:
            #     V[i][j] = -1e4
    # print(V)

    return V

#########################################################
#########################################################
# Simulation functions

# Perform a step of the imaginary wavefuntion


def stepImag(R, I, V, dx, dt, axes):
    Inew = np.zeros(I.shape)
    #Do everything but the boundary
    for i in range(1, len(axes[0])-1):
        for j in range(1, len(axes[1])-1):
            for k in range(1,len(axes[2])-1):
                S = R[i+1][j][k]-2*R[i][j][k]+R[i-1][j][k] +\
                    R[i][j+1][k]-2*R[i][j][k]+R[i][j-1][k] +\
                    R[i][j][k+1]-2*R[i][j][k]+R[i][j][k-1]

                if j < 0.5*len(axes[1]):
                    B = 15
                else:
                    B = 0
                Inew[i][j][k] = I[i][j][k] + h*dt /(2*m*dx**2)*S +\
                    q**2*dt/(2*m*h)*B**2*(j*dx)**2 -\
                    (q/h)*V[i][j][k]*dt*R[i][j][k]

    # #Do the boundary
    # for i in [0, -1]:
    #     for j in [0, -1]:
    #         Inew[i][j] = Inew[3*i+1][3*i+1]

    return Inew


# Perform a step of the real wavefuntion
def stepReal(R, I, V, dx, dt, axes):
    Rnew = np.zeros(R.shape)
    #Do everything but the boundary
    for i in range(1, len(axes[0])-1):
        for j in range(1, len(axes[1])-1):
            for k in range(1, len(axes[2])-1):
                S = I[i+1][j][k]-2*I[i][j][k]+I[i-1][j][k] +\
                    I[i][j+1][k]-2*I[i][j][k]+I[i][j-1][k] +\
                    I[i][j][k+1]-2*I[i][j][k]+I[i][j][k-1]

                if j < 0.5*len(axes[1]):
                    B = 15
                else:
                    B = 0
                Rnew[i][j][k] = R[i][j][k] - h*dt / (2*m*dx**2)*S -\
                    q**2*dt/(2*m*h)*B**2*(j*dx)**2+\
                    (q/h) * V[i][j][k]*dt*I[i][j][k]

    # #Do the boundary
    # for i in [0, -1]:
    #     for j in [0, -1]:
    #         Rnew[i][j] = Rnew[3*i+1][3*i+1]

    return Rnew


def step(R, I, V, dx, dt, axes):
    Inew = stepImag(R, I, V, dx, dt, axes)
    Rnew = stepReal(R, Inew, V, dx, dt, axes)

    prob = Rnew**2 + Inew*I

    return Rnew, Inew, prob


#########################################################
#########################################################
# To solve the wavefuncion
# Simulation parameters
dx = 5e-2
dt = 5e-5

dim = int(3)
h = 1
m = 1
L = 1
q = 1
time = 10

axes, grid = getGrid(dx, Lx=1,Ly=1,Lz=1)

# Initial conditions
V = getV(axes)

psi = getPsi0(axes)
R = psi.real
Iprev = psi.imag

# Do half a time step to calculate the leapfrog imaginary. FTCS
I = stepImag(R, Iprev, V, dx, dt/2, axes)
prob = R**2 + Iprev*I

#########################################################
#########################################################
# Plotting and stuffs

#Create figure
fig = plt.figure(figsize=(15, 15), dpi=40)
ax = fig.add_subplot(111, projection='3d')
plotEvery = 10

# Axis specific
ax.set_title("Wavefunction over time\nTimestep: 0")
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")

ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_zlim(0, 1)

# wave = plt.pcolormesh(grid[0], grid[1], prob, cmap='gray_r', shading='gouraud')

cmap=plt.cm.gray_r#seismic  # winter#gray_r
# colors = cmap(psi.T[0] / max(psi.T[0]))
# colors[:, -1] = psi.T[0] / max(psi.T[0])

wave = ax.scatter([],[],[],s=10)
x = axes[0]
y = axes[1]
z = axes[2]

def animInit():
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)

    p = []
    x = []
    y = []
    z = []
    P = prob.max()
    for i in range(len(prob)):
        for j in range(len(prob[i])):
            for k in range(len(prob[i][j])):
                if prob[i][j][k]/P > 0.01:
                    p.append(prob[i][j][k])
                    x.append(axes[0][i])
                    y.append(axes[1][j])
                    z.append(axes[2][k])
    p = np.array(p)/P
    # print(x,y,z,p)
    
    colors=cmap(p)
    colors[:, -1]=p

    wave._offsets3d = (x,y,z)
    # wave.set_array(p)

    return wave,


def update(i):
    ax.set_title("Wavefunction over time\nTimestep: %d"%i)
    for j in range(plotEvery):
        global R, I, prob
        R, I, prob = step(R, I, V, dx, dt, axes)
    # wave.set_array(prob.ravel())
    # print(i)

    # global wireframe
    # if wireframe:
    #     ax.collections.remove(wireframe)
    # wireframe = ax.plot_wireframe(grid[0], grid[1], grid[2], rstride=1, cstride=1, color='k', linewidth=0.5)

    p=[]
    x=[]
    y=[]
    z=[]
    P = prob.max()
    for i in range(len(prob)):
        for j in range(len(prob[i])):
            for k in range(len(prob[i][j])):
                if prob[i][j][k]/P > 0.001:
                    p.append(prob[i][j][k])
                    x.append(axes[0][i])
                    y.append(axes[1][j])
                    z.append(axes[2][k])
    p = np.array(p)/P

    colors=cmap(p)
    colors[:, -1]=p

    wave._offsets3d = (x, y, z)
    # wave.set_array(p+1)
    plt.draw()
    
    return wave,


anim = animation.FuncAnimation(fig, update, init_func=animInit, frames=int(time/dt), interval=1, blit=False)
# anim = animation.FuncAnimation(fig, update, frames=500, interval=20, blit=False)
# anim.save("example1.gif", fps=30, writer='imagemagick')

plt.show()
