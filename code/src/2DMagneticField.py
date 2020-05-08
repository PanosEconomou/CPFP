# Let's try Biot Savart Law

import numpy as np 
import scipy.constants as c
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import tqdm 


###################################################################################
###################################################################################
# Simulation Functions

dim = 3

# Returns the magnitude of an arbitrary vector
def mag(x):
    return x.dot(x)**0.5

#Creates dim axes, and also a meshgrid if helpful.
def getGrid(dx: float, L: float = 1, Lx=None, Ly=None, Lz=None):
    if Lx == None or Ly == None or Lz == None:
        N = int(L/dx)
        axes = [np.linspace(0, L, N)]*dim
    else:
        L = [Lx, Ly, Lz]
        N = [int(l/dx) for l in L]
        axes = [np.linspace(0, L[i], N[i]) for i in range(dim)]

    grid = np.meshgrid(*axes)

    return axes, grid


# Finds the nearest index on an array based on value
def neari(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def getJ(axes,x0=np.array([0.5,0.5,0.5])):

    Nx = len(axes[0])
    Ny = len(axes[1])
    Nz = len(axes[2])

    Jmax = 1

    J = np.zeros((Nx,Ny,Nz,3))

    # # Uncomment for a linear current
    # for i in range(0,Nx):
    #     J[i][int(Ny/2)][int(Nz/2)] = np.array([1,0,0])

    # # Uncomment for a circular current
    # for i in range(0, Nx):
    #     for j in range(0, Ny):
    #         for k in range(0, Nz):

    #             if axes[2][k] >= 0.5:
    #                 x = axes[0][i]-x0[0]
    #                 y = axes[1][j]-x0[1]

    #                 if abs(x**2+y**2 - 0.2**2) < 0.01:
    #                     if np.sqrt(x**2 + y**2) != 0:
    #                         J[i][j][k] = Jmax*np.array([y/np.sqrt(x**2 + y**2),-x/np.sqrt(x**2 + y**2),0])
    #                     else: 
    #                         J[i][j][k] = Jmax*np.array([0,0,0])

    # Uncomment for a SGD current
    h = 0.5
    w = 0.5
    Lx = axes[0].max()
    Lz = axes[2].max()
    for k in tqdm(range(int(Nz/2),Nz)):
        z = axes[2][k]
        x1 = Lx/2 - w/(2*h)*(z+h-Lz)
        x2 = Lx/2 + w/(2*h)*(z+h-Lz)
        
        i1 = neari(axes[0],x1)
        i2 = neari(axes[0],x2)

        # Add the two y strips
        for j in range(0,Ny):
            J[i1][j][k] = Jmax*np.array([0,1,0])
            J[i2][j][k] = Jmax*np.array([0,-1,0])

        #Complete the paralelogram
        for i in range(i1,i2):
            J[i][0][k] = Jmax*np.array([1,0,0])
            J[i+1][Ny-1][k] = Jmax*np.array([-1,0,0])

    # Add the bottom current loop
    for j in range(0,Ny):
        J[neari(axes[0], Lx/2-w/2)][j][0] = Jmax*np.array([0,1,0])
        J[neari(axes[0], Lx/2+w/2)][j][0] = Jmax*np.array([0,-1,0])

    #Complete the paralelogram
    for i in range(neari(axes[0], Lx/2-w/2),neari(axes[0], Lx/2+w/2)):
        J[i][0][0] = Jmax*np.array([1,0,0])
        J[i+1][Ny-1][0] = Jmax*np.array([-1,0,0])            

    return J


# Calculates a processed list of all the points that contain a current field 
def getJProcessed(axes,dx,J):

    Nx = len(axes[0])
    Ny = len(axes[1])
    Nz = len(axes[2])

    Jp = []

    for i in tqdm(range(0,Nx)):
        for j in range(0,Ny):
            for k in range(0,Nz):
                if mag(J[i][j][k]) > 0:
                    Jp.append([J[i][j][k],np.array([axes[0][i],axes[1][j],axes[2][k]])])

    return np.array(Jp)


# Ccalculates the magnetic field for one position based ont eh current
def calcB(x,y,z, Jp, dx, mu0 = c.mu_0, delta=1e-3):
    R = np.array([x,y,z])
    
    sum = np.array([0.]*dim)
    for J in Jp:
        r = R - J[1]
        sum+= np.cross(J[0],r)/(mag(r)**3+delta)
    sum*= mu0*dx**3/(4*np.pi)

    return sum


def solveB(axes,Jp,dx,mu0=c.mu_0,delta=1e-3):
    Nx = len(axes[0])
    Ny = len(axes[1])
    Nz = len(axes[2])

    B = np.zeros((Nx,Ny,Nz,dim))

    for i in tqdm(range(0,Nx)):
        for j in range(0,Ny):
            for k in range(0,Nz):
                B[i][j][k] = calcB(axes[0][i],axes[1][j],axes[2][k],Jp,dx,mu0=mu0,delta=delta)
    
    return B




###################################################################################
###################################################################################
# Simulation Commands

L = 1
dx = 0.08
mu0 = c.mu_0
delta = 1e-3

axes,grid = getGrid(dx,L)
J = getJ(axes)
Jp = getJProcessed(axes,dx,J)

B = solveB(axes,Jp,dx,mu0,delta)
print(B.max())

########################################################
########################################################
# Plotting stuff

fig = plt.figure(figsize=(8,8),dpi = 100)

ax = fig.add_subplot(111,projection='3d')
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_zlabel("z-axis")
ax.set_title("Magnetic field generated by a current")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)

ax.quiver(grid[0], grid[1], grid[2], J[:,:,:,1], J[:,:,:,0], J[:,:,:,2],length=0.1,color='darkblue')
ax.quiver(grid[0], grid[1], grid[2], B[:,:,:,1], B[:,:,:,0], B[:,:,:,2],length=1e7,color='darkred')

plt.show()
