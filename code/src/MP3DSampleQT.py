# Solve a 3D transient Wavefunction
import numpy as np
import scipy.constants as c
# from pyqtgraph.Qt import QtCore, QtGui
# import pyqtgraph as pg
# import pyqtgraph.opengl as gl
import numpy as np
import itertools
# from tqdm import tqdm
from multiprocessing import Pool
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import axes3d

NCORES = 100

def mag(x):
    return x.dot(x)**0.5

#Creates dim axes, and also a meshgrid if helpful.
def getGrid(dx: float, L: float = 1, Lx=None, Ly=None, Lz=None):
    if type(Lx) == None or type(Ly) == None or type(Lz) == None:
        N = int(L/dx)
        axes = [np.linspace(0, L, N)]*dim
    else:
        L = [Lx, Ly, Lz]
        N = [int(l/dx) for l in L]
        axes = [np.linspace(0, L[i], N[i]) for i in range(dim)]

    grid = np.meshgrid(*axes)

    return axes, grid


# Define a gaussian wavepacket centered at x0 with a particular wave attached ot it
def getPsi0(axes, K=None, C: float = 1, x0=None, s0=0.5e-2**0.5):
    Nx = int(len(axes[0]))
    Ny = int(len(axes[1]))
    Nz = int(len(axes[2]))

    if K == None:
        K = np.zeros(dim)
        K[1] = 40

    if x0 == None:
        x0 = np.zeros(dim)
        x0[0] = 0.5
        x0[1] = 0.2
        x0[2] = 0.5

    psi0 = np.zeros((Nx, Ny, Nz))*1j

    for i in (range(len(axes[0]))):  # xaxis
        for j in range(len(axes[1])):  # yaxis
            for k in range(len(axes[2])):  # z axis
                #set the wavefunction
                sum = (axes[0][i]-x0[0])**2 +\
                    (axes[1][j]-x0[1])**2 +\
                    (axes[2][k]-x0[2])**2

                kf = K[0]*axes[0][i] +\
                    K[1]*axes[1][j] +\
                    K[2]*axes[2][k]

                psi0[i][j][k] = C*np.exp(-(sum/(s0**2)))*np.exp(kf*1j)

    return psi0


# Creates a potential with a really big thing in its center
def getV(axes, r=0.3, x0=np.array([0.5, 0.5, 0.5])):
    Nx = int(len(axes[0]))
    Ny = int(len(axes[1]))
    Nz = int(len(axes[2]))

    V = np.zeros((Nx, Ny, Nz))

    for i in (range(len(axes[0]))):
        for j in range(len(axes[1])):
            for k in range(len(axes[2])):
                x = axes[0][i]
                y = axes[1][j]
                z = axes[2][k]

                R = np.array([x, y, z])

                if mag(R-x0)<= r**2:
                    V[i][j][k] = 1e4

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
            for k in range(1, len(axes[2])-1):
                Inew[i][j][k] = oneStepImag(i,j,k,R,I,V,dx,dt,axes)

    # #Do the boundary
    # for i in [0, -1]:
    #     for j in [0, -1]:
    #         Inew[i][j] = Inew[3*i+1][3*i+1]

    return Inew

def oneStepImag(i,j,k,R,I,V,dx,dt,axes):
    S = R[i+1][j][k]-2*R[i][j][k]+R[i-1][j][k] +\
        R[i][j+1][k]-2*R[i][j][k]+R[i][j-1][k] +\
        R[i][j][k+1]-2*R[i][j][k]+R[i][j][k-1]

    return I[i][j][k] + h*dt/(2*m*dx**2)*S - (1/h)*V[i][j][k]*dt*R[i][j][k]

# Perform a step of the real wavefuntion
def stepReal(R, I, V, dx, dt, axes):
    Rnew = np.zeros(R.shape)
    #Do everything but the boundary
    for i in range(1, len(axes[0])-1):
        for j in range(1, len(axes[1])-1):
            for k in range(1, len(axes[2])-1):
                Rnew[i][j][k] = oneStepReal(i,j,k,R,I,V,dx,dt,axes)
                

    #Do the boundary
    for i in [0, -1]:
        for j in [0, -1]:
            Rnew[i][j] = Rnew[3*i+1][3*i+1]

    return Rnew

def oneStepReal(i,j,k,R,I,V,dx,dt,axes):
    
    S = I[i+1][j][k]-2*I[i][j][k]+I[i-1][j][k] +\
        I[i][j+1][k]-2*I[i][j][k]+I[i][j-1][k] +\
        I[i][j][k+1]-2*I[i][j][k]+I[i][j][k-1]

    return R[i][j][k] - h*dt/(2*m*dx**2)*S + (1/h)*V[i][j][k]*dt*I[i][j][k]


def step(R, I, V, dx, dt, axes):
    Inew = stepImag(R, I, V, dx, dt, axes)
    Rnew = stepReal(R, Inew, V, dx, dt, axes)

    prob = abs(Rnew**2 + Inew*I)

    return Rnew, Inew, prob


def stepMT(R, I, V, dx, dt, axes):
    with Pool(NCORES) as p:
        # data = []
        # for i in range(len(axes[0])):
        #     for j in range(len(axes[1])):
        #         for k in range(len(axes[2])):
        #             data.append((i,j,k,R,I,V,axes))
        
        Inew = np.zeros(I.shape)
        Rnew = np.zeros(R.shape)
        Nx = len(axes[0])
        Ny = len(axes[1])
        Nz = len(axes[2])

        Inew[1:Nx-1,1:Ny-1,1:Nz-1] = np.array([p.apply_async(oneStepImag, (i, j, k, R, I, V, dx, dt, axes)).get() \
            for i,j,k in itertools.product(range(1,Nx-1),range(1,Ny-1),range(1,Nz-1))])\
                .reshape(Nx-2,Ny-2,Nz-2)
        
        Rnew[1:Nx-1,1:Ny-1,1:Nz-1] = np.array([p.apply_async(oneStepReal, (i, j, k, R, I, V, dx, dt, axes)).get() \
            for i, j, k in itertools.product(range(1, Nx-1), range(1, Ny-1), range(1, Nz-1))])\
                .reshape(Nx-2,Ny-2,Nz-2)
        


    prob = abs(Rnew**2 + Inew*I)

    return Rnew, Inew, prob


#########################################################
#########################################################
# To solve the wavefuncion
# Simulation parameters
dx = 0.05
dt = 3e-5

dim = int(3)
h = 1
m = 1
L = 1
time = 10

axes, grid = getGrid(dx, Lx=1, Ly=1, Lz=1)

# Initial conditions
V = getV(axes)

psi = getPsi0(axes)

R = psi.real
Iprev = psi.imag

# Do half a time step to calculate the leapfrog imaginary.
I = stepImag(R, Iprev, V, dx, dt/2, axes)
prob = R**2 + Iprev*I


#########################################################
#########################################################
# Plotting and stuffs

# Plotting Variables
plotevery = 1
level = 1.5
Z = 0.5
sl = int(Z/dx)

#Create figure
fig = plt.figure(figsize=(15, 15), dpi=40)
ax = fig.add_subplot(111, projection='3d')
plotEvery = 1

# Axis specific
ax.set_title("Wavefunction over time\nTimestep: 0")
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")

ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_zlim(0, 1)

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
        global R, I
        print(i)
        R, I, prob = stepMT(R, I, V, dx, dt, axes)
        print(i,prob.max())

    # wave.set_array((prob[:][:][sl]/prob.max()).ravel())
    # print(i)

    global wireframe
    if wireframe:
        ax.collections.remove(wireframe)
    X,Y = np.meshgrid(axes[0],axes[1])
    wireframe = ax.plot_wireframe(X,Y, prob[:][:][sl]/prob.max(), rstride=1, cstride=1, color='k', linewidth=0.5)

    # return wave,


# anim = animation.FuncAnimation(fig, update, init_func=animInit, frames=int(time/dt), interval=1, blit=False)
anim = animation.FuncAnimation(fig, update, frames=500, interval=20, blit=False)
# anim.save("example1.gif", fps=30, writer='imagemagick')

plt.show()



# # Qt Setup
# app = QtGui.QApplication([])
# w = gl.GLViewWidget()
# w.show()
# w.setWindowTitle('pyqtgraph example: GLIsosurface')

# w.setCameraPosition(distance=20)

# g = gl.GLGridItem()
# g.scale(0.5, 0.5, 1)
# w.addItem(g)

# # Plot a the wavefunction isosurface
# verts, faces = pg.isosurface(prob, prob.max()/level)
# md = gl.MeshData(vertexes=verts, faces=faces)
# colors = np.ones((md.faceCount(), 4), dtype=float)
# colors[:, 3] = 0.2
# colors[:, 2] = np.linspace(1, 1, 1)
# md.setFaceColors(colors)

# m1 = gl.GLMeshItem(meshdata=md, smooth=True, shader='balloon')
# m1.setGLOptions('additive')
# w.addItem(m1)
# m1.translate(-L/(2*dx), -L/(2*dx), -L/(2*dx))


# #Plot the potential sphere
# verts, faces = pg.isosurface(V, V.max())
# md = gl.MeshData(vertexes=verts, faces=faces)
# colors = np.ones((md.faceCount(), 4), dtype=float)
# colors[:, 3] = 0.2
# colors[:, 2] = np.linspace(0.5, 0.5, 1)
# md.setFaceColors(colors)

# m2 = gl.GLMeshItem(meshdata=md, smooth=True, shader='balloon')
# m2.setGLOptions('additive')
# w.addItem(m2)
# m2.translate(-L/(2*dx), -L/(2*dx), -L/(2*dx))


# # Set up a slice view of it
# slice = gl.GLViewWidget()
# slice.show()
# slice.setWindowTitle('Slice view at a z-plane')

# slice.setCameraPosition(distance=1)

# g = gl.GLGridItem()
# g.scale(dx,dx,2)
# slice.addItem(g)

# sl = int(Z/dx)
# s = gl.GLSurfacePlotItem(x=axes[0],y=axes[1],z=prob[:][:][sl], shader='heightColor',smooth=False)
# s.translate(-L/2,-L/2,0)

# slice.addItem(s)

# ## Start Qt event loop unless running in interactive mode.
# if __name__ == '__main__':
#     import sys

#     i = 0
#     def update():
#         global R, I, i
#         print(i)
#         R, I, prob = stepMT(R, I, V, dx, dt, axes)
#         if i%plotevery == 0:
#             verts, faces = pg.isosurface(prob, prob.max()/level)
#             print(prob.max())
#             md = gl.MeshData(vertexes=verts, faces=faces)
#             colors = np.ones((md.faceCount(), 4), dtype=float)
#             colors[:, 3] = 0.2
#             colors[:, 2] = np.linspace(1, 1, 1)
#             md.setFaceColors(colors)
#             m1.setMeshData(meshdata=md)
#             m1.meshDataChanged()
            
#             s.setData(z=prob[:][:][sl]/prob.max())
#         i += 1

#     timer = QtCore.QTimer()
#     timer.timeout.connect(update)
#     timer.start(50)

#     pg.setConfigOptions(antialias=True)

#     if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#         QtGui.QApplication.instance().exec_()


        
