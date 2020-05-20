# Solve a 3D transient Wavefunction
import numpy as np
import scipy.constants as c
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import pyqtgraph.exporters
import numpy as np
import pandas as pd
from tqdm import tqdm 
# from BS import *

MULTIPROCESSING = False
CPUs = 4

if MULTIPROCESSING:
    from multiprocessing import Process, Queue
    import itertools


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

    if type(K) == type(None):
        K = np.zeros(dim)
        K[1] = 100

    if x0 == None:
        x0 = np.zeros(dim)
        x0[0] = 0.5
        x0[1] = 0.3
        x0[2] = 0.5

    psi0 = np.zeros((Nx, Ny, Nz))*1j

    for i in tqdm(range(len(axes[0]))):  # xaxis
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
def getV(axes, r=0.2, x0=np.array([0.5, 0.5, 0.5])):
    Nx = int(len(axes[0]))
    Ny = int(len(axes[1]))
    Nz = int(len(axes[2]))

    V = np.zeros((Nx, Ny, Nz))

    for i in tqdm(range(len(axes[0]))):
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

# Finds the nearest index on an array based on value
def neari(array, value):
    array = np.asarray(array)
    idx = np.array([mag(arr - value) for arr in array]).argmin()
    return idx


def getBformfile(axes,filename):
    data = pd.read_csv(filename).to_numpy()
    
    Nx = len(axes[0])
    Ny = len(axes[1])
    Nz = len(axes[2])

    pos = np.array(data[:,0:3])

    B = np.zeros((Nx,Ny,Nz,3))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                r = np.array([axes[0][i],axes[1][j],axes[2][k]])
                B[i][j][k] = np.array(data[neari(pos,r)][3:])

    return B

def B(x,y,z,B0 = 10000, a = 1000, beta = 1):
    return np.array([-a*x,0,B0+a*z])/beta*0

#########################################################
#########################################################
# Simulation functions

# Perform a step of the imaginary wavefuntion
# @jit(target ="cuda")
def stepImag(R, I, V, spin, dx, dt, axes):
    Inew = np.zeros(I.shape)
    #Do everything but the boundary
    for i in range(1, len(axes[0])-1):
        for j in range(1, len(axes[1])-1):
            for k in range(1, len(axes[2])-1):
                Inew[i][j][k] = oneStepImag(i,j,k,R, I, V,spin,dx,dt,axes)

    # #Do the boundary
    # for i in [0, -1]:
    #     for j in [0, -1]:
    #         Inew[i][j] = Inew[3*i+1][3*i+1]

    return Inew

def oneStepImag(i,j,k,R, I, V,spin,dx,dt,axes):
    S = R[i+1][j][k]-2*R[i][j][k]+R[i-1][j][k] +\
        R[i][j+1][k]-2*R[i][j][k]+R[i][j-1][k] +\
        R[i][j][k+1]-2*R[i][j][k]+R[i][j][k-1]

    # return I[i][j][k] + h*dt/(2*m*dx**2)*S - (1/h)*V[i][j][k]*dt*R[i][j][k]
    # return I[i][j][k] + h*dt/(2*m*dx**2)*S + (q/h/m)*spin.dot(B(axes[0][i], axes[1][j], axes[2][k]))*R[i][j][k]*dt
    return I[i][j][k] + dt/(2*dx**2)*S + spin.dot(B(axes[0][i], axes[1][j], axes[2][k]))*R[i][j][k]*dt - V[i][j][k]*dt*R[i][j][k]
    # return I[i][j][k] + dt/(2*dx**2)*S + spin.dot(B[i][j][k])*R[i][j][k]*dt




# Perform a step of the real wavefuntion
# @jit(target ="cuda")
def stepReal(R, I, V, spin, dx, dt, axes):
    Rnew = np.zeros(R.shape)
    #Do everything but the boundary
    for i in range(1, len(axes[0])-1):
        for j in range(1, len(axes[1])-1):
            for k in range(1, len(axes[2])-1):
                Rnew[i][j][k] = oneStepReal(i,j,k,R, I, V,spin,dx,dt,axes)
                

    # #Do the boundary
    # for i in [0, -1]:
    #     for j in [0, -1]:
    #         Rnew[i][j] = Rnew[3*i+1][3*i+1]

    return Rnew

def oneStepReal(i,j,k,R, I, V,spin,dx,dt,axes):
    S = I[i+1][j][k]-2*I[i][j][k]+I[i-1][j][k] +\
        I[i][j+1][k]-2*I[i][j][k]+I[i][j-1][k] +\
        I[i][j][k+1]-2*I[i][j][k]+I[i][j][k-1]

    # return R[i][j][k] - h*dt/(2*m*dx**2)*S + (1/h)*V[i][j][k]*dt*I[i][j][k]
    # return R[i][j][k] - h*dt/(2*m*dx**2)*S - (q/h/m)*spin.dot(B(axes[0][i],axes[1][j],axes[2][k]))*I[i][j][k]*dt
    return R[i][j][k] - dt/(2*dx**2)*S - spin.dot(B(axes[0][i],axes[1][j],axes[2][k]))*I[i][j][k]*dt + V[i][j][k]*dt*I[i][j][k]
    # return R[i][j][k] - dt/(2*dx**2)*S - spin.dot(B[i][j][k])*I[i][j][k]*dt



def process(func,Q,args,R, I, V,spin,dx,dt,axes,VERBOSE=True):
    cnt = 0
    for arg in args:
        Q.put([arg[0],arg[1],arg[2],func(*arg,R, I, V,spin,dx,dt,axes)])
    if VERBOSE:print("\tReturning.")
    return

def step(R, I, V, spin, dx, dt, axes, VERBOSE=True):
    if not MULTIPROCESSING:
        Inew = stepImag(R, I, V, spin, dx, dt, axes)
        Rnew = stepReal(R, Inew, V, spin, dx, dt, axes)
    
    else:
        Nx = len(axes[0])
        Ny = len(axes[1])
        Nz = len(axes[2])

        # Generate the argument list
        if VERBOSE:print("Generating Argument List")
        iters = itertools.product(range(1,Nx-1), range(1, Ny-1), range(1, Nz-1))
        args = np.array([[i,j,k] for i,j,k in iters])
        args = np.array_split(args,CPUs,axis = 0)

        # Start solving for the imaginary component
        if VERBOSE: print("Solving for I")
        Inew = np.zeros(I.shape)
        Q = Queue()

        # Generate processes
        if VERBOSE: print("Generating Processes List")
        processes = []
        for arg in args:
            processes.append(Process(target=process,args=(oneStepImag,Q,arg,R, I, V,spin,dx,dt,axes,VERBOSE)))
        
        if VERBOSE: print("Starting processes")
        for p in processes:
            p.start()
            if VERBOSE: print("\t",p.name," started.")

        if VERBOSE: print("Reassembling")
        while True:
            running = any(p.is_alive() for p in processes)
            while not Q.empty():
                q = Q.get()
                Inew[q[0]][q[1]][q[2]]=q[3]
            if not running:
                break

        # Start solving for the Real component
        if VERBOSE: print("Solving for R")
        Rnew = np.zeros(R.shape)
        Q = Queue()

        # Generate processes
        processes = []
        for arg in args:
            processes.append(Process(target=process,args=(oneStepReal,Q,arg,R,Inew,V,spin,dx,dt,axes,VERBOSE)))
        
        if VERBOSE: print("Starting processes")
        for p in processes:
            p.start()
            if VERBOSE: print("\t",p.name," started.")

        if VERBOSE: print("Reassembling")
        while True:
            running = any(p.is_alive() for p in processes)
            while not Q.empty():
                q = Q.get()
                Rnew[q[0]][q[1]][q[2]]=q[3]
            if not running:
                break


    prob = abs(Rnew**2 + Inew*I)

    return Rnew, Inew, prob


## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    
    #########################################################
    #########################################################
    # To solve the wavefuncion
    # Simulation parameters
    dx = 0.025
    dt = 1e-5

    dim = int(3)

    # Nonedimentionalisation
    m = 1.7911934e-25 #c.m_e   # Particle mass in Kg
    q = c.e     # Particle charge in C (As)

    beta = 1    # B = beta*Bbar (Nondimentionalisation constant for B)

    hbar = c.hbar
    gamma = q/m     # Gyrometric ratio (relativistically corrected)

    L0      = (hbar/(m*gamma*beta))**0.5     # Nondimentionalised Length Coefficient
    T0      = 1/(gamma*beta)                 # Nondimentionalised Time Coefficient
    spin    = np.array([-1/2,0,1/2])         # Nondimentionalised spin
    L       = 1                              # Nondimentionalised container length
    time    = 10                             # Nondimentionalised time
    
    v0 = np.array([0,1e-4,0])*L0/T0 #123111
    Kbar = v0/L0
    print("K : ", Kbar)
    print("lamba: ",2*np.pi/mag(Kbar))
    print("r0: ", L0)
    print("t0: ", T0)
    print("B : ", beta)
    print("s : ", spin)

    axes, grid = getGrid(dx, Lx=1, Ly=1, Lz=1)

    # Initial conditions
    print("Getting V,B")
    V = getV(axes)
    # B = getBformfile(axes,'Bfield.txt')

    print("Getting psi0")
    psi = getPsi0(axes,K=Kbar)

    print("Getting Components")
    R = psi.real
    Iprev = psi.imag

    # Do half a time step to calculate the leapfrog imaginary.
    print("Doing half a step")
    I = stepImag(R, Iprev, V, spin, dx, dt/2, axes)
    print("Calculating Probability")
    prob = R**2 + Iprev*I

    # R,I,prob = step(R,I,V,spin,dx,dt,axes)
    # R,I,prob = step(R,I,V,spin,dx,dt,axes)

    #########################################################
    #########################################################
    # Plotting and stuffs
    import sys

    # Plotting Variables
    plotevery = 1
    level = 2
    Z = 0.5
    EXPORT = True

    # Qt Setup
    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.show()
    w.setWindowTitle('pyqtgraph example: GLIsosurface')

    w.setCameraPosition(azimuth=0,distance=30)

    g = gl.GLGridItem()
    g.scale(0.5, 0.5, 1)
    w.addItem(g)

    # Plot a the wavefunction isosurface
    verts, faces = pg.isosurface(prob, prob.max()/level)
    md = gl.MeshData(vertexes=verts, faces=faces)
    colors = np.ones((md.faceCount(), 4), dtype=float)
    colors[:, 3] = 0.2
    colors[:, 2] = np.linspace(1, 1, 1)
    md.setFaceColors(colors)

    m1 = gl.GLMeshItem(meshdata=md, smooth=True, shader='balloon')
    m1.setGLOptions('additive')
    w.addItem(m1)
    m1.translate(-L/(2*dx), -L/(2*dx), -L/(2*dx))


    #Plot the potential sphere
    verts, faces = pg.isosurface(V, V.max())
    md = gl.MeshData(vertexes=verts, faces=faces)
    colors = np.ones((md.faceCount(), 4), dtype=float)
    colors[:, 3] = 0.2
    colors[:, 2] = np.linspace(0.5, 0.5, 1)
    md.setFaceColors(colors)

    m2 = gl.GLMeshItem(meshdata=md, smooth=True, shader='balloon')
    m2.setGLOptions('additive')
    w.addItem(m2)
    m2.translate(-L/(2*dx), -L/(2*dx), -L/(2*dx))


    # Set up a slice view of it
    slice = gl.GLViewWidget()
    slice.show()
    slice.setWindowTitle('Slice view at a z-plane')

    slice.setCameraPosition(distance=1)

    g = gl.GLGridItem()
    g.scale(dx,dx,2)
    slice.addItem(g)

    sl = int(Z/dx)
    s = gl.GLSurfacePlotItem(x=axes[0],y=axes[1],z=prob[:][:][sl], shader='heightColor',smooth=False)
    s.translate(-L/2,-L/2,0)

    slice.addItem(s)

    i = 0
    def update():
        global R, I, prob, i
        print(i, prob.max())
        R, I, prob = step(R, I, V, spin, dx, dt, axes)
        if i%plotevery == 0:
            verts, faces = pg.isosurface(prob, prob.max()/level)
            md = gl.MeshData(vertexes=verts, faces=faces)
            colors = np.ones((md.faceCount(), 4), dtype=float)
            colors[:, 3] = 0.2
            colors[:, 2] = np.linspace(1, 1, 1)
            md.setFaceColors(colors)
            m1.setMeshData(meshdata=md)
            m1.meshDataChanged()
            
            s.setData(z=prob[:][:][sl]/prob.max())
            # Exporting
            if EXPORT: w.grabFrameBuffer().save('frames2/frame'+str(i)+'.png')
        i+=1

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(50)

    pg.setConfigOptions(antialias=True)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()


        
