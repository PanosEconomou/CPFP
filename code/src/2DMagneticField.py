# Let's try Biot Savart Law

import numpy as np 
import scipy.constants as c
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import tqdm 
import time

MULTIPROCESSING = True
CPUs = 4

if MULTIPROCESSING:
    from multiprocessing import Process, Queue
    import itertools

###################################################################################
###################################################################################
# Simulation Functions

dim = 3

# Returns the magnitude of an arbitrary vector
def mag(x):
    return x.dot(x)**0.5

#Creates dim axes, and also a meshgrid if helpful.
def getGrid(dx: float,Lx, Ly, Lz):
    L = [Lx, Ly, Lz]

    N = [int(abs(l[1]-l[0])/dx) for l in L]
    axes = [np.linspace(L[i][0], L[i][1], N[i]) for i in range(dim)]
    grid = np.meshgrid(*axes)

    return axes, grid


# Finds the nearest index on an array based on value
def neari(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# Cerates a current distribution based on some axes as a stern gerlach magnet
def getJ(axes,w=0.5,h=0.5,H=0.5,D=0.5):
    print("J is being generated")

    Nx = len(axes[0])
    Ny = len(axes[1])
    Nz = len(axes[2])

    Jmax = 1

    J = np.zeros((Nx,Ny,Nz,3))

    # # Uncomment for a linear current
    # for i in range(0,Nx):
    #     J[i][int(Ny/2)][int(Nz/2)] = np.array([1,0,0])

    # # Uncomment for a circular current
    # x0=np.array([0.5,0.5,0.5])
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
    Lx = axes[0].max()
    Lz = axes[2].max()
    for k in tqdm(range(neari(axes[2],Lz-h),Nz)):
        z = axes[2][k]
        x1 = Lx/2 - w/(2*h)*(z+h-Lz)
        x2 = Lx/2 + w/(2*h)*(z+h-Lz)
        
        i1 = neari(axes[0],x1)
        i2 = neari(axes[0],x2)

        # Add the two y strips
        for j in range(neari(axes[1],D),Ny):
            J[i1][j][k] = Jmax*np.array([0,1,0])
            J[i2][j][k] = Jmax*np.array([0,-1,0])

        #Complete the paralelogram
        for i in range(i1,i2):
            J[i][neari(axes[1],D)][k] = Jmax*np.array([1,0,0])
            J[i+1][Ny-1][k] = Jmax*np.array([-1,0,0])

    # Add the bottom current loop
    for k in range(0,neari(axes[2],H)):
        for j in range(neari(axes[1],D),Ny):
            J[neari(axes[0], Lx/2-w/2)][j][k] = Jmax*np.array([0,1,0])
            J[neari(axes[0], Lx/2+w/2)][j][k] = Jmax*np.array([0,-1,0])

        #Complete the paralelogram
        for i in range(neari(axes[0], Lx/2-w/2),neari(axes[0], Lx/2+w/2)):
            J[i][neari(axes[1],D)][k] = Jmax*np.array([1,0,0])
            J[i+1][Ny-1][k]           = Jmax*np.array([-1,0,0])            

    return J


# Calculates a processed list of all the points that contain a current field 
def getJProcessed(axes,dx,J):
    print("J is being processed")

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


# Ccalculates the magnetic field for one position based on the current
def calcB(x,y,z, Jp, dx, mu0 = c.mu_0, delta=1e-3):
    R = np.array([x,y,z])
    
    sum = np.array([0.]*dim)
    for J in Jp:
        r = R - J[1]
        sum+= np.cross(J[0],r)/(mag(r)**3+delta)
    sum*= mu0*dx**3/(4*np.pi)

    return sum


def solveB(axes,Jp,dx,mu0=c.mu_0,delta=1e-3, VERBOSE = True):
    Nx = len(axes[0])
    Ny = len(axes[1])
    Nz = len(axes[2])

    B = np.zeros((Nx,Ny,Nz,dim))

    if not MULTIPROCESSING:
        for i in tqdm(range(0,Nx)):
            for j in range(0,Ny):
                for k in range(0,Nz):
                    B[i][j][k] = calcB(axes[0][i],axes[1][j],axes[2][k],Jp,dx,mu0=mu0,delta=delta)
    
    else:
        # Generate the argument lists and split
        if VERBOSE: print("Generating Argument Lists")
        iters = itertools.product(range(0,Nx), range(0, Ny), range(0, Nz))
        args = np.array([[i,j,k] for i,j,k in iters])
        args = np.array_split(args,CPUs,axis = 0)

        # Create a queue to store all the incoming results
        Q = Queue()

        # Generate the processes
        if VERBOSE: print("Generating Processes List")
        processes = []
        for arg in args:
            processes.append(Process(target=process,args=(Q,axes,arg,Jp,dx,mu0,delta)))

        # Start Processes
        if VERBOSE: print("Starting processes")
        for p in processes:
            # p.daemon = False
            p.start()
            if VERBOSE: print("\t",p.name," started.")

        results = []
        # Reassemble the whole thing and return
        if VERBOSE: print("Reassembling")
        while True:
            running = any(p.is_alive() for p in processes)
            while not Q.empty():
                results.append(Q.get())
            if not running:
                break

        reassemble(B,results)

    
    return B

def reassemble(B,Q):
    for q in Q:
        B[q[0]][q[1]][q[2]] = q[3]

def process(Q,axes,args,Jp,dx,mu0=c.mu_0,delta=1e-3):
    for arg in args:
        # print("\tprocessing: ",arg)
        B = calcB(axes[0][arg[0]],axes[1][arg[1]],axes[2][arg[2]],Jp,dx,mu0=mu0,delta=delta)
        Q.put([arg[0],arg[1],arg[2],B])
    
    print("Retruning...")
    return


def export(axes,B,filename):
    file = open(filename,'w+')
    
    Nx = len(axes[0])
    Ny = len(axes[1])
    Nz = len(axes[2])

    print("Exporting Started")
    for i in tqdm(range(0,Nx)):
        for j in range(0,Ny):
            for k in range(0,Nz):
                x = str(axes[0][i])
                y = str(axes[1][j])
                z = str(axes[2][k])
                file.write(x+","+y+","+z+","+str(B[i][j][k])+"\n")

    file.close()

########################################################
########################################################
# Plotting stuff

def plot(axes,J,B):
    print("Started plotting")
    # Figure
    fig = plt.figure(figsize=(16,8),dpi = 100)

    # Add 3D quiver plot
    ax = fig.add_subplot(121,projection='3d')
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")
    ax.set_title("Magnetic field generated by a current")

    ax.set_xlim(axes[0].min(), axes[0].max())
    ax.set_ylim(axes[1].min(), axes[1].max())
    ax.set_zlim(axes[2].min(), axes[2].max())


    ax.quiver(grid[0], grid[1], grid[2], J[:,:,:,1], J[:,:,:,0], J[:,:,:,2],length=0.1,color='darkblue')
    ax.quiver(grid[0], grid[1], grid[2], B[:,:,:,1], B[:,:,:,0], B[:,:,:,2],length=1e7,color='darkred')

    # Add cross section plot
    Y  = axes[2].max()/2
    nY = neari(axes[2],Y)

    # Get a 3D array of the magnitude
    Bmag = np.zeros(B[:,nY,:,0].shape)

    for i in tqdm(range(len(Bmag))):
        for j in range(len(Bmag[i])):
            Bmag[i][j] = mag(B[i][nY][j])

    ax2 = fig.add_subplot(122)
    ax2.imshow(np.rot90(Bmag, k=1, axes=(0, 1)),cmap='hot')
    ax2.set_title('Slice at Y = %.2f'%Y)
    ax2.set_xlabel('x-axis')
    ax2.set_ylabel('y-axis')

    plt.show()




###################################################################################
###################################################################################
# Simulation Commands

# Main thread
if __name__ == '__main__':

    L = 2
    dx = 0.5
    mu0 = c.mu_0
    delta = 1e-3

    axes,grid = getGrid(dx,(0,L),(0,L),(0,L))

    J = getJ(axes,w=1,h=0.5,H=0.5)
    Jp = getJProcessed(axes,dx,J)

    # dx = 0.2
    # axes,grid = getGrid(dx,(0.5,1.5),(0,1),(0.5,1.5))

    B = solveB(axes,Jp,dx,mu0,delta)
    print(B.max())

    export(axes,B,"Bfield.txt")

    plot(axes,J,B)

