# Solve a 1D transient Wavefunction
from LA import *
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import scipy.constants as c
from tqdm import tqdm


# Set the time of the running simulation
time = 10
dx = 1e-3
dt = dx**2/2#5e-7
iterations = int(time/dt)

# X-Axis
M = 1
x = np.linspace(0,M,int(M/dx))

# Potential
V = np.zeros(len(x))
# V[int(0.6/dx):int(0.7/dx)] = 1000
V[0] = 1e10
V[int(len(V)*0.6):int(len(V)*0.8)] = 0*1e6

# Initial condition
m  = 1 #c.electron_mass
k0 = 500
w0 = 0
x0 = 0.5
s  = 1e-3**0.5
C = 10#1/(s*(2*np.pi)**0.5)
psi0 = C*np.exp(-(x-x0)**2/(s**2))#*np.exp(k0*x*1j)

R0 = psi0.real
R0[0] = 0
R0[-1] = 0

# Perform half a step to get I0
psi1 = psi0-dt/2*1j*V*psi0 - dt/(4j*dx**2)*(np.append(psi0[1:], [0], axis=0)-2*psi0+np.append([0], psi0[:-1], axis=0))

I0 = psi1.imag
I0[0] = 0
I0[-1] = 0
R = R0
I = I0
prob = I0**2+R0**2

def step(i):
    # Step halfway to calculate the Imaginary part
    global R,I,prob
    Rk = R
    Ik = I
    
    N = len(Rk)

    Ii = np.zeros(N)
    for j in range(1,N-1):
        Ii[j] = Ik[j] + dt/(2*dx**2)*(Rk[j+1]-2*Rk[j]+Rk[j-1]) - dt*V[j]*Rk[j]
    Ii[0] = Ii[1]
    Ii[-1] = Ii[-2]

    Ri = np.zeros(N)
    for j in range(1,N-1):
        Ri[j] = Rk[j] - dt/(2*dx**2)*(Ii[j+1]-2*Ii[j]+Ii[j-1]) + dt*V[j]*Ii[j]
    Ri[0] = Ri[1]
    Ri[-1] = Ri[-2]


    #Renormalisation
    Ri = np.array(Ri)
    Ii = np.array(Ii)
    # Ri[0] = 0
    # Ri[-1]= 0
    # Ii[0] = 0
    # Ii[-1]= 0

    prob = R**2+Ii*Ik

    R = Ri
    I = Ii

## Animation Commands
fig = plt.figure(figsize = (8,8),dpi = 80)
ax = fig.add_subplot(111)
ax.set_title("Wavefunction Over time\nTimestep: 0")
ax.set_xlabel("x-axis [m]")
ax.set_ylabel("Probability")
ax.grid()
ax.set_xlim(0,1)
ax.set_ylim(0,200)

plot, = ax.plot([],[],c='k')
# plotR, = ax.plot([], [], c='blue')
# plotI, = ax.plot([], [], c='darkred')

def initAnim():
    # psi = P0
    plot.set_data([],[])#x,prob)
    # plotR.set_data(x,R)
    # plotI.set_data(x,I)

    ax.set_title("Wavefunction Over time\nTimestep: 0")
    ax.set_xlabel("x-axis [m]")
    ax.set_ylabel("Probability")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 200)

    return plot, #plotR, plotI

def update(i):
    ax.set_title("Wavefunction Over time\nTimestep: %d"%i)

    for i in range(10):
        step(i)
   
    # prob = (R**2+I**2)
    plot.set_data(x,prob)
    # plotR.set_data(x, R)
    # plotI.set_data(x, I)

    # print("\nTimestep:",i)
    # for i in range(len(R)):
    #     print(R[i],I[i])

    # ax.set_ylim(0, 200)#max(prob))

    return plot, #plotR, plotI


anim = animation.FuncAnimation(fig, update, init_func=initAnim, frames=int(500), interval=1, blit=False)
anim.save("1D_STATIONARY.mp4", fps=30, writer='ffmpeg')
plt.show()
