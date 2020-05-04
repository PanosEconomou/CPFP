# This is designed to plot a 3D function as a probability density cloud
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import tqdm

def mag(x):
    m = (x.dot(x))**0.5
    return m

# Create an array with all the points needed to plot of this form
# [[0.  0.  0.0]
#  [0.  0.  0.1]
#  [0.  0.  0.2]
#  ...
#  [1.  1.  0.3]
#  [1.  1.  0.4]
#  [1.  1.  0.5]]

x = []

dx = 0.05

Lx = 1
Ly = 1
Lz = 1

Nx = int(Lx/dx)+1
Ny = int(Ly/dx)+1
Nz = int(Lz/dx)+1

for i in tqdm(range(Nx)):
    for j in range(Ny):
        for k in range(Nz):
            x.append([i*dx,j*dx,k*dx])

x = np.array(x)


#Ploting a gaussian function
x0 = np.array([0.5, 0.5, 0.5])
s = 1e-1

# Calculate the function for each coordinate
psi = []
for i in tqdm(range(Nx)):
    for j in range(Ny):
        for k in range(Nz):
            psi.append([np.exp(-mag(x[k+Nz*j+Nz*Ny*i]-x0)**2/s**2)])
psi = np.array(psi)
# print(psi.T[0])

#Plot
fig = plt.figure(figsize = (10,10), dpi = 70)
ax = fig.add_subplot(111,projection='3d')

ax.set_title("3D Density plot")
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')

cmap = plt.cm.seismic#winter#gray_r
colors = cmap(psi.T[0] / max(psi.T[0]))
print(x.T[0])
print(x.T[1])
print(x.T[2])
print(psi.T[0])
colors[:, -1] = psi.T[0] / max(psi.T[0])


ax.scatter(x.T[0], x.T[1], x.T[2],c=colors,s=1)

plt.show()
