# -*- coding: utf-8 -*-
"""
This example uses the isosurface function to convert a scalar field
(a hydrogen orbital) into a mesh for 3D display.
"""

## Add path to library (just for examples; you do not need this)
import numpy as np

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.show()
w.setWindowTitle('pyqtgraph example: GLIsosurface')

w.setCameraPosition(distance=40)

g = gl.GLGridItem()
g.scale(2, 2, 1)
w.addItem(g)


## Define a scalar field from which we will generate an isosurface

def psi(i, j, k, offset=(25, 25, 50)):
    x = i-offset[0]
    y = j-offset[1]
    z = k-offset[2]
    th = np.arctan2(z, (x**2+y**2)**0.5)
    phi = np.arctan2(y, x)
    r = (x**2 + y**2 + z ** 2)**0.5
    a0 = 1
    #ps = (1./81.) * (2./np.pi)**0.5 * (1./a0)**(3/2) * (6 - r/a0) * (r/a0) * np.exp(-r/(3*a0)) * np.cos(th)
    ps = (1./81.) * 1./(6.*np.pi)**0.5 * (1./a0)**(3/2) * \
        (r/a0)**2 * np.exp(-r/(3*a0)) * (3 * np.cos(th)**2 - 1)
    return ps

# Create a 3d normal distribution
N = 10
m = 0
M = 1
x = np.linspace(m, M, N)
y = np.linspace(m, M, N)
z = np.linspace(m, M, N)

def normal(X,Y,Z):
    X-=50
    X/= 100
    Y-=50
    Y /= 100
    Z-=50
    Z /= 100
    r = np.array([0,0,0])
    s = 1**0.5
    A = 1

    return A*np.exp(-((X-r[0])**2 + (Y-r[1])**2 + (Z-r[2])**2)/(s**2))

print("Generating scalar field..")
data = np.abs(np.fromfunction(normal, (100, 100, 100)))
print(data)
print("Generating isosurface..")
verts, faces = pg.isosurface(data, data.max()/1.0005)

md = gl.MeshData(vertexes=verts, faces=faces)

colors = np.ones((md.faceCount(), 4), dtype=float)
colors[:, 3] = 0.2
colors[:, 2] = np.linspace(0, 1, colors.shape[0])
md.setFaceColors(colors)

m2 = gl.GLMeshItem(meshdata=md, smooth=True, shader='balloon')
m2.setGLOptions('additive')

w.addItem(m2)
m2.translate(-50, -50, -50)


## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
