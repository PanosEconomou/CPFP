from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
import sys

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.show()
g = gl.GLGridItem()
w.addItem(g)

# pos = np.random.randint(-10, 10, size=(100, 10, 3))
# pos[:, :, 2] = np.abs(pos[:, :, 2])

# ScatterPlotItems = {}
# for point in np.arange(10):
#     ScatterPlotItems[point] = gl.GLScatterPlotItem(pos=pos[:, point, :])
#     w.addItem(ScatterPlotItems[point])

# color = np.zeros((pos.shape[0], 10, 4), dtype=np.float32)
# color[:, :, 0] = 1
# color[:, :, 1] = 0
# color[:, :, 2] = 0.5
# color[0:5, :, 3] = np.tile(np.arange(1, 6)/5., (10, 1)).T


# def update():
#     ## update volume colors
#     global color
#     for point in np.arange(10):
#         ScatterPlotItems[point].setData(color=color[:, point, :])
#     color = np.roll(color, 1, axis=0)


t = QtCore.QTimer()
# t.timeout.connect(update)
t.start(100)


## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
