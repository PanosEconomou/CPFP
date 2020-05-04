from distutils.core import setup
from Cython.Build import cythonize

setup(ext_module = cythonize('3DSampleQT.pyx'))