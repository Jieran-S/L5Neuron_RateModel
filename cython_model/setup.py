from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension('network', ['network.pyx'],)]

for e in ext_modules:
    e.cython_directives = {'language_level': "3"}

setup(
  name='Rate network',
  ext_modules=ext_modules,
  include_dirs=[numpy.get_include()],
  cmdclass={'build_ext': build_ext},
)