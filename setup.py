
"""
NOTE about the 'setup.py' deprecation.
Despite the fact that 'setup.py' is now deprecated and replaced by 'pyproject.toml', it is still required to include C++ modules in pip packages.
Here is a minimal 'setup.py' that includes the C++ code in the package to complement the 'pyproject.toml' (this is fucked up)
"""

# Imports ----------------------------------------------------------------------
import os
from setuptools import setup, Extension

# Extensions -------------------------------------------------------------------

# Define extension (C++ code that need to be compiled)
compile_args = ['-std=c++17', '-O3']
link_args = ['-O3']
plmdca_ext = Extension(
    name='structuredca.dca_model.dca_solvers.plmdca.lib_plmdcaBackend',
    sources=[ # .cpp files
        'structuredca/dca_model/dca_solvers/plmdca/plmdcaBackend.cpp',
        'structuredca/dca_model/dca_solvers/plmdca/plmdca.cpp',
        'structuredca/dca_model/dca_solvers/plmdca/lbfgs/lib/lbfgs.cpp',
    ],
    include_dirs=[ # .h directories
        'structuredca/dca_model/dca_solvers/plmdca/include/',
        'structuredca/dca_model/dca_solvers/plmdca/lbfgs/include/',
    ],
    language='c++',
    extra_compile_args=compile_args,  # optimization and other flags
    extra_link_args=link_args,
)

# Setup ------------------------------------------------------------------------
setup(
    ext_modules = [plmdca_ext],
)