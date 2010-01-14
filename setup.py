#!/usr/bin/env python

import os

from distribute_setup import use_setuptools
use_setuptools()

from setuptools import setup, find_packages, Extension
from setup_helper import *

boost_suffixes_to_try = ["-mt", ""] 
coin_libs = ["Clp", "CoinUtils", "Osi", "OsiClp"]


LIB_DIRS = []
BOOST_PYTHON_LIBRARY = "boost_python"
BOOST_INCLUDE_PATH = os.environ.get('BOOST_INCLUDE_PATH', '/usr/include/boost')
COIN_INCLUDE_PATH = os.environ.get('COIN_INCLUDE_PATH', '/usr/include/coin')

for suffix in boost_suffixes_to_try:    
    candidate_name = "boost_python" + suffix
    if has_library(candidate_name):
        BOOST_PYTHON_LIBRARY = candidate_name
        LIB_DIRS.append(library_path(candidate_name))
        break

for coin_lib in coin_libs:
    if has_library(coin_lib):
        lib_path = library_path(coin_lib)
        if lib_path not in LIB_DIRS:
            LIB_DIRS.append(lib_path)
    else:
        raise Exception("The %s library is required to build this project" % coin_lib)


setup(name='mclp',
      description="A python wrapper for Peter Gehler and Sebastian Nowozin's Multiclass LP Boosting package",
      packages = find_packages(exclude=['ez_setup']),
      include_package_data = True,
      zip_safe = False,
      setup_requires=['nose>=0.11'],
      test_suite = "nose.collector",
      ext_modules = [
        Extension('mclp._mclp', 
                  ['mclp/LPBoostPythonWrapper.cpp', 'mclp/original_src/LPBoostMulticlassClassifier.cpp'], 
                  libraries = [BOOST_PYTHON_LIBRARY, 'Clp', 'CoinUtils', 'Osi', 'OsiClp'],
                  library_dirs = LIB_DIRS,
                  include_dirs = ['mclp/original_src', COIN_INCLUDE_PATH, BOOST_INCLUDE_PATH],
                  )
        ],
      )
