from ctypes.util import find_library
import os.path

def library_path(lib_name):
    lib = find_library(lib_name)
    if lib is None:
        return None
    else:
        return os.path.split(lib)[0]
        
def has_library(lib_name):
    return find_library(lib_name) is not None