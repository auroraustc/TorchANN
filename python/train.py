#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import torch as tf
import os
import gc

from class_and_function import *

"""Load coordinates, sym_coordinates, energy, force, type, n_atoms and parameters"""
###parameters incomplete
parameters = Parameters()
read_parameters_flag = read_parameters(parameters)
if (read_parameters_flag != 0):
    print("Reading parameters error with code %d\n"%read_parameters_flag)
    exit()

COORD = np.fromfile("./COORD.BIN", dtype = np.float64)
SYM_COORD = np.fromfile("./SYM_COORD.BIN", dtype = np.float64)
ENERGY = np.fromfile("./ENERGY.BIN", dtype = np.float64)
FORCE = np.fromfile("./FORCE.BIN", dtype = np.float64)
TYPE = np.fromfile("./TYPE.BIN", dtype = np.int32)
N_ATOMS = np.fromfile("./N_ATOMS.BIN", dtype = np.int32)
print(N_ATOMS)
print(np.dtype(np.float64).itemsize)
press_any_key_exit("Read complete. Press any key to continue\n")
"""Reshape COORD, FORCE as [Nfrmaes, N_Atoms_this_frame * 3] and SYM_COORD as [Nframes, N_Atoms_this_frame * SELA_max * 4]"""
"""DO NOT use np.reshape because it cannot deal with frames with different number of atoms."""
COORD_Reshape = reshape_to_frame_wise(COORD, N_ATOMS, parameters, 1)
print("COORD_Reshape: \n", COORD_Reshape)
FORCE_Reshape = reshape_to_frame_wise(FORCE, N_ATOMS, parameters, 1)
print("FORCE_Reshape: \n", FORCE_Reshape)
SYM_COORD_Reshape = reshape_to_frame_wise(SYM_COORD, N_ATOMS, parameters, 2)
print("SYM_COORD_Reshape: \n", SYM_COORD_Reshape)

"""SYM_COORD_Reshape_2 = read_reshape_DeepMD(N_ATOMS, parameters)
print("2:")
print(SYM_COORD_Reshape_2)"""

press_any_key_exit("Press any key to free memory.\n")
del COORD
del FORCE
del SYM_COORD
gc.collect()
press_any_key_exit("Memory free complete.\n")