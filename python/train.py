import numpy as np
import torch as tf
import os

from class_and_function import *

"""Load coordinates, sym_coordinates, energy, force, type, n_atoms and parameters"""
###parameters incomplete
parameters = Parameters()
read_parameters_flag = read_parameters(parameters)
if (read_parameters_flag != 0):
    print("Reading parameters error with code %d"%read_parameters_flag)
    exit()

COORD = np.fromfile("./COORD.BIN", dtype = np.float64)
SYM_COORD = np.fromfile("./SYM_COORD.BIN", dtype = np.float64)
ENERGY = np.fromfile("./ENERGY.BIN", dtype = np.float64)
FORCE = np.fromfile("./FORCE.BIN", dtype = np.float64)
TYPE = np.fromfile("./TYPE.BIN", dtype = np.int32)
N_ATOMS = np.fromfile("./N_ATOMS.BIN", dtype = np.int32)
print(N_ATOMS)
print(np.dtype(np.float64).itemsize)
"""Reshape COORD, FORCE as [Nfrmaes, N_Atoms_this_frame * 3] and SYM_COORD as [Nframes, N_Atoms_this_frame * SELA_max * 4]"""
"""DO NOT use np.reshape because it cannot deal with frames with different number of atoms."""
COORD_Reshape = reshape_to_frame_wise(COORD, N_ATOMS, parameters, 1)
print("COORD_Reshape: \n", COORD_Reshape)
FORCE_Reshape = reshape_to_frame_wise(FORCE, N_ATOMS, parameters, 1)
print("FORCE_Reshape: \n", FORCE_Reshape)
#SYM_COORD_Reshape = reshape_to_frame_wise(SYM_COORD, N_ATOMS, parameters, 2)
#print("SYM_COORD_Reshape: \n", SYM_COORD_Reshape)

###not complete_test
fp = open("./SYM_COORD.BIN", "rb")
interval = parameters.SEL_A_max * 4 * 1 * np.dtype(np.float64).itemsize
fp.seek(interval * (N_ATOMS[0] * 5 + 2), os.SEEK_SET)
tmp = fp.read(parameters.SEL_A_max * 4 * 1 * np.dtype(np.float64).itemsize)
fp_out = open("./tmp", "wb")
fp_out.write(tmp)
tmp = np.fromfile("./tmp", dtype = np.float64)
tmp1 = np.array([np.append([], tmp)])
print(tmp1)
fp_out.close()
fp.close()