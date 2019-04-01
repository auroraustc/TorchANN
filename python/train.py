import numpy as np
import torch as tf

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

"""Reshape COORD, FORCE as [Nfrmaes, N_Atoms_this_frame * 3] and SYM_COORD as [Nframes, N_Atoms_this_frame * SELA_max * 4]"""
"""DO NOT use np.reshape because it cannot deal with frames with different number of atoms."""
start_atom = 0
end_atom = N_ATOMS[0] - 1
FORCE_Reshape = np.array([np.append([], [FORCE[j] for j in range(start_atom * 3, (end_atom + 1) * 3)])], dtype = np.float64)
for i in range(1, parameters.Nframes_tot):
    start_atom = end_atom+ 1
    end_atom = end_atom + N_ATOMS[i]
    #print(np.append([], [FORCE[j] for j in range(start_atom * 3, (end_atom + 1) * 3)]))
    FORCE_Reshape = np.append(FORCE_Reshape, [np.append([], [FORCE[j] for j in range(start_atom * 3, (end_atom + 1) * 3)])], axis = 0)

print(FORCE_Reshape)
