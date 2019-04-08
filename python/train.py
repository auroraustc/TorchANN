#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import torch as tf
import torch.utils.data
import os
import gc
import time
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
parameters.Nframes_tot = len(N_ATOMS)
print("Total number of frames: ", parameters.Nframes_tot)
print("Number of atoms aligned: ", N_ATOMS[0])
#press_any_key_exit("Read complete. Press any key to continue\n")
"""Reshape COORD, FORCE as [Nfrmaes, N_Atoms_this_frame * 3] and SYM_COORD as [Nframes, N_Atoms_this_frame * SELA_max * 4]"""
"""DO NOT use np.reshape because it cannot deal with frames with different number of atoms."""
"""Use np.reshape now and DO NOT use the damn reshape_to_frame_wise functions because I have aligned the number of atoms in all frames"""
#COORD_Reshape = reshape_to_frame_wise(COORD, N_ATOMS, parameters, 1)
COORD_Reshape = np.reshape(COORD, (parameters.Nframes_tot, -1))
print("COORD_Reshape: shape = ", COORD_Reshape.shape)#, "\n", COORD_Reshape)
#FORCE_Reshape = reshape_to_frame_wise(FORCE, N_ATOMS, parameters, 1)
FORCE_Reshape = np.reshape(FORCE, (parameters.Nframes_tot, -1))
print("FORCE_Reshape: shape = ", FORCE_Reshape.shape)#, "\n", FORCE_Reshape)
#SYM_COORD_Reshape = reshape_to_frame_wise(SYM_COORD, N_ATOMS, parameters, 2)
SYM_COORD_Reshape = np.reshape(SYM_COORD, (parameters.Nframes_tot, -1))

COORD_Reshape_tf = tf.from_numpy(COORD_Reshape)
SYM_COORD_Reshape_tf = tf.from_numpy(SYM_COORD_Reshape)
ENERGY_tf = tf.from_numpy(ENERGY)
FORCE_Reshape_tf = tf.from_numpy(FORCE_Reshape)
N_ATOMS_tf = tf.from_numpy(N_ATOMS)

print("SYM_COORD_Reshape: shape = ", SYM_COORD_Reshape.shape)#, "\n", SYM_COORD_Reshape)

"""SYM_COORD_Reshape_2 = read_reshape_DeepMD(N_ATOMS, parameters)
print("2:")
print(SYM_COORD_Reshape_2)"""

"""#No need to free memory now because the reshape_to_frame_wise function is not used
press_any_key_exit("Press any key to free memory.\n")
del COORD
del FORCE
del SYM_COORD
gc.collect()
press_any_key_exit("Memory free complete.\n")
"""

"""Now all the needed information has been stored in the COORD_Reshape, SYM_COORD_Reshape, 
   ENERGY and FORCE_Reshape array."""
print("Data pre-processing complete. Building net work.\n")

DATA_SET = tf.utils.data.TensorDataset(COORD_Reshape_tf, SYM_COORD_Reshape_tf, ENERGY_tf, FORCE_Reshape_tf, N_ATOMS_tf)
"""#Seems that no need to free memory...
press_any_key_exit("Press any key to free memory.\n")
del COORD_Reshape
del SYM_COORD_Reshape
del FORCE_Reshape
press_any_key_exit("Memory free complete.\n")
"""
TRAIN_LOADER = tf.utils.data.DataLoader(DATA_SET, batch_size = parameters.batch_size, shuffle = True)
START_TRAIN_TIMER = time.time()
for epoch in range(parameters.epoch):
    for batch_idx, data_cur in enumerate(TRAIN_LOADER):
        START_BATCH_TIMER = time.time()
        COORD_Reshape_tf_cur, SYM_COORD_Reshape_tf_cur, ENERGY_tf_cur, FORCE_Reshape_tf_cur, N_ATOMS_tf_cur = data_cur

        END_BATCH_TIMER = time.time()
        print("Epoch %-10d Batch %-10d: %10.3f s"%(epoch, batch_idx, END_BATCH_TIMER - START_BATCH_TIMER))
        #print(COORD_Reshape_tf_cur, SYM_COORD_Reshape_tf_cur, ENERGY_tf_cur, FORCE_Reshape_tf_cur, N_ATOMS_tf_cur)
END_TRAIN_TIMER = time.time()
ELAPSED_TRAIN = END_TRAIN_TIMER - START_TRAIN_TIMER
print("Training complete. Time elapsed: %10.3f s\n"%ELAPSED_TRAIN)