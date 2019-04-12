#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import math
import torch as tf
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import horovod.torch as hvd
import os
import gc
import time
from class_and_function import *

tf.set_default_dtype(tf.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
#hvd.init()
#tf.cuda.set_device(hvd.local_rank())
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
print("SYM_COORD_Reshape: shape = ", SYM_COORD_Reshape.shape)#, "\n", SYM_COORD_Reshape)
TYPE_Reshape = np.reshape(TYPE, (parameters.Nframes_tot, -1))
print("TYPE_Reshape: shape = ", TYPE_Reshape.shape)

COORD_Reshape_tf = tf.from_numpy(COORD_Reshape)
SYM_COORD_Reshape_tf = tf.from_numpy(SYM_COORD_Reshape)
ENERGY_tf = tf.from_numpy(ENERGY)
FORCE_Reshape_tf = tf.from_numpy(FORCE_Reshape)
N_ATOMS_tf = tf.from_numpy(N_ATOMS)
TYPE_Reshape_tf = tf.from_numpy(TYPE_Reshape)

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

ONE_ATOM_NET = []
for type_idx in range(len(parameters.type_index_all_frame)):
    ONE_ATOM_NET.append(one_atom_net(parameters))
for type_idx in range(len(parameters.type_index_all_frame)):
    ONE_ATOM_NET[type_idx].to(device)
ONE_ATOM_NET_PARAMS = []
for type_idx in range(len(parameters.type_index_all_frame)):
    ONE_ATOM_NET_PARAMS += list(ONE_ATOM_NET[type_idx].parameters())

ONE_BATCH_NET = one_batch_net(parameters)
ONE_BATCH_NET = ONE_BATCH_NET.to(device)
if (tf.cuda.device_count() > 1):
    ONE_BATCH_NET = nn.DataParallel(ONE_BATCH_NET)
print(ONE_BATCH_NET)

COORD_Reshape_tf, SYM_COORD_Reshape_tf, ENERGY_tf,  FORCE_Reshape_tf, N_ATOMS_tf, TYPE_Reshape_tf \
    = \
COORD_Reshape_tf.to(device), SYM_COORD_Reshape_tf.to(device), ENERGY_tf.to(device), \
FORCE_Reshape_tf.to(device), N_ATOMS_tf.to(device), TYPE_Reshape_tf.to(device)
DATA_SET = tf.utils.data.TensorDataset(COORD_Reshape_tf, SYM_COORD_Reshape_tf,
                                       ENERGY_tf, FORCE_Reshape_tf, N_ATOMS_tf, TYPE_Reshape_tf)
"""Seems that no need to free memory...
press_any_key_exit("Press any key to free memory.\n")
del COORD_Reshape
del SYM_COORD_Reshape
del FORCE_Reshape
press_any_key_exit("Memory free complete.\n")
"""
TRAIN_LOADER = tf.utils.data.DataLoader(DATA_SET, batch_size = parameters.batch_size, shuffle = True)
OPTIMIZER2 = optim.Adam(ONE_BATCH_NET.parameters(), lr = parameters.start_lr)
OPTIMIZER = optim.LBFGS(ONE_BATCH_NET.parameters(), lr = parameters.start_lr)
CRITERION = nn.MSELoss()
LR_SCHEDULER = tf.optim.lr_scheduler.ExponentialLR(OPTIMIZER2, parameters.decay_rate)
START_TRAIN_TIMER = time.time()
STEP_CUR = 0
print("Start training using device: ", device, ", count: ", tf.cuda.device_count())
#with tf.autograd.profiler.profile(enabled = True, use_cuda=True) as prof:
#hvd.broadcast_parameters(state_dict_, root_rank = 0)
if (True):
    for epoch in range(parameters.epoch):
        START_EPOCH_TIMER = time.time()
        for batch_idx, data_cur in enumerate(TRAIN_LOADER):
            START_BATCH_TIMER = time.time()
            #print(ONE_BATCH_NET(data_cur))

            """COORD_Reshape_tf_cur, SYM_COORD_Reshape_tf_cur, ENERGY_tf_cur, \
            FORCE_Reshape_tf_cur, N_ATOMS_tf_cur, TYPE_Reshape_tf_cur = data_cur"""
            """for i in range(len(data_cur)):
                data_cur[i] = data_cur[i].to(device)"""
            ###Adams
            #correct
            OPTIMIZER2.zero_grad()
            E_cur_batch = ONE_BATCH_NET(data_cur, parameters, device)
            loss_cur_batch = CRITERION(E_cur_batch, data_cur[2]) / math.sqrt(len(data_cur[1]))
            loss_cur_batch.backward()
            OPTIMIZER2.step()
            #correct end
            ###Adams end

            ###LBFGS
            #correct
            """def closure():
                OPTIMIZER.zero_grad()
                E_cur_batch = ONE_BATCH_NET(data_cur, parameters, device)
                loss_cur_batch = CRITERION(E_cur_batch, data_cur[2]) / math.sqrt(len(data_cur[1]))
                loss_cur_batch.backward()
                return loss_cur_batch
            OPTIMIZER.step(closure)"""
            #correct end
            ###LBFGS end


            if ((STEP_CUR % parameters.decay_steps == 0)):# and (STEP_CUR > 0)):
                LR_SCHEDULER.step()
                print("LR update: lr = %f"%OPTIMIZER2.param_groups[0].get("lr"))

            END_BATCH_TIMER = time.time()
            ###Adams
            print("Epoch: %-10d, Batch: %-10d, loss: %10.3f eV/atom, time: %10.3f s" % (
            epoch, batch_idx, loss_cur_batch / N_ATOMS[0], END_BATCH_TIMER - START_BATCH_TIMER))
            ###Adams end

            ###LBFGS
            """print("Epoch: %-10d, Batch: %-10d, loss: %10.3f eV/atom, time: %10.3f s" % (
                epoch, batch_idx, closure() / N_ATOMS[0], END_BATCH_TIMER - START_BATCH_TIMER))"""
            ###LBFGS end

            #print(COORD_Reshape_tf_cur, SYM_COORD_Reshape_tf_cur, ENERGY_tf_cur, FORCE_Reshape_tf_cur, N_ATOMS_tf_cur)
            STEP_CUR += 1


            """if (STEP_CUR >= 2):
                break"""
        END_EPOCH_TIMER = time.time()


        if (False):
            break

END_TRAIN_TIMER = time.time()
ELAPSED_TRAIN = END_TRAIN_TIMER - START_TRAIN_TIMER
print("Training complete. Time elapsed: %10.3f s\n"%ELAPSED_TRAIN)
#print(prof.table(sort_by = "cuda_time"))