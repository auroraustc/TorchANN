#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import math
import torch as tf
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import os
import gc
import time
from class_and_function import *

tf.set_default_dtype(tf.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
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
"""FILTER_NET = []
for type_idx in range(len(parameters.type_index_all_frame)):
    FILTER_NET.append(filter_net(parameters).to(device))
print("FILTER_NET structure: \n", FILTER_NET)
FITTING_NET = []
for type_idx in range(len(parameters.type_index_all_frame)):
    FITTING_NET.append(fitting_net(parameters).to(device))
print("FITTING_NET structure:\n", FITTING_NET)
NET_PARAMS_LIST = []
for type_idx in range(len(parameters.type_index_all_frame)):
    NET_PARAMS_LIST += list(FILTER_NET[type_idx].parameters())
    NET_PARAMS_LIST += list(FITTING_NET[type_idx].parameters())"""
#print("NET_PARAMS_LIST:\n", NET_PARAMS_LIST)
ONE_ATOM_NET = []
for type_idx in range(len(parameters.type_index_all_frame)):
    ONE_ATOM_NET.append(one_atom_net(parameters).to(device))
ONE_ATOM_NET_PARAMS = []
for type_idx in range(len(parameters.type_index_all_frame)):
    ONE_ATOM_NET_PARAMS += list(ONE_ATOM_NET[type_idx].parameters())

COORD_Reshape_tf, SYM_COORD_Reshape_tf, ENERGY_tf,  FORCE_Reshape_tf, N_ATOMS_tf, TYPE_Reshape_tf \
    = \
COORD_Reshape_tf.to(device), SYM_COORD_Reshape_tf.to(device), ENERGY_tf.to(device), \
FORCE_Reshape_tf.to(device), N_ATOMS_tf.to(device), TYPE_Reshape_tf.to(device)
DATA_SET = tf.utils.data.TensorDataset(COORD_Reshape_tf, SYM_COORD_Reshape_tf,
                                       ENERGY_tf, FORCE_Reshape_tf, N_ATOMS_tf, TYPE_Reshape_tf)
"""#Seems that no need to free memory...
press_any_key_exit("Press any key to free memory.\n")
del COORD_Reshape
del SYM_COORD_Reshape
del FORCE_Reshape
press_any_key_exit("Memory free complete.\n")
"""
TRAIN_LOADER = tf.utils.data.DataLoader(DATA_SET, batch_size = parameters.batch_size, shuffle = True)
OPTIMIZER = optim.Adam(ONE_ATOM_NET_PARAMS, lr = parameters.start_lr)
OPTIMIZER2 = optim.LBFGS(ONE_ATOM_NET_PARAMS, lr = parameters.start_lr)
CRITERION = nn.MSELoss()
LR_SCHEDULER = tf.optim.lr_scheduler.ExponentialLR(OPTIMIZER, parameters.decay_rate)
START_TRAIN_TIMER = time.time()
STEP_CUR = 0
print("Start training using device: ", device, ", count: ", tf.cuda.device_count())
with tf.autograd.profiler.profile(enabled = True, use_cuda=True) as prof:
    for epoch in range(parameters.epoch):
        START_EPOCH_TIMER = time.time()
        for batch_idx, data_cur in enumerate(TRAIN_LOADER):
            START_BATCH_TIMER = time.time()

            COORD_Reshape_tf_cur, SYM_COORD_Reshape_tf_cur, ENERGY_tf_cur, \
            FORCE_Reshape_tf_cur, N_ATOMS_tf_cur, TYPE_Reshape_tf_cur = data_cur
            ###Adams
            OPTIMIZER.zero_grad()
            # print(SYM_COORD_Reshape_tf_cur.view(parameters.batch_size, N_ATOMS[0], -1))
            SYM_COORD_Reshape_tf_cur_Reshape = tf.reshape(SYM_COORD_Reshape_tf_cur, (
            len(SYM_COORD_Reshape_tf_cur), N_ATOMS[0], parameters.SEL_A_max, 4))
            SYM_COORD_Reshape_tf_cur_Reshape_slice = SYM_COORD_Reshape_tf_cur_Reshape.narrow(3, 0, 1)
            # print(SYM_COORD_Reshape_tf_cur_Reshape.shape)
            # print(SYM_COORD_Reshape_tf_cur_Reshape_slice)
            # G_cur_list = []
            # E_cur_batch_list = []
            E_cur_batch = tf.zeros(len(SYM_COORD_Reshape_tf_cur)).to(device)
            # E_cur_frame = tf.zeros(1).to(device)
            """for frame_idx in range(len(SYM_COORD_Reshape_tf_cur)):
                #G_cur_frame_list = []
                G_cur_frame = tf.zeros(N_ATOMS[0], parameters.SEL_A_max,
                                       parameters.filter_neuron[len(parameters.filter_neuron) - 1]).to(device)
                for atom_idx in range(N_ATOMS[0]):
                    type_idx_cur_atom = parameters.type_index_all_frame.index(TYPE_Reshape_tf_cur[frame_idx][atom_idx])
                    #G_cur_frame_list.append(FILTER_NET[type_idx_cur_atom](SYM_COORD_Reshape_tf_cur_Reshape_slice[frame_idx][atom_idx]))
                    G_cur_frame[atom_idx] = FILTER_NET[type_idx_cur_atom](SYM_COORD_Reshape_tf_cur_Reshape_slice[frame_idx][atom_idx])
                #G_cur_frame = tf.cat(G_cur_frame_list, dim = 0)
                #G_cur_frame = G_cur_frame.view(N_ATOMS[0], parameters.SEL_A_max,
                #                               parameters.filter_neuron[len(parameters.filter_neuron) - 1])

                #print(G_cur_frame_list)
                #print(G_cur_frame)
                RG_cur_frame = tf.bmm(SYM_COORD_Reshape_tf_cur_Reshape[frame_idx].transpose(1, 2), G_cur_frame)
                GRRG_cur_frame = tf.bmm(RG_cur_frame.transpose(1, 2), RG_cur_frame.narrow(2, 0, parameters.axis_neuron))
                GRRG_cur_frame = tf.reshape(GRRG_cur_frame, (-1, parameters.filter_neuron[len(parameters.filter_neuron) - 1] * parameters.axis_neuron))
                #E_cur_frame_list = []
                E_cur_frame = tf.zeros(N_ATOMS[0])
                for atom_idx in range(N_ATOMS[0]):
                    type_idx_cur_atom = parameters.type_index_all_frame.index(TYPE_Reshape_tf_cur[frame_idx][atom_idx])
                    #E_cur_frame_list.append(FITTING_NET[type_idx_cur_atom](GRRG_cur_frame[atom_idx]))
                    E_cur_frame[atom_idx] = FITTING_NET[type_idx_cur_atom](GRRG_cur_frame[atom_idx])
                #E_cur_frame = tf.cat(E_cur_frame_list)
                #E_cur_frame = tf.sum(E_cur_frame)
                #print(E_cur_frame)
                #E_cur_batch_list.append(E_cur_frame)
                E_cur_batch[frame_idx] = sum(E_cur_frame)"""
            for frame_idx in range(len(SYM_COORD_Reshape_tf_cur)):
                E_cur_frame = tf.zeros(1).to(device)
                for atom_idx in range(N_ATOMS[0]):
                    type_idx_cur_atom = parameters.type_index_all_frame.index(TYPE_Reshape_tf_cur[frame_idx][atom_idx])
                    E_cur_atom = ONE_ATOM_NET[type_idx_cur_atom](SYM_COORD_Reshape_tf_cur_Reshape[frame_idx][atom_idx],
                                                                 SYM_COORD_Reshape_tf_cur_Reshape_slice[frame_idx][
                                                                     atom_idx], parameters)
                    E_cur_frame += E_cur_atom
                E_cur_batch[frame_idx] = E_cur_frame
            # print(E_cur_batch_list)
            # E_cur_batch = tf.stack(E_cur_batch_list)
            # print(E_cur_batch)
            # print(ENERGY_tf_cur)

            loss_cur_batch = CRITERION(E_cur_batch, ENERGY_tf_cur) / math.sqrt(len(SYM_COORD_Reshape_tf_cur))
            loss_cur_batch.to(device)
            loss_cur_batch.backward()
            OPTIMIZER.step()
            ###Adams end

            """###LBFGS

            ###LBFGS end"""
            if ((STEP_CUR % parameters.decay_steps == 0) and (STEP_CUR > 0)):
                LR_SCHEDULER.step()

            END_BATCH_TIMER = time.time()
            print("Epoch: %-10d, Batch: %-10d, loss: %10.6feV, time: %10.3f s" % (
            epoch, batch_idx, loss_cur_batch, END_BATCH_TIMER - START_BATCH_TIMER))
            # print(COORD_Reshape_tf_cur, SYM_COORD_Reshape_tf_cur, ENERGY_tf_cur, FORCE_Reshape_tf_cur, N_ATOMS_tf_cur)
            STEP_CUR += 1

            if (STEP_CUR >= 2):
                break
        END_EPOCH_TIMER = time.time()
        print("Epoch: %-8d, loss: %10.6f eV, time: %6.3f" % (
        epoch, loss_cur_batch, END_EPOCH_TIMER - START_EPOCH_TIMER))

        if (epoch >= 0):
            break

"""for epoch in range(parameters.epoch):
    START_EPOCH_TIMER = time.time()
    for batch_idx, data_cur in enumerate(TRAIN_LOADER):
        START_BATCH_TIMER = time.time()

        COORD_Reshape_tf_cur, SYM_COORD_Reshape_tf_cur, ENERGY_tf_cur, \
            FORCE_Reshape_tf_cur, N_ATOMS_tf_cur, TYPE_Reshape_tf_cur = data_cur
        ###Adams
        OPTIMIZER.zero_grad()
        #print(SYM_COORD_Reshape_tf_cur.view(parameters.batch_size, N_ATOMS[0], -1))
        SYM_COORD_Reshape_tf_cur_Reshape = tf.reshape(SYM_COORD_Reshape_tf_cur, (len(SYM_COORD_Reshape_tf_cur), N_ATOMS[0], parameters.SEL_A_max, 4))
        SYM_COORD_Reshape_tf_cur_Reshape_slice = SYM_COORD_Reshape_tf_cur_Reshape.narrow(3, 0, 1)
        #print(SYM_COORD_Reshape_tf_cur_Reshape.shape)
        #print(SYM_COORD_Reshape_tf_cur_Reshape_slice)
        #G_cur_list = []
        #E_cur_batch_list = []
        E_cur_batch = tf.zeros(len(SYM_COORD_Reshape_tf_cur)).to(device)
        #E_cur_frame = tf.zeros(1).to(device)
        ""for frame_idx in range(len(SYM_COORD_Reshape_tf_cur)):
            #G_cur_frame_list = []
            G_cur_frame = tf.zeros(N_ATOMS[0], parameters.SEL_A_max,
                                   parameters.filter_neuron[len(parameters.filter_neuron) - 1]).to(device)
            for atom_idx in range(N_ATOMS[0]):
                type_idx_cur_atom = parameters.type_index_all_frame.index(TYPE_Reshape_tf_cur[frame_idx][atom_idx])
                #G_cur_frame_list.append(FILTER_NET[type_idx_cur_atom](SYM_COORD_Reshape_tf_cur_Reshape_slice[frame_idx][atom_idx]))
                G_cur_frame[atom_idx] = FILTER_NET[type_idx_cur_atom](SYM_COORD_Reshape_tf_cur_Reshape_slice[frame_idx][atom_idx])
            #G_cur_frame = tf.cat(G_cur_frame_list, dim = 0)
            #G_cur_frame = G_cur_frame.view(N_ATOMS[0], parameters.SEL_A_max,
            #                               parameters.filter_neuron[len(parameters.filter_neuron) - 1])

            #print(G_cur_frame_list)
            #print(G_cur_frame)
            RG_cur_frame = tf.bmm(SYM_COORD_Reshape_tf_cur_Reshape[frame_idx].transpose(1, 2), G_cur_frame)
            GRRG_cur_frame = tf.bmm(RG_cur_frame.transpose(1, 2), RG_cur_frame.narrow(2, 0, parameters.axis_neuron))
            GRRG_cur_frame = tf.reshape(GRRG_cur_frame, (-1, parameters.filter_neuron[len(parameters.filter_neuron) - 1] * parameters.axis_neuron))
            #E_cur_frame_list = []
            E_cur_frame = tf.zeros(N_ATOMS[0])
            for atom_idx in range(N_ATOMS[0]):
                type_idx_cur_atom = parameters.type_index_all_frame.index(TYPE_Reshape_tf_cur[frame_idx][atom_idx])
                #E_cur_frame_list.append(FITTING_NET[type_idx_cur_atom](GRRG_cur_frame[atom_idx]))
                E_cur_frame[atom_idx] = FITTING_NET[type_idx_cur_atom](GRRG_cur_frame[atom_idx])
            #E_cur_frame = tf.cat(E_cur_frame_list)
            #E_cur_frame = tf.sum(E_cur_frame)
            #print(E_cur_frame)
            #E_cur_batch_list.append(E_cur_frame)
            E_cur_batch[frame_idx] = sum(E_cur_frame)""
        for frame_idx in range(len(SYM_COORD_Reshape_tf_cur)):
            E_cur_frame = tf.zeros(1).to(device)
            for atom_idx in range(N_ATOMS[0]):
                type_idx_cur_atom = parameters.type_index_all_frame.index(TYPE_Reshape_tf_cur[frame_idx][atom_idx])
                E_cur_atom = ONE_ATOM_NET[type_idx_cur_atom](SYM_COORD_Reshape_tf_cur_Reshape[frame_idx][atom_idx], SYM_COORD_Reshape_tf_cur_Reshape_slice[frame_idx][atom_idx], parameters)
                E_cur_frame += E_cur_atom
            E_cur_batch[frame_idx] = E_cur_frame
        #print(E_cur_batch_list)
        #E_cur_batch = tf.stack(E_cur_batch_list)
        #print(E_cur_batch)
        #print(ENERGY_tf_cur)

        loss_cur_batch = CRITERION(E_cur_batch, ENERGY_tf_cur) / math.sqrt(len(SYM_COORD_Reshape_tf_cur))
        loss_cur_batch.to(device)
        loss_cur_batch.backward()
        OPTIMIZER.step()
        ###Adams end

        if ((STEP_CUR % parameters.decay_steps == 0) and (STEP_CUR > 0)):
            LR_SCHEDULER.step()

        END_BATCH_TIMER = time.time()
        print("Epoch: %-10d, Batch: %-10d, loss: %10.6feV, time: %10.3f s"%(epoch, batch_idx, loss_cur_batch, END_BATCH_TIMER - START_BATCH_TIMER))
        #print(COORD_Reshape_tf_cur, SYM_COORD_Reshape_tf_cur, ENERGY_tf_cur, FORCE_Reshape_tf_cur, N_ATOMS_tf_cur)
        STEP_CUR += 1

        if (STEP_CUR >= 2):
            break
    END_EPOCH_TIMER = time.time()
    print("Epoch: %-8d, loss: %10.6f eV, time: %6.3f"%(epoch, loss_cur_batch, END_EPOCH_TIMER - START_EPOCH_TIMER))


    if (epoch >= 0):
        break
"""
END_TRAIN_TIMER = time.time()
ELAPSED_TRAIN = END_TRAIN_TIMER - START_TRAIN_TIMER
print("Training complete. Time elapsed: %10.3f s\n"%ELAPSED_TRAIN)
print(prof.table(sort_by = "cuda_time"))