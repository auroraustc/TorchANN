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
from ctypes import *
from class_and_function import *

tf.set_default_dtype(tf.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
MYDLL = CDLL("../c/libNNMD.so")
MYDLL.init_read_coord.argtypes = [c_int, c_int, c_int, c_int]
MYDLL.init_read_coord.restype = POINTER(c_double)
MYDLL.init_read_nei_coord.argtypes = [c_int, c_int, c_int, c_int]
MYDLL.init_read_nei_coord.restype = POINTER(c_double)
MYDLL.init_read_nei_idx.argtypes = [c_int, c_int, c_int, c_int]
MYDLL.init_read_nei_idx.restype = POINTER(c_int)

MYDLL.compute_derivative_sym_coord_to_coord_one_frame_DeePMD.argtypes = [c_int, c_int, c_int, c_int, c_double, c_double, POINTER(c_double), POINTER(c_double), POINTER(c_int)]
MYDLL.compute_derivative_sym_coord_to_coord_one_frame_DeePMD.restype = c_void_p
MYDLL.freeme.argtypes = [c_void_p]
MYDLL.freeme.restypes = []
f_out = open("./LOSS.OUT", "w")
f_out.close()
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
NEI_IDX = np.fromfile("./NEI_IDX.BIN", dtype = np.int32)
NEI_COORD = np.fromfile("./NEI_COORD.BIN", dtype = np.float64)
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
NEI_IDX_Reshape = np.reshape(NEI_IDX, (parameters.Nframes_tot, -1))
print("NEI_IDX_Reshape: shape = ", NEI_IDX_Reshape.shape)
NEI_COORD_Reshape = np.reshape(NEI_COORD, (parameters.Nframes_tot, -1))
print("NEI_COORD_Reshape: shape = ", NEI_COORD_Reshape.shape)

COORD_Reshape_tf = tf.from_numpy(COORD_Reshape)
SYM_COORD_Reshape_tf = tf.from_numpy(SYM_COORD_Reshape)
ENERGY_tf = tf.from_numpy(ENERGY)
FORCE_Reshape_tf = tf.from_numpy(FORCE_Reshape)
N_ATOMS_tf = tf.from_numpy(N_ATOMS)
TYPE_Reshape_tf = tf.from_numpy(TYPE_Reshape)
NEI_IDX_Reshape_tf = tf.from_numpy(NEI_IDX_Reshape)
NEI_COORD_Reshape_tf = tf.from_numpy(NEI_COORD_Reshape)
FRAME_IDX_tf = tf.ones(len(COORD_Reshape_tf), dtype = tf.int32)
for i in range(len(FRAME_IDX_tf)):
    FRAME_IDX_tf[i] = i

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

ONE_BATCH_NET = one_batch_net(parameters)
ONE_BATCH_NET = ONE_BATCH_NET.to(device)
TOTAL_NUM_PARAMS = sum(p.numel() for p in ONE_BATCH_NET.parameters() if p.requires_grad)
if (tf.cuda.device_count() > 1):
    ONE_BATCH_NET = nn.DataParallel(ONE_BATCH_NET)
print(ONE_BATCH_NET)
print("Number of parameters in the net: %d"%TOTAL_NUM_PARAMS)


"""COORD_Reshape_tf, SYM_COORD_Reshape_tf, ENERGY_tf,  FORCE_Reshape_tf, N_ATOMS_tf, TYPE_Reshape_tf, NEI_IDX_Reshape_tf, \
NEI_COORD_Reshape_tf, FRAME_IDX_tf \
    = \
COORD_Reshape_tf.to(device), SYM_COORD_Reshape_tf.to(device), ENERGY_tf.to(device), FORCE_Reshape_tf.to(device), \
N_ATOMS_tf.to(device), TYPE_Reshape_tf.to(device), NEI_IDX_Reshape_tf.to(device), NEI_COORD_Reshape_tf.to(device), \
FRAME_IDX_tf.to(device)"""

DATA_SET = tf.utils.data.TensorDataset(COORD_Reshape_tf, SYM_COORD_Reshape_tf, \
                                       ENERGY_tf, FORCE_Reshape_tf, N_ATOMS_tf, TYPE_Reshape_tf, \
                                       NEI_IDX_Reshape_tf, NEI_COORD_Reshape_tf, FRAME_IDX_tf)
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
CRITERION = nn.MSELoss(reduction = "mean")
LR_SCHEDULER = tf.optim.lr_scheduler.ExponentialLR(OPTIMIZER2, parameters.decay_rate)
START_TRAIN_TIMER = time.time()
STEP_CUR = 0
read_coord_res = MYDLL.init_read_coord(parameters.Nframes_tot, 0, parameters.SEL_A_max, N_ATOMS[0])
read_nei_coord_res = MYDLL.init_read_nei_coord(parameters.Nframes_tot, 0, parameters.SEL_A_max, N_ATOMS[0])
read_nei_idx_res = MYDLL.init_read_nei_idx(parameters.Nframes_tot, 0, parameters.SEL_A_max, N_ATOMS[0])
print("Start training using device: ", device, ", count: ", tf.cuda.device_count())
#with tf.autograd.profiler.profile(enabled = True, use_cuda=True) as prof:
#hvd.broadcast_parameters(state_dict_, root_rank = 0)

if (True):
    for epoch in range(parameters.epoch):
        START_EPOCH_TIMER = time.time()
        if (parameters.epoch != 1 ):
            pref_e = (parameters.limit_pref_e - parameters.start_pref_e) * 1.0 / (parameters.epoch - 1.0) * epoch + parameters.start_pref_e
            pref_f = (parameters.limit_pref_f - parameters.start_pref_f) * 1.0 / (parameters.epoch - 1.0) * epoch + parameters.start_pref_f
        else:
            pref_e = parameters.start_pref_e
            pref_f = parameters.start_pref_f
        if ((epoch % parameters.decay_epoch == 0)):  # and (STEP_CUR > 0)):
            LR_SCHEDULER.step()
            print("LR update: lr = %f" % OPTIMIZER2.param_groups[0].get("lr"))
        for batch_idx, data_cur in enumerate(TRAIN_LOADER):
            for i in range(len(data_cur)):
                data_cur[i] = data_cur[i].to(device)
            START_BATCH_TIMER = time.time()
            data_cur[1].requires_grad = True
            NEI_IDX_Reshape_tf_cur = data_cur[6]
            NEI_IDX_Reshape_tf_cur = tf.reshape(NEI_IDX_Reshape_tf_cur, (len(NEI_IDX_Reshape_tf_cur), data_cur[4][0], parameters.SEL_A_max))
            FORCE_Reshape_tf_cur = data_cur[3]
            FORCE_Reshape_tf_cur_Reshape = tf.reshape(FORCE_Reshape_tf_cur, (len(FORCE_Reshape_tf_cur), data_cur[4][0] * 3))
            FORCE_net_tf_cur = tf.zeros((len(FORCE_Reshape_tf_cur), data_cur[4][0] * 3)).to(device)
            NEI_COORD_Reshape_tf_cur = data_cur[7]


            ###Adams
            #correct
            OPTIMIZER2.zero_grad()
            if (data_cur[1].grad):
                data_cur[1].grad.data.zero_()
            E_cur_batch = ONE_BATCH_NET(data_cur, parameters, device)
            #Energy loss part
            loss_E_cur_batch = CRITERION(E_cur_batch, data_cur[2])
            #Force loss part
            SYM_COORD_Reshape_tf_cur_grad = tf.autograd.grad(tf.sum(E_cur_batch), data_cur[1], create_graph = True)
            loss_F_cur_batch = tf.zeros(1).to(device)
            for i in range(len(data_cur[1])):
                size = data_cur[4][0] * parameters.SEL_A_max * 4 * data_cur[4][0] * 3
                size = size.item()
                res_from_c = MYDLL.compute_derivative_sym_coord_to_coord_one_frame_DeePMD(parameters.Nframes_tot, data_cur[8][i], parameters.SEL_A_max, data_cur[4][0], parameters.cutoff_2, parameters.cutoff_1, read_coord_res, read_nei_coord_res, read_nei_idx_res)
                res_from_c_copy = np.frombuffer((c_double * size).from_address(res_from_c), np.float64)
                res_from_c_copy = res_from_c_copy.reshape(data_cur[4][0] * parameters.SEL_A_max * 4, data_cur[4][0] * 3)
                res_from_c_copy = tf.from_numpy(res_from_c_copy).to(device)
                FORCE_net_tf_cur[i] = tf.mm(SYM_COORD_Reshape_tf_cur_grad[0][i].reshape(1, len(SYM_COORD_Reshape_tf_cur_grad[0][i])), res_from_c_copy)
                #loss_F_cur_batch += CRITERION() / math.sqrt
                MYDLL.freeme(res_from_c)
            #force_cur_batch = calc_force(SYM_COORD_Reshape_tf_cur_grad[0], NEI_IDX_Reshape_tf_cur, data_cur[0], NEI_COORD_Reshape_tf_cur, parameters, data_cur[4][0], device)
            loss_F_cur_batch = CRITERION(FORCE_net_tf_cur, FORCE_Reshape_tf_cur_Reshape)
            loss_cur_batch = pref_e * loss_E_cur_batch + pref_f * loss_F_cur_batch
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




            END_BATCH_TIMER = time.time()
            f_out = open("./LOSS.OUT","a")
            ###Adams
            print("Epoch: %-10d, Batch: %-10d, lossE: %10.3f eV/atom, lossF: %10.3f eV/A, time: %10.3f s" % (
            epoch, batch_idx, loss_E_cur_batch / N_ATOMS[0], loss_F_cur_batch, END_BATCH_TIMER - START_BATCH_TIMER))
            print("Epoch: %-10d, Batch: %-10d, lossE: %10.3f eV/atom, lossF: %10.3f eV/A, time: %10.3f s" % (\
                epoch, batch_idx, loss_E_cur_batch / N_ATOMS[0], loss_F_cur_batch, END_BATCH_TIMER - START_BATCH_TIMER),\
                  file = f_out)
            ###Adams end

            ###LBFGS
            """print("Epoch: %-10d, Batch: %-10d, loss: %10.3f eV/atom, time: %10.3f s" % (
                epoch, batch_idx, closure() / N_ATOMS[0], END_BATCH_TIMER - START_BATCH_TIMER))"""
            ###LBFGS end

            #print(COORD_Reshape_tf_cur, SYM_COORD_Reshape_tf_cur, ENERGY_tf_cur, FORCE_Reshape_tf_cur, N_ATOMS_tf_cur)
            f_out.close()
            STEP_CUR += 1


            """if (STEP_CUR >= 2):
                break"""
        END_EPOCH_TIMER = time.time()


        if (False):
            break

if (tf.cuda.device_count() > 1):
    torch.save(ONE_BATCH_NET.module.state_dict(), "./freeze_model_DataParallal.pytorch")
    print("Model saved to ./freeze_model_DataParallal.pytorch")
else:
    torch.save(ONE_BATCH_NET.state_dict(), "./freeze_model.pytorch")
    print("Model saved to ./freeze_model.pytorch")
MYDLL.freeme(read_coord_res)
MYDLL.freeme(read_nei_coord_res)
MYDLL.freeme(read_nei_idx_res)
END_TRAIN_TIMER = time.time()
ELAPSED_TRAIN = END_TRAIN_TIMER - START_TRAIN_TIMER
print("Training complete. Time elapsed: %10.3f s\n"%ELAPSED_TRAIN)
#print(prof.table(sort_by = "cuda_time"))