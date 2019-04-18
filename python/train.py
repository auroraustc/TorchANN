#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import math
import torch as tf
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
#import horovod.torch as hvd
import os
import gc
import time
from ctypes import *
from class_and_function import *

tf.set_default_dtype(tf.float64)
device = tf.device('cuda' if torch.cuda.is_available() else 'cpu')
#hvd.init()
#tf.cuda.set_device(0)
#print("hvd.size():", hvd.size())
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
"""Load coordinates, sym_coordinates, energy, force, type, n_atoms and parameters"""
###parameters incomplete
parameters = Parameters()
read_parameters_flag = read_parameters(parameters)
print("All parameters:")
print(parameters)#incomplete, add __str__ method
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

"""Input normalization"""
avg_ = tf.ones(len(SYM_COORD_Reshape_tf), 4)
std_ = tf.ones(len(SYM_COORD_Reshape_tf), 4)
for frame_idx in range(len(SYM_COORD_Reshape_tf)):
    SYM_COORD_Reshape_tf_cur_frame = tf.reshape(SYM_COORD_Reshape_tf[frame_idx], (N_ATOMS_tf[frame_idx], parameters.SEL_A_max, 4))
    avg_sr = tf.sum(SYM_COORD_Reshape_tf_cur_frame.narrow(2, 0, 1)) /  N_ATOMS_tf[frame_idx] / parameters.SEL_A_max
    avg_sr2 = tf.sum((SYM_COORD_Reshape_tf_cur_frame.narrow(2, 0, 1))**2) /  N_ATOMS_tf[frame_idx] / parameters.SEL_A_max
    std_sr = tf.sqrt(avg_sr2 - avg_sr**2)
    avg_xyz = tf.sum(SYM_COORD_Reshape_tf_cur_frame.narrow(2, 1, 3)) /  N_ATOMS_tf[frame_idx] / parameters.SEL_A_max / 3
    avg_xyz2 = tf.sum((SYM_COORD_Reshape_tf_cur_frame.narrow(2, 1, 3))**2) / N_ATOMS_tf[frame_idx] / parameters.SEL_A_max / 3
    std_xyz = tf.sqrt(avg_xyz2 - avg_xyz**2)
    avg_[frame_idx][0] = avg_sr
    avg_[frame_idx][1:] = avg_xyz
    std_[frame_idx][0] = std_sr
    std_[frame_idx][1:] = std_xyz
    SYM_COORD_Reshape_tf_cur_frame = (SYM_COORD_Reshape_tf_cur_frame - avg_[frame_idx]) / std_[frame_idx]
    SYM_COORD_Reshape_tf_cur_frame = tf.reshape(SYM_COORD_Reshape_tf_cur_frame, (-1,))
    SYM_COORD_Reshape_tf[frame_idx] = SYM_COORD_Reshape_tf_cur_frame

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
                                       NEI_IDX_Reshape_tf, NEI_COORD_Reshape_tf, FRAME_IDX_tf, avg_, std_)
#TRAIN_SAMPLER = tf.utils.data.distributed.DistributedSampler(DATA_SET, num_replicas=2, rank=hvd.rank())
"""Seems that no need to free memory...
press_any_key_exit("Press any key to free memory.\n")
del COORD_Reshape
del SYM_COORD_Reshape
del FORCE_Reshape
press_any_key_exit("Memory free complete.\n")
"""
#TRAIN_LOADER = tf.utils.data.DataLoader(DATA_SET, batch_size = parameters.batch_size, sampler = TRAIN_SAMPLER)
TRAIN_LOADER = tf.utils.data.DataLoader(DATA_SET, batch_size = parameters.batch_size, shuffle = True)
OPTIMIZER2 = optim.Adadelta(ONE_BATCH_NET.parameters(), lr = parameters.start_lr)
#OPTIMIZER2 = hvd.DistributedOptimizer(OPTIMIZER2, named_parameters=ONE_BATCH_NET.named_parameters())
"""
###DO NOT use LBFGS. LBFGS is horrible on such kind of optimizations
###Adam work also horribly. So it is better to use Adadelta
OPTIMIZER = optim.LBFGS(ONE_BATCH_NET.parameters(), lr = parameters.start_lr)
"""

#hvd.broadcast_parameters(ONE_BATCH_NET.state_dict(), root_rank=0)
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


            ###Adam
            #correct
            if (data_cur[1].grad):
                data_cur[1].grad.data.zero_()
            #Energy
            E_cur_batch = ONE_BATCH_NET(data_cur, parameters, device)
            #Energy loss part
            loss_E_cur_batch = CRITERION(E_cur_batch, data_cur[2])
            #Force
            loss_F_cur_batch = tf.zeros(1).to(device)

            D_E_D_SYM_COORD_cur_batch = tf.autograd.grad(tf.sum(E_cur_batch), data_cur[1], create_graph = True)[0]
            for frame_idx in range(len(data_cur[1])):
                col = N_ATOMS[0] * 3
                row = parameters.SEL_A_max * N_ATOMS[0] * 4
                size = col * row
                D_SYM_COORD_D_COORD_cur_frame = MYDLL.compute_derivative_sym_coord_to_coord_one_frame_DeePMD(parameters.Nframes_tot, frame_idx, parameters.SEL_A_max, N_ATOMS[0], parameters.cutoff_2, parameters.cutoff_1, read_coord_res, read_nei_coord_res, read_nei_idx_res)
                D_SYM_COORD_D_COORD_cur_frame_copy = np.frombuffer((c_double * size).from_address(D_SYM_COORD_D_COORD_cur_frame), np.float64)
                D_SYM_COORD_D_COORD_cur_frame_copy = np.reshape(D_SYM_COORD_D_COORD_cur_frame_copy, (row, col))
                D_SYM_COORD_D_COORD_cur_frame_copy = (tf.from_numpy(D_SYM_COORD_D_COORD_cur_frame_copy).to(device))*(-1)
                D_SYM_COORD_D_COORD_cur_frame_copy = D_SYM_COORD_D_COORD_cur_frame_copy.transpose(0,1)
                D_SYM_COORD_D_COORD_cur_frame_copy = tf.reshape((tf.reshape(D_SYM_COORD_D_COORD_cur_frame_copy, (col, N_ATOMS_tf[0], parameters.SEL_A_max, 4))) / data_cur[10][frame_idx], (col, row)).transpose(0, 1)
                FORCE_net_tf_cur[frame_idx] = (-1) * tf.mm(D_E_D_SYM_COORD_cur_batch[frame_idx].reshape(1, len(D_E_D_SYM_COORD_cur_batch[frame_idx])), D_SYM_COORD_D_COORD_cur_frame_copy)
                MYDLL.freeme(D_SYM_COORD_D_COORD_cur_frame)
            if ((STEP_CUR % 50 == 0)):
                print(FORCE_net_tf_cur)
            loss_F_cur_batch = CRITERION(FORCE_net_tf_cur, data_cur[3])

            loss_cur_batch = pref_e * loss_E_cur_batch   + pref_f * loss_F_cur_batch
            OPTIMIZER2.zero_grad()
            loss_cur_batch.backward()
            OPTIMIZER2.step()
            #correct end
            ###Adam end


            """
            ###LBFGS
            #correct
            # correct
            def closure():
                #if (data_cur[1].grad):
                #    data_cur[1].grad.data.zero_()
                # Energy
                E_cur_batch = ONE_BATCH_NET(data_cur, parameters, device)
                # Energy loss part
                loss_E_cur_batch = CRITERION(E_cur_batch, data_cur[2])
                # Force
                loss_F_cur_batch = tf.zeros(1).to(device)

                D_E_D_SYM_COORD_cur_batch = tf.autograd.grad(tf.sum(E_cur_batch), data_cur[1], retain_graph=True)[0]
                for frame_idx in range(len(data_cur[1])):
                    col = N_ATOMS[0] * 3
                    row = parameters.SEL_A_max * N_ATOMS[0] * 4
                    size = col * row
                    D_SYM_COORD_D_COORD_cur_frame = MYDLL.compute_derivative_sym_coord_to_coord_one_frame_DeePMD(
                        parameters.Nframes_tot, frame_idx, parameters.SEL_A_max, N_ATOMS[0], parameters.cutoff_2,
                        parameters.cutoff_1, read_coord_res, read_nei_coord_res, read_nei_idx_res)
                    D_SYM_COORD_D_COORD_cur_frame_copy = np.frombuffer(
                        (c_double * size).from_address(D_SYM_COORD_D_COORD_cur_frame), np.float64)
                    D_SYM_COORD_D_COORD_cur_frame_copy = np.reshape(D_SYM_COORD_D_COORD_cur_frame_copy, (row, col))
                    D_SYM_COORD_D_COORD_cur_frame_copy = (tf.from_numpy(D_SYM_COORD_D_COORD_cur_frame_copy).to(
                        device)) * (
                                                             -1)
                    D_SYM_COORD_D_COORD_cur_frame_copy = D_SYM_COORD_D_COORD_cur_frame_copy.transpose(0, 1)
                    D_SYM_COORD_D_COORD_cur_frame_copy = tf.reshape(
                        (tf.reshape(D_SYM_COORD_D_COORD_cur_frame_copy,
                                    (col, N_ATOMS_tf[0], parameters.SEL_A_max, 4))) /
                        data_cur[10][frame_idx], (col, row)).transpose(0, 1)
                    FORCE_net_tf_cur[frame_idx] = (-1) * tf.mm(
                        D_E_D_SYM_COORD_cur_batch[frame_idx].reshape(1, len(D_E_D_SYM_COORD_cur_batch[frame_idx])),
                        D_SYM_COORD_D_COORD_cur_frame_copy)
                    MYDLL.freeme(D_SYM_COORD_D_COORD_cur_frame)
                if ((STEP_CUR % 50 == 0)):
                    print(FORCE_net_tf_cur)
                loss_F_cur_batch = CRITERION(FORCE_net_tf_cur, data_cur[3])

                loss_cur_batch = pref_e * loss_E_cur_batch + pref_f * loss_F_cur_batch
                OPTIMIZER.zero_grad()
                loss_cur_batch.backward()
                return loss_cur_batch
            OPTIMIZER.step(closure)
            # correct end
            #correct end
            ###LBFGS end
            """




            END_BATCH_TIMER = time.time()


            ###Adam
            f_out = open("./LOSS.OUT","a")
            print("Epoch: %-10d, Batch: %-10d, lossE: %10.3f eV/atom, lossF: %10.6f eV/A, time: %10.3f s" % (
            epoch, batch_idx, loss_E_cur_batch / N_ATOMS[0], loss_F_cur_batch, END_BATCH_TIMER - START_BATCH_TIMER))
            print("Epoch: %-10d, Batch: %-10d, lossE: %10.3f eV/atom, lossF: %10.3f eV/A, time: %10.3f s" % (\
                epoch, batch_idx, loss_E_cur_batch / N_ATOMS[0], loss_F_cur_batch, END_BATCH_TIMER - START_BATCH_TIMER),\
                  file = f_out)
            f_out.close()
            ###Adam end


            """
            ###LBFGS
            f_out = open("./LOSS.OUT", "a")
            print("Epoch: %-10d, Batch: %-10d, lossE: %10.3f eV/atom, lossF: %10.6f eV/A, time: %10.3f s" % (
            epoch, batch_idx, closure() / N_ATOMS[0], closure() / N_ATOMS[0], END_BATCH_TIMER - START_BATCH_TIMER))
            print("Epoch: %-10d, Batch: %-10d, lossE: %10.3f eV/atom, lossF: %10.3f eV/A, time: %10.3f s" % (\
                epoch, batch_idx, closure() / N_ATOMS[0], closure() / N_ATOMS[0], END_BATCH_TIMER - START_BATCH_TIMER),\
                  file = f_out)
            f_out.close()
            ###LBFGS end
            """


            #print(COORD_Reshape_tf_cur, SYM_COORD_Reshape_tf_cur, ENERGY_tf_cur, FORCE_Reshape_tf_cur, N_ATOMS_tf_cur)

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
"""
###No need to free because here is nearly the end of the program
MYDLL.freeme(read_coord_res)
MYDLL.freeme(read_nei_coord_res)
MYDLL.freeme(read_nei_idx_res)
"""
END_TRAIN_TIMER = time.time()
ELAPSED_TRAIN = END_TRAIN_TIMER - START_TRAIN_TIMER
print("Training complete. Time elapsed: %10.3f s\n"%ELAPSED_TRAIN)
#print(prof.table(sort_by = "cuda_time"))